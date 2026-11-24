from argparse import Namespace
from contextlib import contextmanager, nullcontext, redirect_stdout
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory

from video_lab import (ROOT, SPATIAL, TEMPORAL, blocks, check_output, decoder, encode_frames,
                       ffmpeg, infer_block, model_options, probe, publish, render_spec)


SCHEMA = 1
PAUSED = 75


def add_commands(commands):
    job = commands.add_parser("run", help="start a resumable video job")
    job.add_argument("input", type=Path)
    job.add_argument("output", type=Path)
    job.add_argument("--upscale", choices=sorted(SPATIAL))
    job.add_argument("--interpolate", choices=sorted(TEMPORAL))
    job.add_argument("--order", choices=("interpolate-first", "upscale-first"), default="interpolate-first")
    job.add_argument("--job-dir", type=Path)
    job.add_argument("--scene-threshold", type=float, help="also detect cuts with the given FFmpeg scene score")
    model_options(job)
    for name in ("resume", "status", "pause"):
        commands.add_parser(name).add_argument("job_dir", type=Path)


def show(event, **details):
    print(json.dumps({"event": event, **details}, ensure_ascii=False), flush=True)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, value):
    with TemporaryDirectory(dir=path.parent) as temp:
        staged = Path(temp) / path.name
        with staged.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        staged.replace(path)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def matches(path, checksum):
    return path.is_file() and digest(path) == checksum


def receipt_for(path):
    return path.with_suffix(".json")


def cached(path, expected=None):
    try:
        receipt = read_json(receipt_for(path))
        if (expected is None or receipt.get("inputs") == expected) and matches(path, receipt["sha256"]):
            return receipt["sha256"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


@contextmanager
def exclusive(path):
    # The operating system releases this lock if the process exits unexpectedly.
    with path.open("a+b") as stream:
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"Already running: {path}") from exc
        yield


def scene_cuts(source, video, threshold):
    if threshold is None:
        return []
    if not 0 < threshold < 1:
        raise ValueError("--scene-threshold must be between 0 and 1")
    rate = Fraction(video["avg_frame_rate"])
    filters = (f"settb={rate.denominator}/{rate.numerator},setpts=N,"
               f"select='gt(scene,{threshold})',metadata=mode=print:key=lavfi.scene_score:file=-")
    log = ffmpeg("-i", source, "-map", f"0:{video['index']}", "-vf", filters, "-an", "-f", "null", "-")
    return [int(value) for value in re.findall(rb"pts:\s*(-?\d+)", log)]


def new_plan(args):
    engines = [engine for engine in (args.interpolate, args.upscale) if engine]
    if not engines:
        raise ValueError("Select --upscale, --interpolate, or both")
    if args.size and not args.upscale:
        raise ValueError("--size requires --upscale")
    if args.output.suffix.lower() not in (".mkv", ".mp4"):
        raise ValueError("Use an .mkv or .mp4 destination")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.order == "upscale-first":
        engines.reverse()
    source = args.input.resolve(strict=True)
    input_digest = digest(source)
    video = probe(source, timing=True)["video"]
    total = video["decoded_frames"]
    cuts = sorted(set(args.cut_frame + scene_cuts(source, video, args.scene_threshold)))
    if any(cut < 0 or cut >= total for cut in cuts):
        raise ValueError("Cut frames must be inside the source")
    options = {key: getattr(args, key) for key in ("chunk", "overlap", "tile", "basic_model", "motion_scale", "precision")}
    stages = []
    start, count = args.start_frame, args.frames
    for engine in engines:
        size = args.size if engine in SPATIAL else None
        stage_opts = dict(options, engine=engine, size=size, start_frame=start, frames=count, cut_frame=cuts)
        stop, rate, _, target = render_spec(Namespace(**stage_opts), video)
        factor = 2 if engine in TEMPORAL else 1
        count = (stop - start) * factor
        stages.append(dict(options=stage_opts, rate=str(rate), frames=count, size=target, factor=factor,
                           blocks=list(blocks(start, stop, cuts, args.chunk))))
        cuts = [(cut - start) * factor for cut in cuts if start < cut < stop]
        start = 0
        video = dict(video, width=target[0], height=target[1], avg_frame_rate=str(rate), decoded_frames=count,
                     constant_grid=True, pix_fmt="gbrp16le", color_range="pc")
    first = stages[0]
    full = args.start_frame == 0 and first["frames"] == total * first["factor"]
    from engines import model_files

    weights = {}
    for engine in engines:
        for path in model_files(engine, args.basic_model):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Prepare model files before starting the job: {path}")
            weights[str(path)] = digest(path)
    return dict(schema=SCHEMA, input=str(source), input_sha256=input_digest, output=str(args.output.resolve()),
                copy_tracks=full, weights=weights, stages=stages)


class FrameReader:
    def __init__(self, source, video, stop):
        self.source = source
        self.video = video
        self.stop = stop
        self.process = None
        self.buffer = None
        self.shape = video["height"], video["width"], 3

    def read(self, first, last):
        import numpy as np

        if self.process is None:
            self.process = subprocess.Popen(decoder(self.source, self.video, first, self.stop), stdout=subprocess.PIPE)
            self.first = self.cursor = first
        if first < self.first:
            raise ValueError("Frame requests must advance in source order")
        keep = self.buffer[max(0, first-self.first):] if self.buffer is not None else None
        frame_bytes = self.shape[0] * self.shape[1] * 6
        while self.cursor < first:
            if len(self.process.stdout.read(frame_bytes)) != frame_bytes:
                raise ValueError(f"Decoder ended at frame {self.cursor}")
            self.cursor += 1
        if last > self.cursor:
            raw = self.process.stdout.read((last-self.cursor) * frame_bytes)
            if len(raw) != (last-self.cursor) * frame_bytes:
                raise ValueError(f"Decoder ended before frame {last}")
            new = np.frombuffer(raw, dtype="<u2").reshape(-1, *self.shape)
            keep = np.concatenate((keep, new)) if keep is not None and len(keep) else new
            self.cursor = last
        if keep is None:
            raise ValueError("Empty frame request")
        self.first, self.buffer = first, keep
        return keep[:last-first]

    def close(self):
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.stdout.close()
            self.process.wait()


def stage_directory(job, index):
    return job / f"stage-{index + 1}"


def stage_paths(directory, stage):
    return [directory / f"block-{i:06d}.mkv" for i in range(len(stage["blocks"]))]


def assemble(directory, paths, stage, output):
    playlist = directory / "blocks.ffconcat"
    playlist.write_text("ffconcat version 1.0\n" + "".join(f"file '{p.name}'\n" for p in paths), encoding="ascii")
    rate = Fraction(stage["rate"])
    vf = f"settb={rate.denominator}/{rate.numerator},setpts=N"
    if output.suffix == ".mp4":
        vf += ",scale=out_color_matrix=bt709:out_range=limited,format=yuv420p"
        codec = ("-c:v", "libx264", "-preset", "slow", "-crf", "17", "-colorspace", "bt709", "-color_range", "tv")
    else:
        codec = ("-c:v", "ffv1", "-color_range", "pc")
    ffmpeg("-xerror", "-f", "concat", "-safe", "1", "-i", playlist, "-map", "0:v:0", "-an",
           "-vf", vf, "-r", str(rate), "-fps_mode", "cfr", "-color_primaries", "bt709", "-color_trc", "bt709",
           *codec, output)
    check_output(output, rate, stage["frames"], tuple(stage["size"]))


def run_stage(job, index):
    plan = read_json(job / "job.json")
    stage = plan["stages"][index]
    args = Namespace(**stage["options"])
    if args.size:
        args.size = tuple(args.size)
    directory = stage_directory(job, index)
    directory.mkdir(exist_ok=True)
    source = Path(plan["input"]) if index == 0 else stage_directory(job, index-1) / "assembled.mkv"
    source_checksum = plan["input_sha256"] if index == 0 else read_json(receipt_for(source))["sha256"]
    paths = stage_paths(directory, stage)
    with exclusive(directory / "worker.lock"):
        checksums = [cached(path, source_checksum) for path in paths]
        needed = [i for i, checksum in enumerate(checksums) if not checksum]
        show("stage", stage=index + 1, engine=args.engine, reused=len(paths)-len(needed), total=len(paths))
        if (job / "pause.request").exists():
            return PAUSED
        if needed:
            input_rate = plan["stages"][index-1]["rate"] if index else None
            video = probe(source, timing=True, expected_rate=input_rate)["video"]
            if input_rate:
                video["avg_frame_rate"] = input_rate
            stop, rate, internal_size, size = render_spec(args, video)
            reader = FrameReader(source, video, stop)
            try:
                lease = nullcontext() if args.engine in ("hold", "lanczos") else exclusive(ROOT / ".runtime" / "gpu.lock")
                with lease:
                    with redirect_stdout(sys.stderr):
                        from engines import make_engine

                        engine = make_engine(args, Fraction(video["avg_frame_rate"]))
                    for i in needed:
                        if (job / "pause.request").exists():
                            return PAUSED
                        left, right, first, last = stage["blocks"][i]
                        path = paths[i]
                        with TemporaryDirectory(dir=directory) as temp:
                            staged = Path(temp) / path.name
                            frames = infer_block(args, reader.read, engine, internal_size, left, right, first, last)
                            with redirect_stdout(sys.stderr):
                                encode_frames(staged, frames, rate, internal_size, size)
                            check_output(staged, rate, (last-first)*stage["factor"], size)
                            checksums[i] = digest(staged)
                            staged.replace(path)
                            write_json(receipt_for(path), {"sha256": checksums[i], "inputs": source_checksum})
                        show("block", stage=index+1, block=i+1, total=len(paths), first=first, stop=last)
            finally:
                reader.close()
        if (job / "pause.request").exists():
            return PAUSED
        if index + 1 < len(plan["stages"]):
            intermediate = directory / "assembled.mkv"
            if not cached(intermediate, checksums):
                with TemporaryDirectory(dir=directory) as temp:
                    staged = Path(temp) / intermediate.name
                    assemble(directory, paths, stage, staged)
                    checksum = digest(staged)
                    staged.replace(intermediate)
                    write_json(receipt_for(intermediate), {"sha256": checksum, "inputs": checksums})


def finish(job, plan):
    destination = Path(plan["output"])
    source = Path(plan["input"])
    stage = plan["stages"][-1]
    directory = stage_directory(job, len(plan["stages"])-1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temp:
        staged = Path(temp) / ("video" + destination.suffix.lower())
        assemble(directory, stage_paths(directory, stage), stage, staged)
        if plan["copy_tracks"]:
            input_tracks = probe(source)
            offset = -Fraction(input_tracks["video"].get("start_time", "0"))
            muxed = Path(temp) / ("muxed" + staged.suffix)
            ffmpeg("-copyts", "-i", staged, "-itsoffset", f"{float(offset):.6f}", "-i", source,
                   "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?", "-map_chapters", "-1",
                   "-avoid_negative_ts", "disabled", "-c", "copy", muxed)
            check_output(muxed, stage["rate"], stage["frames"], tuple(stage["size"]))
            output_tracks = probe(muxed)
            for kind in ("audio", "subtitles"):
                if [s["codec_name"] for s in input_tracks[kind]] != [s["codec_name"] for s in output_tracks[kind]]:
                    raise ValueError(f"The output {kind} streams differ from the source")
            staged = muxed
        write_json(job / "result.json", {"sha256": digest(staged)})
        publish(staged, destination)


def execute_job(job, plan, *, resume):
    if plan["schema"] != SCHEMA:
        raise ValueError("Unsupported job manifest version")
    destination = Path(plan["output"])
    result = job / "result.json"
    if destination.exists():
        if not result.is_file() or not matches(destination, read_json(result)["sha256"]):
            raise FileExistsError(destination)
        show("complete", job=str(job), output=str(destination), reused=True)
        return
    if resume:
        if not matches(Path(plan["input"]), plan["input_sha256"]):
            raise ValueError("Source file changed since the job was planned")
        for path, checksum in plan["weights"].items():
            if not matches(Path(path), checksum):
                raise ValueError(f"Model file changed since the job was planned: {path}")
    for index, _ in enumerate(plan["stages"]):
        worker = subprocess.run([sys.executable, "-X", "utf8", "-u", __file__, str(job), str(index)])
        if worker.returncode == PAUSED or (job / "pause.request").exists() and worker.returncode == 0:
            show("paused", job=str(job), stage=index+1)
            return
        worker.check_returncode()
    finish(job, plan)
    show("complete", job=str(job), output=str(destination))


def dispatch(args):
    job = (args.job_dir or args.output.with_name(args.output.name + ".job")).resolve() if args.action == "run" else args.job_dir.resolve()
    manifest = job / "job.json"
    if args.action == "pause":
        read_json(manifest)
        (job / "pause.request").touch()
        show("pause-requested", job=str(job))
        return
    if args.action == "status":
        plan = read_json(manifest)
        counts = []
        for index, stage in enumerate(plan["stages"]):
            paths = stage_paths(stage_directory(job, index), stage)
            counts.append(dict(engine=stage["options"]["engine"], total=len(paths),
                               committed=sum(path.is_file() and receipt_for(path).is_file() for path in paths)))
        show("status", job=str(job), output=plan["output"], stages=counts,
             pause_requested=(job / "pause.request").is_file(),
             complete=(job / "result.json").is_file() and Path(plan["output"]).is_file())
        return
    if args.action == "run":
        job.mkdir(parents=True, exist_ok=True)
    with exclusive(job / "job.lock"):
        if args.action == "run":
            if manifest.exists():
                raise FileExistsError(f"Job already exists; use resume: {job}")
            plan = new_plan(args)
            write_json(manifest, plan)
            show("job", job=str(job), output=plan["output"])
        else:
            plan = read_json(manifest)
            (job / "pause.request").unlink(missing_ok=True)
        execute_job(job, plan, resume=args.action == "resume")


if __name__ == "__main__":
    raise SystemExit(run_stage(Path(sys.argv[1]), int(sys.argv[2])))
