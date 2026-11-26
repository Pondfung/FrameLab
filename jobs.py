from argparse import ArgumentTypeError, Namespace
from bisect import bisect_right
from contextlib import contextmanager, nullcontext, redirect_stdout
from fractions import Fraction
from math import ceil
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic

from video_lab import (NVIDIA_VFX, NVENC, ROOT, SPATIAL, TEMPORAL, blocks, cfr_filter, check_output, command, decoder,
                       as_bytes, decode, encode_frames, ffmpeg, infer_block, model_options, output_index, probe, publish,
                       render_spec)


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
    job.add_argument("--fps", type=parse_rate, help="target CFR, e.g. 60 or 24000/1001")
    job.add_argument("--scene-threshold", type=float, help="also detect cuts with the given FFmpeg scene score")
    job.add_argument("--vfg-auto-cuts", action="store_true",
                     help="scan for hard cuts before VFG and disable the NVIDIA SDK shot detector")
    job.add_argument("--vfg-review", action="store_true",
                     help="record suspected VFG anomalies before VSR or encoding (report only)")
    job.add_argument("--vfg-retry", action="store_true",
                     help="retry a near-copy once in reverse and use it only after motion verification")
    model_options(job)
    for name in ("resume", "status", "pause"):
        commands.add_parser(name).add_argument("job_dir", type=Path)
    for name, help_text in (
        ("review-slots", "locate frames in a finished VFG MP4 job"),
        ("review-repair", "make a localized comparison copy around a marked output frame"),
        ("review-import", "repair a marked frame in an imported CFR video"),
    ):
        review = commands.add_parser(name, help=help_text)
        review.add_argument("input" if name == "review-import" else "job_dir", type=Path)
        review.add_argument("first", type=int, help="first zero-based output frame")
        if name != "review-import":
            review.add_argument("--last", type=int, help="last output frame, inclusive")
            review.add_argument("--infer-range", action="store_true", help="expand a single mark using the VFG report")
        review.add_argument("--label", choices=("echo", "freeze", "ghosting"), action="append", default=[])
        if name != "review-slots":
            review.add_argument("output", type=Path)
            review.add_argument("--choice", choices=("keep", "reverse", "rife", "gimm-vfi"), required=True)
            review.add_argument("--motion", choices=("normal", "fast"))


def parse_rate(value):
    try:
        rate = Fraction(value)
    except (ValueError, ZeroDivisionError):
        raise ArgumentTypeError("Use a decimal or rational frame rate") from None
    if not 1 <= rate <= 240 or rate.denominator > 1_000_000:
        raise ArgumentTypeError("Target frame rate must be between 1 and 240 fps (up to six decimal places)")
    return rate


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


def cached(path, expected=None, *, review_version=None, retry_version=None):
    try:
        receipt = read_json(receipt_for(path))
        if review_version is not None:
            review = receipt.get("vfg_review", {})
            if (review.get("version"), review.get("retry_version")) != (review_version, retry_version):
                return None
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
    # setpts=N makes each reported PTS the zero-based source index of the new shot.
    # These FFmpeg boundaries are independent of the VFG SDK shot detector.
    rate = Fraction(video["avg_frame_rate"])
    filters = (f"settb={rate.denominator}/{rate.numerator},setpts=N,"
               f"select='gt(scene,{threshold})',metadata=mode=print:key=lavfi.scene_score:file=-")
    log = ffmpeg("-i", source, "-map", f"0:{video['index']}", "-vf", filters, "-an", "-f", "null", "-")
    return [int(value) for value in re.findall(rb"pts:\s*(-?\d+)", log)]


def frame_at_rate(frame, rate, target):
    # FFmpeg's fps filter rounds positive timestamps to the nearest frame, with ties rounded up.
    return int(frame * target / rate + Fraction(1, 2))


def nvenc_available(size, rate):
    width, height = size
    return subprocess.run(
        command("ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
                "-f", "lavfi", "-i", f"color=size={width}x{height}:rate={rate}",
                "-frames:v", "1", *NVENC, "-bf", "0", "-video_track_timescale", rate.numerator,
                "-f", "mp4", "-movflags", "frag_keyframe+empty_moov", "pipe:1"),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def sampled_index(tick, rate, target):
    return ceil((tick + Fraction(1, 2)) * rate / target) - 1


def sample_block(frames, first, last, rate, target):
    # The fps filter rounds each input PTS and emits the last frame at or before
    # each output tick. Use absolute indices so cuts and blocks keep that grid.
    out_first = frame_at_rate(first, rate, target)
    out_last = frame_at_rate(last, rate, target)
    frames = iter(frames)
    position = first - 1
    for tick in range(out_first, out_last):
        wanted = sampled_index(tick, rate, target)
        while position < wanted:
            frame = next(frames)
            position += 1
        yield frame


def set_memory_inputs(stages):
    # Bound application-owned frame buffers. Model activations are managed by
    # each backend. Reserve room for float outputs, RGB48 copies and lookahead.
    final = stages[-1]
    buffered_stages = stages[:-1] if final["options"]["engine"] == "nvidia-vsr" else stages
    buffered_bytes = sum(width * height * 48 for width, height in (s["size"] for s in buffered_stages))
    max_chunk = max(1, (512 * 1024**2) // buffered_bytes)
    for stage in buffered_stages:
        options = stage["options"]
        options["chunk"] = min(options["chunk"], max_chunk)
        stage["blocks"] = list(blocks(options["start_frame"], stage["blocks"][-1][3],
                                      options["cut_frame"], options["chunk"]))
    final["memory_inputs"] = stages[:-1]
    return [final]


def new_plan(args):
    target_rate = getattr(args, "fps", None)
    if not (args.interpolate or args.upscale or target_rate):
        raise ValueError("Select --upscale, --interpolate, or --fps")
    if args.size and not args.upscale:
        raise ValueError("--size requires --upscale")
    if args.output.suffix.lower() not in (".mkv", ".mp4"):
        raise ValueError("Use an .mkv or .mp4 destination")
    if args.output.exists():
        raise FileExistsError(args.output)
    source = args.input.resolve(strict=True)
    input_digest = digest(source)
    video = probe(source, timing=True, allow_tail_gap=True)["video"]
    native_rate = Fraction(video["avg_frame_rate"])
    engines = []
    if args.interpolate:
        if target_rate is None:
            engines.append(args.interpolate)
        elif args.interpolate == "nvidia-vfg":
            if target_rate > native_rate:
                engines.append(args.interpolate)
        elif args.interpolate != "hold":
            working_rate = native_rate
            while working_rate < target_rate:
                engines.append(args.interpolate)
                working_rate *= 2
                if len(engines) > 8:
                    raise ValueError("Target needs more than eight interpolation passes")
    if args.upscale:
        engines.insert(0 if args.order == "upscale-first" else len(engines), args.upscale)
    if not engines:
        # Keep the same lossless block/checkpoint path for FPS-only exports.
        engines.append("lanczos")
    total = video["decoded_frames"]
    manual_cuts = sorted(set(args.cut_frame))
    auto_cuts = getattr(args, "vfg_auto_cuts", False)
    if auto_cuts and "nvidia-vfg" not in engines:
        raise ValueError("--vfg-auto-cuts needs VFG interpolation above the source frame rate")
    retry = getattr(args, "vfg_retry", False)
    review = retry or getattr(args, "vfg_review", False)
    if review and "nvidia-vfg" not in engines:
        raise ValueError("VFG review and retry need VFG interpolation above the source frame rate")
    cuts = sorted(set(manual_cuts + scene_cuts(source, video, args.scene_threshold)))
    if any(cut < 0 or cut >= total for cut in cuts):
        raise ValueError("Cut frames must be inside the source")
    cut_detection = None
    if auto_cuts:
        from scene_detection import scan_cuts

        show("phase", phase="scan_cuts")
        cut_detection = scan_cuts(source, video)
        cut_detection.update(manual=manual_cuts, scene_threshold=args.scene_threshold,
                             threshold_cuts=sorted(set(cuts) - set(manual_cuts)))
        cuts = sorted(set(cuts + cut_detection["detected"]))
        cut_detection["effective"] = cuts
    options = {key: getattr(args, key) for key in ("chunk", "overlap", "tile", "basic_model", "motion_scale", "precision")}
    stages = []
    start, count = args.start_frame, args.frames
    for engine in engines:
        input_rate = Fraction(video["avg_frame_rate"])
        retime = (target_rate is not None and target_rate < input_rate and engine in SPATIAL
                  and bool(stages) and stages[-1]["options"]["engine"] in TEMPORAL)
        if retime:
            count = max(1, frame_at_rate(count, input_rate, target_rate))
            cuts = sorted({frame_at_rate(cut, input_rate, target_rate) for cut in cuts})
            video = dict(video, avg_frame_rate=str(target_rate), decoded_frames=count)
        size = ((args.size or (video["width"], video["height"])) if engine == "lanczos"
                else args.size if engine in SPATIAL else None)
        stage_opts = dict(options, engine=engine, size=size, start_frame=start, frames=count, cut_frame=cuts)
        if engine == "nvidia-vfg" and target_rate is not None:
            stage_opts["target_rate"] = str(target_rate)
        if engine == "nvidia-vfg":
            stage_opts["vfg_sdk_shot_detection"] = not auto_cuts and getattr(args, "vfg_sdk_shot_detection", True)
        if engine == "nvidia-vsr":
            stage_opts["overlap"] = 0
        stop, rate, _, target = render_spec(Namespace(**stage_opts), video)
        if retime:
            input_rate = target_rate
        if not stages:
            full = start == 0 and stop == total
        count = output_index(stop, start, stop, input_rate, rate)
        stages.append(dict(options=stage_opts, rate=str(rate), frames=count, size=target,
                           blocks=list(blocks(start, stop, cuts, args.chunk))))
        if review and engine == "nvidia-vfg":
            from vfg_review import VERSION

            review_options = dict(version=VERSION, input_rate=str(input_rate), source_offset=args.start_frame - start)
            if retry:
                from vfg_retry import VERSION as RETRY_VERSION, motion_backend

                motion_backend()
                review_options["retry_version"] = RETRY_VERSION
            stages[-1]["vfg_review"] = review_options
        if rate % input_rate == 0:
            stages[-1]["factor"] = int(rate / input_rate)
        if retime:
            stages[-1]["retime_input"] = True
        cuts = [output_index(cut, start, stop, input_rate, rate) for cut in cuts if start < cut < stop]
        start = 0
        video = dict(video, width=target[0], height=target[1], avg_frame_rate=str(rate), decoded_frames=count,
                     constant_grid=True, pix_fmt="gbrp16le", color_range="pc")
    from engines import model_files

    weights = {}
    for engine in engines:
        for path in model_files(engine, args.basic_model):
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"Prepare model files before starting the job: {path}")
            weights[str(path)] = digest(path)
    final = stages[-1]
    output_rate = target_rate or Fraction(final["rate"])
    output_frames = max(1, frame_at_rate(final["frames"], Fraction(final["rate"]), output_rate))
    nvidia_chain = engines == ["nvidia-vfg", "nvidia-vsr"]
    nvidia_single = (len(engines) == 1 and engines[0] in NVIDIA_VFX
                     and output_rate == Fraction(final["rate"]))
    if args.output.suffix.lower() == ".mp4" and (nvidia_chain or nvidia_single):
        first = stages[0]
        start, stop = first["options"]["start_frame"], first["blocks"][-1][3]
        # A target barely above the source can round the final block to zero frames.
        spans = [span for span in first["blocks"]
                 if output_index(span[2], start, stop, native_rate, output_rate) < output_frames]
        stages = [dict(first, size=final["size"], blocks=spans, nvenc_blocks=True)]
    elif args.output.suffix.lower() == ".mp4":
        model_rate = Fraction(final["rate"])
        temporal_only = not args.upscale and args.interpolate in ("gimm-vfi", "rife") and engines[-1] == args.interpolate
        spatial_only = (not args.interpolate and len(engines) == 1
                        and engines[0] in ("basicvsrpp", "seedvr2", "seedvr2-7b") and output_rate == model_rate)
        rife_vsr = engines[-1] == "nvidia-vsr" and engines[:-1] == ["rife"] * (len(engines) - 1) and len(engines) > 1
        if (temporal_only or spatial_only or rife_vsr) and nvenc_available(final["size"], output_rate):
            final["nvenc_blocks"] = True
            if output_rate != model_rate:
                final["encode_rate"] = str(output_rate)
            if rife_vsr or (temporal_only and len(stages) > 1):
                stages = set_memory_inputs(stages)
    plan = dict(schema=SCHEMA, input=str(source), input_sha256=input_digest, output=str(args.output.resolve()),
                copy_tracks=full, weights=weights, stages=stages, stream_blocks=True,
                output_rate=str(output_rate), output_frames=output_frames, nvenc=bool(NVIDIA_VFX.intersection(engines)))
    if cut_detection is not None:
        plan["cut_detection"] = cut_detection
    return plan


class FrameReader:
    def __init__(self, source, video, stop, *, playlist=False, retime=None, bounded_reads=False):
        self.source = source
        self.video = video
        self.stop = stop
        self.playlist = playlist
        self.retime = retime
        self.bounded_reads = bounded_reads
        self.process = None
        self.buffer = None
        self.shape = video["height"], video["width"], 3

    def read_bytes(self, size):
        pipe = self.process.stdout
        if not self.bounded_reads:
            return pipe.read(size)
        # Large reads on Windows pipes can take several seconds per block.
        raw = bytearray(size)
        view = memoryview(raw)
        offset = 0
        while offset < size:
            amount = pipe.readinto(view[offset:offset + 1024**2])
            if not amount:
                break
            offset += amount
        return view[:offset]

    def read(self, first, last):
        import numpy as np

        if self.process is None:
            self.process = subprocess.Popen(
                decoder(self.source, self.video, first, self.stop, playlist=self.playlist, retime=self.retime),
                stdout=subprocess.PIPE)
            self.first = self.cursor = first
        if first < self.first:
            raise ValueError("Frame requests must advance in source order")
        keep = self.buffer[max(0, first-self.first):] if self.buffer is not None else None
        frame_bytes = self.shape[0] * self.shape[1] * 6
        while self.cursor < first:
            if len(self.read_bytes(frame_bytes)) != frame_bytes:
                raise ValueError(f"Decoder ended at frame {self.cursor}")
            self.cursor += 1
        if last > self.cursor:
            raw = self.read_bytes((last-self.cursor) * frame_bytes)
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


class TemporalReader:
    def __init__(self, upstream, stage, engine, rate):
        self.upstream = upstream
        self.args = Namespace(**stage["options"])
        self.engine = engine
        self.rate = rate
        self.spans = stage["blocks"]
        self.stop = self.spans[-1][3]
        self.ends = [span[3] for span in self.spans]
        self.size = tuple(stage["size"])
        self.block = None

    def read(self, first, last):
        import numpy as np

        width, height = self.size
        result = np.empty((last - first, height, width, 3), dtype="<u2")
        start = self.args.start_frame
        cursor = first
        while cursor < last:
            index = bisect_right(self.ends, cursor // 2 + start)
            if self.block != index:
                left, right, begin, end = self.spans[index]
                frames = infer_block(self.args, self.upstream, self.engine, self.size, left, right, begin, end,
                                     input_rate=self.rate / 2, output_rate=self.rate, stop=self.stop)
                self.block_first = (begin - start) * 2
                self.buffer = np.empty(((end - begin) * 2, height, width, 3), dtype="<u2")
                for offset, frame in enumerate(frames):
                    if frame.dtype.kind == "f":
                        frame = np.frombuffer(as_bytes(frame), dtype="<u2").reshape(height, width, 3)
                    self.buffer[offset] = frame
                self.block = index
            offset = cursor - self.block_first
            count = min(last - cursor, len(self.buffer) - offset)
            result[cursor-first:cursor-first+count] = self.buffer[offset:offset+count]
            cursor += count
        return result


def retimed_reader(read, rate, target, total):
    def at_rate(first, last):
        # A request is scoped to one output block. Match the existing fps
        # filter's global grid and last-frame extension.
        indices = [min(total - 1, sampled_index(tick, rate, target)) for tick in range(first, last)]
        origin = indices[0]
        return read(origin, indices[-1] + 1)[[index - origin for index in indices]]

    return at_rate


def stage_directory(job, index):
    return job / f"stage-{index + 1}"


def stage_paths(directory, stage):
    suffix = ".mp4" if stage.get("nvenc_blocks") else ".mkv"
    return [directory / f"block-{i:06d}{suffix}" for i in range(len(stage["blocks"]))]


def make_playlist(directory, paths):
    playlist = directory / "blocks.ffconcat"
    playlist.write_text("ffconcat version 1.0\n" + "".join(f"file '{p.name}'\n" for p in paths), encoding="ascii")
    return playlist


def final_encode(frames, *args):
    with subprocess.Popen(command("ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
                                  "-progress", "pipe:1", *args), stdout=subprocess.PIPE, text=True) as process:
        last = monotonic()
        for line in process.stdout:
            key, _, value = line.strip().partition("=")
            if key == "frame" and value.isdecimal() and monotonic() - last >= 1:
                show("phase", phase="encode", done=int(value), total=frames)
                last = monotonic()
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, process.args)


def assemble(directory, paths, stage, output, *, output_rate=None, output_frames=None, final=False, nvenc=False):
    playlist = make_playlist(directory, paths)
    rate = Fraction(stage["rate"])
    target = Fraction(output_rate or rate)
    count = output_frames or stage["frames"]
    vf = cfr_filter(rate, target, count)
    if output.suffix == ".mp4":
        vf += ",scale=out_color_matrix=bt709:out_range=limited,format=yuv420p"
        codec = ("-c:v", "libx264", "-preset", "slow", "-crf", "17", "-colorspace", "bt709", "-color_range", "tv")
        if nvenc:
            codec = NVENC
    else:
        codec = ("-c:v", "ffv1", "-color_range", "pc")
    options = ("-xerror", "-f", "concat", "-safe", "1", "-i", playlist, "-map", "0:v:0", "-an",
               "-vf", vf, "-r", str(target), "-fps_mode", "cfr", "-color_primaries", "bt709", "-color_trc", "bt709",
               *codec, output)
    if final:
        show("phase", phase="encode", done=0, total=count)
        final_encode(count, *options)
        show("phase", phase="verify")
    else:
        ffmpeg(*options)
    check_output(output, target, count, tuple(stage["size"]))


def join_nvenc_blocks(directory, paths, rate, output):
    playlist = make_playlist(directory, paths)
    rate = Fraction(rate)
    # MP4 segment durations round to microseconds in the concat demuxer.
    # Assign the exact CFR grid from packet order before writing the final MP4.
    timestamp = f"N*{rate.denominator}"
    timing = (f"setts=pts={timestamp}:dts={timestamp}:duration={rate.denominator}"
              f":time_base=1/{rate.numerator}")
    ffmpeg("-xerror", "-f", "concat", "-safe", "1", "-i", playlist, "-map", "0:v:0",
           "-c:v", "copy", "-bsf:v", timing, "-video_track_timescale", rate.numerator, output)


def stage_source(job, plan, index):
    if index == 0:
        return Path(plan["input"]), plan["input_sha256"], None
    previous = plan["stages"][index - 1]
    directory = stage_directory(job, index - 1)
    if not plan.get("stream_blocks"):
        source = directory / "assembled.mkv"
        return source, read_json(receipt_for(source))["sha256"], None
    paths = stage_paths(directory, previous)
    hashes = b"".join(bytes.fromhex(read_json(receipt_for(path))["sha256"]) for path in paths)
    stage = plan["stages"][index]
    if stage.get("retime_input"):
        hashes += f"fps:{stage['rate']}:{stage['frames']}".encode("ascii")
    video = dict(index=0, width=previous["size"][0], height=previous["size"][1], avg_frame_rate=previous["rate"],
                 decoded_frames=previous["frames"], constant_grid=True, pix_fmt="gbrp16le", color_range="pc")
    return make_playlist(directory, paths), hashlib.sha256(hashes).hexdigest(), video


def configure_nvidia_worker():
    log_dir = ROOT / ".runtime" / "vfx-logs"
    log_dir.mkdir(exist_ok=True)
    os.chdir(log_dir)
    # Keep Python's JSON stream on a duplicate before redirecting native SDK logs.
    stdout = sys.stdout.fileno()
    sys.stdout = os.fdopen(os.dup(stdout), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), stdout)
    if os.name == "nt":
        import ctypes
        import msvcrt

        handle = ctypes.c_void_p(msvcrt.get_osfhandle(stdout))
        if not ctypes.windll.kernel32.SetStdHandle(-11, handle):  # STD_OUTPUT_HANDLE
            raise ctypes.WinError()


def run_stage(job, index):
    plan = read_json(job / "job.json")
    stage = plan["stages"][index]
    args = Namespace(**stage["options"])
    if args.size:
        args.size = tuple(args.size)
    directory = stage_directory(job, index)
    directory.mkdir(exist_ok=True)
    source, source_checksum, video = stage_source(job, plan, index)
    paths = stage_paths(directory, stage)
    nvenc_blocks = stage.get("nvenc_blocks", False)
    with exclusive(directory / "worker.lock"):
        show("phase", phase="cache", stage=index+1)
        review_options = stage.get("vfg_review")
        review_version = retry_version = None
        if review_options is not None:
            from vfg_review import BlockReview, SUPPORTED_VERSIONS

            review_version = review_options["version"]
            if review_version not in SUPPORTED_VERSIONS:
                raise ValueError("Unsupported VFG review version for this job")
            retry_version = review_options.get("retry_version")
            if retry_version is not None:
                from vfg_retry import VERSION as RETRY_VERSION

                if retry_version != RETRY_VERSION:
                    raise ValueError("Unsupported VFG retry version for this job")
        repair = stage.get("repair_slots", {})
        method = repair.get("method", "reverse")
        by_block = repair.get("blocks", stage.get("reverse_slots", {}))
        block_slots = [by_block.get(str(i), []) for i in range(len(paths))]
        inputs = [dict(source=source_checksum, **{method: slots}) if slots else source_checksum
                  for slots in block_slots]
        checksums = [cached(path, expected, review_version=review_version, retry_version=retry_version)
                     for path, expected in zip(paths, inputs)]
        needed = [i for i, checksum in enumerate(checksums) if not checksum]
        show("stage", stage=index + 1, engine=args.engine, reused=len(paths)-len(needed), total=len(paths))
        if (job / "pause.request").exists():
            return PAUSED
        if needed:
            input_rate = plan["stages"][index-1]["rate"] if index else None
            playlist = video is not None
            if video is None:
                show("phase", phase="probe", stage=index+1)
                video = probe(source, timing=True, expected_rate=input_rate, allow_tail_gap=index == 0)["video"]
            if input_rate:
                video["avg_frame_rate"] = input_rate
            memory_inputs = stage.get("memory_inputs", [])
            source_video = video
            if memory_inputs:
                previous = memory_inputs[-1]
                video = dict(video, avg_frame_rate=previous["rate"], decoded_frames=previous["frames"],
                             pix_fmt="gbrp16le", color_range="pc")
            retime = Fraction(stage["rate"]) if stage.get("retime_input") else None
            model_video = (dict(video, avg_frame_rate=stage["rate"], decoded_frames=stage["frames"])
                           if retime is not None else video)
            stop, rate, internal_size, size = render_spec(args, model_video)
            model_rate = Fraction(model_video["avg_frame_rate"])
            # A fused manifest stores the VFG options and the larger VSR output size.
            chain_size = None
            if nvenc_blocks and args.engine == "nvidia-vfg" and tuple(stage["size"]) != size:
                chain_size = internal_size = size = tuple(stage["size"])
            if memory_inputs:
                reader = FrameReader(source, source_video, memory_inputs[0]["blocks"][-1][3], bounded_reads=True)
            else:
                reader = FrameReader(source, video, stop, playlist=playlist, retime=retime, bounded_reads=nvenc_blocks)
            read = reader.read
            try:
                lease = nullcontext() if args.engine in ("hold", "lanczos") else exclusive(ROOT / ".runtime" / "gpu.lock")
                with lease:
                    if args.engine in NVIDIA_VFX:
                        configure_nvidia_worker()
                    show("phase", phase="load", stage=index+1)
                    with redirect_stdout(sys.stderr):
                        from engines import make_engine, nvidia_vfg

                        engine = (nvidia_vfg(args, upscale_size=chain_size) if chain_size
                                  else make_engine(args, model_rate))
                        if method != "reverse":
                            from engines import local_interpolator

                            replacement = local_interpolator(method, args, model_rate)
                        else:
                            replacement = None
                        for previous in memory_inputs:
                            rate_in_memory = Fraction(previous["rate"])
                            # GIMM's model operates on pairs without an input-rate parameter.
                            temporal = (engine if args.engine == "gimm-vfi"
                                        else make_engine(Namespace(**previous["options"]), rate_in_memory / 2))
                            read = TemporalReader(read, previous, temporal, rate_in_memory).read
                        if memory_inputs and retime:
                            read = retimed_reader(read, Fraction(video["avg_frame_rate"]), retime, video["decoded_frames"])
                    show("phase", phase="infer_nvenc" if chain_size else "infer", stage=index+1)
                    for committed, i in enumerate(needed, start=len(paths)-len(needed)+1):
                        if (job / "pause.request").exists():
                            return PAUSED
                        left, right, first, last = stage["blocks"][i]
                        path = paths[i]
                        first_model = output_index(first, args.start_frame, stop, model_rate, rate)
                        last_model = output_index(last, args.start_frame, stop, model_rate, rate)
                        review = (BlockReview(right, stop, first, first_model, rate, review_options["source_offset"],
                                              version=review_version, retry_version=retry_version)
                                  if review_options is not None else None)
                        with TemporaryDirectory(dir=directory) as temp:
                            staged = Path(temp) / path.name
                            if memory_inputs and args.engine == "nvidia-vsr":
                                # Write each enlarged frame before computing the next one.
                                # The encoded checkpoint can therefore be larger than the RAM window.
                                frames = (frame for pos in range(first, last) for frame in
                                          infer_block(args, read, engine, internal_size, left, right, pos, pos+1,
                                                      input_rate=model_rate, output_rate=rate, stop=stop))
                            else:
                                frames = infer_block(args, read, engine, internal_size, left, right, first, last,
                                                     input_rate=model_rate, output_rate=rate, stop=stop, review=review,
                                                     repair_slots=block_slots[i], replacement=replacement)
                            encode_rate = Fraction(stage.get("encode_rate", rate))
                            count = last_model - first_model
                            if encode_rate != rate:
                                frames = sample_block(frames, first_model, last_model, rate, encode_rate)
                                count = (frame_at_rate(last_model, rate, encode_rate)
                                         - frame_at_rate(first_model, rate, encode_rate))
                            with redirect_stdout(sys.stderr):
                                encode_frames(staged, frames, encode_rate, internal_size, size, nvenc=nvenc_blocks)
                            check_output(staged, encode_rate, count, size)
                            checksums[i] = digest(staged)
                            staged.replace(path)
                            receipt = dict(sha256=checksums[i], inputs=inputs[i])
                            if review is not None:
                                receipt["vfg_review"] = dict(version=review_version, **review.summary())
                            write_json(receipt_for(path), receipt)
                        show("block", stage=index+1, block=i+1, committed=committed, total=len(paths), first=first, stop=last)
            finally:
                reader.close()
        if (job / "pause.request").exists():
            return PAUSED
        if index + 1 < len(plan["stages"]) and not plan.get("stream_blocks"):
            intermediate = directory / "assembled.mkv"
            if not cached(intermediate, checksums):
                show("phase", phase="assemble", stage=index+1)
                with TemporaryDirectory(dir=directory) as temp:
                    staged = Path(temp) / intermediate.name
                    assemble(directory, paths, stage, staged)
                    checksum = digest(staged)
                    staged.replace(intermediate)
                    write_json(receipt_for(intermediate), {"sha256": checksum, "inputs": checksums})


def write_review(job, plan):
    for index, stage in enumerate(plan["stages"]):
        options = stage.get("vfg_review")
        if options is None:
            continue
        from vfg_review import report

        args = stage["options"]
        offset = options["source_offset"]
        spans = stage["blocks"]
        cuts = [cut + offset for cut in args["cut_frame"] if spans[0][0] < cut < spans[-1][1]]
        paths = stage_paths(stage_directory(job, index), stage)
        details = report((read_json(receipt_for(path))["vfg_review"] for path in paths),
                         stage=index+1, input_rate=Fraction(options["input_rate"]), output_rate=stage["rate"],
                         source_start=args["start_frame"] + offset, cuts=cuts,
                         sdk_shot_detection=args["vfg_sdk_shot_detection"], version=options["version"],
                         retry_version=options.get("retry_version"))
        path = job / "vfg-review.json"
        write_json(path, details)
        show("vfg-review", path=str(path), generated=details["roles"]["generated"],
             suspects=len(details["suspects"]))


def finish(job, plan):
    destination = Path(plan["output"])
    source = Path(plan["input"])
    stage = plan["stages"][-1]
    rate = plan.get("output_rate", stage["rate"])
    frames = plan.get("output_frames", stage["frames"])
    directory = stage_directory(job, len(plan["stages"])-1)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temp:
        staged = Path(temp) / ("video" + destination.suffix.lower())
        paths = stage_paths(directory, stage)
        nvenc_blocks = stage.get("nvenc_blocks", False)
        if nvenc_blocks:
            show("phase", phase="join_nvenc")
            join_nvenc_blocks(directory, paths, rate, staged)
        else:
            assemble(directory, paths, stage, staged, output_rate=rate, output_frames=frames,
                     final=True, nvenc=plan.get("nvenc", False))
        if plan["copy_tracks"]:
            input_tracks = probe(source)
        if plan["copy_tracks"] and (input_tracks["audio"] or input_tracks["subtitles"]):
            show("phase", phase="mux")
            offset = -Fraction(input_tracks["video"].get("start_time", "0"))
            muxed = Path(temp) / ("muxed" + staged.suffix)
            ffmpeg("-copyts", "-i", staged, "-itsoffset", f"{float(offset):.6f}", "-i", source,
                   "-map", "0:v:0", "-map", "1:a?", "-map", "1:s?", "-map_chapters", "-1",
                   "-avoid_negative_ts", "disabled", "-c", "copy", muxed)
            show("phase", phase="verify_mux")
            check_output(muxed, rate, frames, tuple(stage["size"]))
            output_tracks = probe(muxed)
            for kind in ("audio", "subtitles"):
                if [s["codec_name"] for s in input_tracks[kind]] != [s["codec_name"] for s in output_tracks[kind]]:
                    raise ValueError(f"The output {kind} streams differ from the source")
            staged = muxed
        elif nvenc_blocks:
            show("phase", phase="verify")
            check_output(staged, rate, frames, tuple(stage["size"]))
        write_review(job, plan)
        show("phase", phase="publish")
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
        show("phase", phase="validate")
        if not matches(Path(plan["input"]), plan["input_sha256"]):
            raise ValueError("Source file changed since the job was planned")
        for path, checksum in plan["weights"].items():
            if not matches(Path(path), checksum):
                raise ValueError(f"Model file changed since the job was planned: {path}")
    show("plan", stages=len(plan["stages"]))
    for index, _ in enumerate(plan["stages"]):
        worker = subprocess.run([sys.executable, "-X", "utf8", "-u", __file__, str(job), str(index)])
        if worker.returncode == PAUSED or (job / "pause.request").exists() and worker.returncode == 0:
            show("paused", job=str(job), stage=index+1)
            return
        worker.check_returncode()
    finish(job, plan)
    show("complete", job=str(job), output=str(destination))


def review_slots(plan, first, last):
    stages = plan["stages"]
    stage = stages[0]
    if (len(stages) != 1 or stage["options"]["engine"] != "nvidia-vfg"
            or not stage.get("nvenc_blocks") or "reverse_slots" in stage or "repair_slots" in stage):
        raise ValueError("Select an original VFG or VFG → VSR MP4 job")
    last = first if last is None else last
    if not 0 <= first <= last < plan["output_frames"]:
        raise ValueError("The review range must be inside the output")
    source = Path(plan["input"])
    video = probe(source)["video"]
    source_rate = Fraction(video["avg_frame_rate"])
    rate = Fraction(stage["rate"])
    start = stage["options"]["start_frame"]
    spans = stage["blocks"]
    starts = [span[2] for span in spans]
    slots = []
    for frame in range(first, last + 1):
        position = start + frame * source_rate / rate
        anchor = int(position)
        offset = position - anchor
        block = bisect_right(starts, anchor) - 1
        right = spans[block][1]
        if not offset:
            role = "source"
        elif anchor + 1 >= right:
            role = "tail_hold" if right == spans[-1][1] else "cut_hold"
        else:
            role = "generated"
        slots.append(dict(frame=frame, time=str(Fraction(frame, 1) / rate), block=block, role=role,
                          anchors=[anchor, min(anchor + 1, right - 1)], offset=str(offset)))
    return slots


def review_mark_slots(job, plan, frame, labels):
    """Expand a mark only where the recorded motion evidence agrees."""
    marked = review_slots(plan, frame, frame)[0]
    if marked["role"] in ("cut_hold", "tail_hold"):
        return [marked]
    try:
        report = read_json(job / "vfg-review.json")
    except FileNotFoundError:
        report = {}
    if "freeze" in labels:
        for run in report.get("anchor_holds", ()):
            first, last = run["first_output_frame"], run["last_output_frame"]
            if first <= frame <= last:
                slots = review_slots(plan, first, last)
                return [slot for slot in slots if slot["role"] == "generated"] or [marked]
        # Older exports can lack a report. Measure the visible hold on the
        # encoded timeline, then apply the original source/cut protection.
        output = Path(plan["output"])
        video = dict(probe(output)["video"], decoded_frames=plan["output_frames"])
        start, images = review_neighbours(output, video, frame)
        first, last = hold_span(images, frame - start)
        slots = review_slots(plan, start + first, start + last)
        return [slot for slot in slots if slot["role"] == "generated"] or [marked]
    if "echo" in labels and marked["role"] == "generated":
        ticks = {item["output_frame"] for item in report.get("suspects", ())
                 if "anchor_echo" in item["reasons"] and item["source_frames"] == marked["anchors"]}
        if frame in ticks:
            # The forward frame inside an ABA sequence can already be correct.
            return [review_slots(plan, tick, tick)[0] for tick in sorted(ticks)]
    # Occlusion/ghosting has no range detector with validated precision.
    return [marked]


def repair_job(args, job, plan, slots):
    output = args.output.resolve()
    if output.suffix.lower() != ".mp4" or output == Path(plan["output"]):
        raise ValueError("Choose a separate MP4 destination for the comparison")
    result = read_json(job / "result.json")
    if not matches(Path(plan["output"]), result["sha256"]):
        raise ValueError("The original export no longer matches its receipt")
    stage = plan["stages"][0]
    blocks_to_repair = {}
    if args.choice != "keep":
        for slot in slots:
            if slot["role"] == "generated":
                blocks_to_repair.setdefault(str(slot["block"]), []).append(slot["frame"])
        if not blocks_to_repair:
            raise ValueError("The mark does not identify a generated frame. Inspect nearby frames or the source.")
    weights = dict(plan["weights"])
    if args.choice in ("rife", "gimm-vfi"):
        from engines import local_model_files

        if args.choice == "rife" and any(Fraction(slot["offset"]).denominator > 256 for slot in slots
                                        if slot["role"] == "generated"):
            raise ValueError("RIFE cannot process this fractional timing. Choose VFG or GIMM-VFI.")
        for path in local_model_files(args.choice, stage["options"]["basic_model"]):
            weights[str(path)] = digest(path)
    choice = dict(original_job=str(job), original_sha256=result["sha256"], choice=args.choice,
                  slots=slots, labels=sorted(set(args.label)), motion=args.motion)
    if args.infer_range:
        choice["marked_frame"] = args.first
    repair = output.with_name(output.name + ".job")
    if repair == job:
        raise ValueError("Choose a separate job directory")
    repair.mkdir(exist_ok=True, parents=True)
    with exclusive(repair / "job.lock"):
        selection = repair / "review-selection.json"
        manifest = repair / "job.json"
        if selection.exists():
            if read_json(selection) != choice:
                raise ValueError("This destination already has a different review selection")
        else:
            if output.exists() or manifest.exists():
                raise FileExistsError(output)
            write_json(selection, choice)
        if not blocks_to_repair:
            show("review-kept", output=plan["output"], selection=str(selection))
            return
        if args.choice == "reverse":
            repair_stage = dict(stage, reverse_slots=blocks_to_repair)
        else:
            repair_stage = dict(stage, repair_slots=dict(method=args.choice, blocks=blocks_to_repair))
        repair_plan = dict(plan, output=str(output), stages=[repair_stage], weights=weights)
        if manifest.exists():
            if read_json(manifest) != repair_plan:
                raise ValueError("The comparison job differs from the review selection")
        directory = stage_directory(repair, 0)
        directory.mkdir(exist_ok=True)
        original_paths = stage_paths(stage_directory(job, 0), stage)
        paths = stage_paths(directory, repair_stage)
        for index, (original, target) in enumerate(zip(original_paths, paths)):
            if str(index) in blocks_to_repair:
                continue
            checksum = cached(original, plan["input_sha256"])
            if checksum is None:
                raise ValueError(f"Original block is missing or invalid: {original}")
            if not target.exists():
                try:
                    os.link(original, target)
                except OSError:
                    # Hard links are limited to the source volume.
                    with TemporaryDirectory(dir=directory) as temp:
                        staged = Path(temp) / target.name
                        shutil.copyfile(original, staged)
                        staged.replace(target)
            if not matches(target, checksum):
                raise ValueError(f"Comparison block does not match the original: {target}")
            if not receipt_for(target).exists():
                write_json(receipt_for(target), read_json(receipt_for(original)))
        if not manifest.exists():
            write_json(manifest, repair_plan)
        (repair / "pause.request").unlink(missing_ok=True)
        execute_job(repair, repair_plan, resume=True)


def review_neighbours(source, video, frame):
    import numpy as np

    start = max(0, frame - 12)
    stop = min(video["decoded_frames"], frame + 13)
    filters = f"trim=start_frame={start}:end_frame={stop},scale=160:90:flags=area,format=gray"
    data = subprocess.check_output(command(
        "ffmpeg", "-v", "error", "-nostdin", "-i", source, "-map", f"0:{video['index']}",
        "-vf", filters, "-fps_mode", "passthrough", "-threads:v", "1", "-f", "rawvideo", "-",
    ))
    images = np.frombuffer(data, dtype=np.uint8).reshape(-1, 90, 160).astype(np.float32)
    if len(images) != stop - start:
        raise ValueError("Could not read the marked frame and its neighbours")
    return start, images


def hold_span(images, marked):
    from vfg_review import MAX_COPY, distance

    first = last = marked
    while first > 0 and distance(images[marked], images[first - 1]) <= MAX_COPY:
        first -= 1
    while last + 1 < len(images) and distance(images[marked], images[last + 1]) <= MAX_COPY:
        last += 1
    return first, last


def imported_repair_slots(source, video, frame, labels):
    from scene_detection import classify
    from vfg_review import MIN_MOTION, distance

    start, images = review_neighbours(source, video, frame)
    marked = frame - start
    first, last = hold_span(images, marked) if "freeze" in labels else (marked, marked)
    if first == 0 or last == len(images) - 1:
        raise ValueError("The marked frames need a usable anchor on each side")
    if "freeze" in labels and distance(images[first - 1], images[last + 1]) < MIN_MOTION:
        raise ValueError("The anchor frames are too similar to reconstruct this freeze reliably")
    for boundary in range(first, last + 2):
        result = classify(boundary + start, 0, images[max(0, boundary-3):boundary],
                          images[boundary:boundary+3])
        if result["decision"] == "cut":
            raise ValueError("A probable shot cut crosses the marked frames")
    left, right = first + start - 1, last + start + 1
    rate = Fraction(video["avg_frame_rate"])
    return [dict(frame=tick, role="replacement", anchors=[left, right],
                 time=str(Fraction(tick, 1) / rate), offset=str(Fraction(tick-left, right-left)))
            for tick in range(left + 1, right)]


def encode_imported_repair(source, video, rate, slots, frames, output):
    width, height = video["width"], video["height"]
    first, last = slots[0]["frame"], slots[-1]["frame"]
    # Encode the original video plus the small replacement range in one FFmpeg pass.
    # The Python pipe carries only the repaired frames.
    filters = (
        f"[0:{video['index']}]split=2[before][after];"
        f"[before]trim=end_frame={first},setpts=PTS-STARTPTS,format=yuv420p,setsar=1[a];"
        f"[after]trim=start_frame={last+1},setpts=PTS-STARTPTS,format=yuv420p,setsar=1[c];"
        "[1:v]scale=in_range=full:out_color_matrix=bt709:out_range=limited,format=yuv420p,setsar=1[b];"
        f"[a][b][c]concat=n=3:v=1:a=0,settb={rate.denominator}/{rate.numerator},setpts=N[out]"
    )
    with (output.parent / "encoder.log").open("w+b") as log:
        process = subprocess.Popen(command(
            "ffmpeg", "-v", "error", "-nostdin", "-i", source,
            "-f", "rawvideo", "-pixel_format", "rgb48le", "-video_size", f"{width}x{height}",
            "-framerate", str(rate), "-i", "pipe:0", "-filter_complex", filters, "-map", "[out]",
            "-fps_mode", "passthrough", "-an", "-color_primaries", "bt709", "-color_trc", "bt709",
            *NVENC, "-bf", "0", "-video_track_timescale", rate.numerator, output,
        ), stdin=subprocess.PIPE, stderr=log)
        try:
            for frame in frames:
                process.stdin.write(as_bytes(frame))
            process.stdin.close()
            if process.wait():
                raise BrokenPipeError
        except BrokenPipeError as exc:
            process.wait()
            log.seek(0)
            raise RuntimeError(log.read().decode("utf-8", errors="replace").strip()
                               or "FFmpeg could not encode the repair preview") from exc
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()


def repair_import(args):
    source = args.input.resolve(strict=True)
    output = args.output.resolve()
    if output.suffix.lower() != ".mp4" or source == output:
        raise ValueError("Choose a separate MP4 destination for the imported review")
    if output.exists():
        raise FileExistsError(output)
    info = probe(source, timing=True, allow_tail_gap=True)
    video = info["video"]
    rate = Fraction(video["avg_frame_rate"])
    opts = Namespace(engine="nvidia-vfg", size=None, start_frame=0, frames=None,
                     chunk=24, overlap=0, basic_model=0, motion_scale=0.5, precision="fp16")
    stop, _, size, _ = render_spec(opts, video)
    if not 0 <= args.first < stop:
        raise ValueError("The mark is outside the video")
    checksum = digest(source)
    selected = []
    if args.choice != "keep":
        if any(dimension % 2 for dimension in size):
            raise ValueError("MP4 repair previews require even video dimensions")
        selected = imported_repair_slots(source, video, args.first, args.label)
    choice = dict(source=str(source), original_sha256=checksum, marked_frame=args.first,
                  choice=args.choice, slots=selected, labels=sorted(set(args.label)), motion=args.motion)
    job = output.with_name(output.name + ".job")
    job.mkdir(parents=True, exist_ok=True)
    with exclusive(job / "job.lock"):
        if output.exists() or (job / "job.json").exists():
            raise FileExistsError(output)
        if not selected:
            write_json(job / "review-selection.json", choice)
            show("review-kept", output=str(source))
            return
        left, right = selected[0]["anchors"]
        pair = decode(source, video, left, right + 1)[[0, -1]]
        positions = tuple(Fraction(slot["offset"]) for slot in selected)
        with TemporaryDirectory(dir=output.parent) as scratch:
            stage = Path(scratch) / "video.mp4"
            with exclusive(ROOT / ".runtime" / "gpu.lock"):
                configure_nvidia_worker()
                with redirect_stdout(sys.stderr):
                    from engines import local_interpolator, local_model_files, nvidia_vfg

                    if args.choice == "reverse":
                        frames = nvidia_vfg(opts)(pair, positions, selected=set(positions))
                    else:
                        local_model_files(args.choice, opts.basic_model)
                        engine = local_interpolator(args.choice, opts, rate)
                        frames = (engine(pair, position) for position in positions)
                    encode_imported_repair(source, video, rate, selected, frames, stage)
            if info["audio"] or info["subtitles"]:
                muxed = Path(scratch) / "muxed.mp4"
                offset = -Fraction(video.get("start_time", "0"))
                ffmpeg("-copyts", "-i", stage, "-itsoffset", f"{float(offset):.6f}", "-i", source, "-map", "0:v:0",
                       "-map", "1:a?", "-map", "1:s?", "-map_chapters", "1", "-avoid_negative_ts", "disabled",
                       "-c", "copy", muxed)
                stage = muxed
            check_output(stage, rate, stop, size)
            if not matches(source, checksum):
                raise ValueError("The imported video changed during review")
            write_json(job / "review-selection.json", choice)
            publish(stage, output)
    show("complete", job=str(job), output=str(output))


def dispatch(args):
    if args.action == "review-import":
        repair_import(args)
        return
    job = (args.job_dir or args.output.with_name(args.output.name + ".job")).resolve() if args.action == "run" else args.job_dir.resolve()
    manifest = job / "job.json"
    if args.action in ("review-slots", "review-repair"):
        with exclusive(job / "job.lock"):
            plan = read_json(manifest)
            if not matches(Path(plan["input"]), plan["input_sha256"]):
                raise ValueError("The source has changed since export")
            if args.infer_range and args.last is not None:
                raise ValueError("--infer-range accepts one marked frame")
            slots = (review_mark_slots(job, plan, args.first, args.label) if args.infer_range
                     else review_slots(plan, args.first, args.last))
            if args.action == "review-slots":
                if not Path(plan["output"]).is_file() or not (job / "result.json").is_file():
                    raise ValueError("Wait for the original export to complete")
                show("review-slots", output=plan["output"], fps=plan["output_rate"], slots=slots)
            else:
                repair_job(args, job, plan, slots)
        return
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
            show("phase", phase="prepare")
            plan = new_plan(args)
            write_json(manifest, plan)
            show("job", job=str(job), output=plan["output"])
        else:
            plan = read_json(manifest)
            (job / "pause.request").unlink(missing_ok=True)
        execute_job(job, plan, resume=args.action == "resume")


if __name__ == "__main__":
    raise SystemExit(run_stage(Path(sys.argv[1]), int(sys.argv[2])))
