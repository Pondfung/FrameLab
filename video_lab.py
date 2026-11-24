import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
SAMPLES = (
    ("cranes", "湖面如镜，白鹤伴飞2-许负1级.mp4", []),
    ("banchao", "班超_20260817_113713_fengyutang.mp4", [245]),
    ("yuji", "虞姬1级_partA_0813.mp4", [19]),
    ("chentang", "陈汤_2级_partA_0811-1.mp4", []),
    ("hanxin", "韩信_2级_PART_A_0720.mp4", [448]),
)
SPATIAL = {"basicvsrpp", "seedvr2", "seedvr2-7b", "lanczos"}
TEMPORAL = {"gimm-vfi", "rife", "hold"}


def command(tool, *args):
    return [tool, *map(str, args)]


def run(tool, *args):
    return subprocess.run(command(tool, *args), check=True, stdout=subprocess.PIPE).stdout


def ffmpeg(*args):
    return run("ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", *args)


def same_rate(first, second):
    first, second = Fraction(first), Fraction(second)
    # Matroska stores integer nanoseconds and ffprobe approximates the quotient.
    return first > 0 and second > 0 and abs(1 / first - 1 / second) <= Fraction(10, 10**9)


def probe(path, *, count=False, timing=False, expected_rate=None):
    args = ["-count_frames"] if count else []
    data = json.loads(run("ffprobe", "-v", "error", *args, "-show_streams", "-of", "json", path))
    video = next((stream for stream in data["streams"] if stream["codec_type"] == "video"
                  and not stream.get("disposition", {}).get("attached_pic")), None)
    if video is None:
        raise ValueError(f"No video stream in {path}")
    if timing:
        frames = json.loads(run(
            "ffprobe", "-v", "error", "-select_streams", str(video["index"]), "-show_frames",
            "-show_entries", "frame=best_effort_timestamp", "-of", "json", path,
        ))["frames"]
        timestamps = [int(frame["best_effort_timestamp"]) for frame in frames]
        time_base = Fraction(video["time_base"])
        rate = Fraction(expected_rate or video["avg_frame_rate"])
        # MP4 and Matroska can round PTS to their own timestamp grid.
        video["constant_grid"] = bool(timestamps and rate) and all(
            abs((pts - timestamps[0]) - Fraction(i, 1) / rate / time_base) <= 1
            for i, pts in enumerate(timestamps)
        )
        video["decoded_frames"] = len(timestamps)
    return {"video": video, "audio": [s for s in data["streams"] if s["codec_type"] == "audio"],
            "subtitles": [s for s in data["streams"] if s["codec_type"] == "subtitle"]}


def publish(file, destination):
    # On Windows rename refuses an existing destination.
    if destination.exists():
        raise FileExistsError(destination)
    file.rename(destination)


def prepare(source, out):
    sample_dir = out / "inputs"
    sample_dir.mkdir(parents=True, exist_ok=True)
    films = []
    for key, filename, cuts in SAMPLES:
        path = source / filename
        tracks = probe(path, timing=True)
        films.append({"id": key, "file": str(path.resolve()), "reviewed_cuts": cuts, **tracks})

    fixtures = [
        ("yuji", 30, 48, "scale=960:540:flags=lanczos", "24", "yuji-540p.mkv", 48, (960, 540)),
        ("chentang", 52, 48, "scale=960:540:flags=lanczos", "24", "chentang-540p.mkv", 48, (960, 540)),
        ("cranes", 60, 72, r"select=not(mod(n\,2))", "12", "cranes-12fps.mkv", 36, (1920, 1080)),
    ]
    inputs = []
    for key, first, count, vf, rate, name, frames, size in fixtures:
        film = next(film for film in films if film["id"] == key)
        path = sample_dir / name
        if not path.exists():
            with TemporaryDirectory(dir=sample_dir) as temp:
                staged = Path(temp) / name
                ffmpeg("-i", film["file"], "-map", "0:v:0", "-an", "-vf",
                       f"trim=start_frame={first}:end_frame={first+count},{vf},settb=1/{rate},setpts=N,setsar=1",
                       "-r", rate, "-fps_mode", "cfr", "-c:v", "ffv1", staged)
                check_output(staged, rate, frames, size)
                publish(staged, path)
        check_output(path, rate, frames, size)
        inputs.append({"path": path.name, "source": key, "start_frame": first, "source_frames": count})

    with TemporaryDirectory(dir=out) as temp:
        manifest = Path(temp) / "samples.json"
        manifest.write_text(json.dumps({"sources": films, "fixtures": inputs}, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        manifest.replace(out / "samples.json")
    return out / "samples.json"


def check_output(path, rate, count, size):
    video = probe(path, timing=True, expected_rate=rate)["video"]
    actual = (Fraction(video["avg_frame_rate"]), video["decoded_frames"], (video["width"], video["height"]))
    expected = (Fraction(rate), count, size)
    if not same_rate(actual[0], expected[0]) or actual[1:] != expected[1:] or not video["constant_grid"]:
        raise ValueError(f"Output validation failed for {path}: expected {expected}, got {actual}")


def decoder(path, video, first, stop):
    color_range = "full" if video.get("color_range") == "pc" or video["pix_fmt"] in ("yuvj420p", "gbrp16le") else "limited"
    vf = (f"trim=start_frame={first}:end_frame={stop},setpts=PTS-STARTPTS,"
          f"scale=in_color_matrix=bt709:in_range={color_range},format=rgb48le")
    return command("ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-xerror",
                   "-threads", "2", "-i", path, "-map", f"0:{video['index']}", "-vf", vf, "-an",
                   "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-threads:v", "1", "-")


def decode(path, video, first, stop):
    import numpy as np

    raw = subprocess.check_output(decoder(path, video, first, stop))
    expected = (stop - first) * video["width"] * video["height"] * 6
    if len(raw) != expected:
        raise ValueError(f"Decoder returned {len(raw)} bytes, expected {expected}")
    return np.frombuffer(raw, dtype="<u2").reshape(-1, video["height"], video["width"], 3)


def as_bytes(frame):
    import numpy as np

    if frame.dtype.kind == "f":
        if not np.isfinite(frame).all():
            raise ValueError("Model returned a non-finite RGB value")
        return np.rint(np.clip(frame, 0, 1) * 65535).astype("<u2").tobytes()
    return frame.astype("<u2", copy=False).tobytes()


def render_spec(args, video):
    if (not video["constant_grid"] or video.get("field_order") not in ("progressive", None, "unknown")
        or video.get("sample_aspect_ratio") not in (None, "1:1", "N/A")
        or video.get("pix_fmt") not in ("yuv420p", "yuvj420p", "yuv422p", "yuv444p", "gbrp16le")
        or any(video.get(key) not in (None, "bt709", "unknown", "gbr", "rgb")
               for key in ("color_space", "color_transfer", "color_primaries"))):
        raise ValueError("This evaluator expects progressive, square-pixel BT.709 SDR CFR (8-bit YUV or 16-bit RGB)")
    total = video["decoded_frames"]
    stop = min(total, args.start_frame + args.frames) if args.frames is not None else total
    if not 0 <= args.start_frame < stop or args.chunk < 1 or args.overlap < 0:
        raise ValueError("Invalid frame range or chunk parameters")
    width, height = video["width"], video["height"]
    out_rate = Fraction(video["avg_frame_rate"]) * (2 if args.engine in TEMPORAL else 1)
    internal_size = (width * 4, height * 4) if args.engine == "basicvsrpp" else (width, height)
    requested_size = args.size or internal_size
    if args.engine in TEMPORAL and args.size:
        raise ValueError("--size applies to upscaling engines")
    if args.engine in ("lanczos", "seedvr2", "seedvr2-7b") and not args.size:
        raise ValueError(f"{args.engine} requires --size WIDTHxHEIGHT")
    if args.engine in ("seedvr2", "seedvr2-7b"):
        internal_size = requested_size
    if args.engine == "seedvr2-7b" and args.size[0] * height != args.size[1] * width:
        raise ValueError("SeedVR2 7B requires a target matching the input aspect ratio")
    return stop, out_rate, internal_size, requested_size


def blocks(start, stop, cuts, chunk):
    boundaries = [start, *sorted({cut for cut in cuts if start < cut < stop}), stop]
    for left, right in zip(boundaries, boundaries[1:]):
        for first in range(left, right, chunk):
            yield left, right, first, min(first + chunk, right)


def infer_block(args, read, engine, internal_size, left, right, first, last):
    if args.engine in TEMPORAL:
        frames = read(first, min(last + 1, right))
        generated = engine(frames)
        if len(generated) != len(frames) - 1:
            raise ValueError("Interpolation engine returned an unexpected number of frames")
        for i, frame in enumerate(frames[:last-first]):
            middle = generated[i] if i < len(generated) else frame
            if middle.shape != frame.shape:
                raise ValueError(f"Unexpected interpolation output shape: {middle.shape}")
            yield frame
            yield middle
    else:
        context_first = max(left, first - args.overlap)
        context_last = min(right, last + args.overlap)
        frames = read(context_first, context_last)
        generated = engine(frames)
        if len(generated) != len(frames):
            raise ValueError("Super-resolution engine changed the frame count")
        for frame in generated[first-context_first:last-context_first]:
            if (frame.shape[1], frame.shape[0]) != internal_size or frame.shape[-1] != 3:
                raise ValueError(f"Unexpected model output shape: {frame.shape}")
            yield frame


def encode_frames(path, frames, rate, internal_size, size):
    inputs = f"{internal_size[0]}x{internal_size[1]}"
    target = f"{size[0]}:{size[1]}"
    with TemporaryDirectory(dir=path.parent) as scratch:
        log = Path(scratch) / "encoder.log"
        with log.open("wb") as errors:
            process = subprocess.Popen(command(
                "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error",
                "-f", "rawvideo", "-pixel_format", "rgb48le", "-video_size", inputs, "-framerate", str(rate),
                "-i", "-", "-an", "-vf", f"scale={target}:flags=lanczos,setsar=1,format=gbrp16le",
                "-color_range", "pc", "-color_primaries", "bt709", "-color_trc", "bt709",
                "-c:v", "ffv1", path,
            ), stdin=subprocess.PIPE, stderr=errors)
            try:
                for frame in frames:
                    process.stdin.write(as_bytes(frame))
                process.stdin.close()
                if process.wait() != 0:
                    raise RuntimeError(log.read_text(encoding="utf-8", errors="replace"))
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait()


def render(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.output.suffix.lower() != ".mkv":
        raise ValueError("Use an .mkv destination for this FFV1 evaluator")
    video = probe(args.input, timing=True)["video"]
    stop, out_rate, internal_size, requested_size = render_spec(args, video)
    total = video["decoded_frames"]

    from engines import make_engine

    engine = make_engine(args, Fraction(video["avg_frame_rate"]))
    def read(first, last):
        return decode(args.input, video, first, last)

    def frames():
        for left, right, first, last in blocks(args.start_frame, stop, args.cut_frame, args.chunk):
            yield from infer_block(args, read, engine, internal_size, left, right, first, last)
            print(f"{args.engine}: frames {first}..{last-1}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as scratch:
        staged = Path(scratch) / "video.mkv"
        encode_frames(staged, frames(), out_rate, internal_size, requested_size)
        expected = (stop - args.start_frame) * (2 if args.engine in TEMPORAL else 1)
        check_output(staged, out_rate, expected, requested_size)
        if args.start_frame == 0 and stop == total and Fraction(video.get("start_time", "0")) == 0:
            muxed = Path(scratch) / "with-audio.mkv"
            ffmpeg("-copyts", "-i", staged, "-i", args.input, "-map", "0:v:0", "-map", "1:a?",
                   "-map", "1:s?", "-c", "copy", muxed)
            check_output(muxed, out_rate, expected, requested_size)
            publish(muxed, args.output)
        else:
            publish(staged, args.output)


def score(args):
    import re

    candidate = probe(args.candidate, timing=True)["video"]
    reference = probe(args.reference, timing=True)["video"]
    size = candidate["width"], candidate["height"]
    if size != (reference["width"], reference["height"]):
        raise ValueError("Candidate and reference dimensions differ")
    rate = Fraction(candidate["avg_frame_rate"])
    if not candidate["constant_grid"] or not reference["constant_grid"] or rate != Fraction(reference["avg_frame_rate"]):
        raise ValueError("Candidate and reference require the same CFR grid")
    count = args.frames or candidate["decoded_frames"]
    if count < 1 or count > candidate["decoded_frames"]:
        raise ValueError("Invalid candidate frame count")
    if args.reference_start_frame < 0 or args.reference_start_frame + count > reference["decoded_frames"]:
        raise ValueError("Invalid reference frame interval")
    if args.inserted_only and count < 3:
        raise ValueError("An inserted-frame comparison needs at least three output frames")
    filters = []
    for index, (video, first) in enumerate(((candidate, 0), (reference, args.reference_start_frame))):
        last = first + count
        steps = [f"trim=start_frame={first}:end_frame={last}"]
        if args.inserted_only:
            steps.extend([r"select=mod(n\,2)", f"trim=end_frame={(count-1)//2}"])
        # Identical timestamps and sample encoding on both branches.
        steps.extend(["scale=out_color_matrix=bt709:out_range=limited", "format=yuv420p",
                      f"settb={rate.denominator}/{rate.numerator}", "setpts=N", f"split[p{index}][s{index}]"])
        filters.append(f"[{index}:{video['index']}]" + ",".join(steps))
    filters.extend(["[p0][p1]psnr=shortest=1[p]", "[s0][s1]ssim=shortest=1[s]"])
    result = subprocess.run(command(
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "info", "-filter_complex_threads", "2",
        "-i", args.candidate, "-i", args.reference, "-filter_complex", ";".join(filters),
        "-map", "[p]", "-map", "[s]", "-f", "null", "-",
    ), stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", check=True)
    psnr = re.search(r"PSNR .*?average:(\S+)", result.stderr)
    ssim = re.search(r"SSIM .*?All:(\S+)", result.stderr)
    if psnr is None or ssim is None:
        raise ValueError("FFmpeg produced no aligned frame scores")
    print(json.dumps({"frames_scored": (count-1)//2 if args.inserted_only else count,
                      "psnr_db": psnr.group(1), "ssim": float(ssim.group(1))}, indent=2))


def doctor():
    import torch
    import torchvision

    print(f"Torch {torch.__version__}, torchvision {torchvision.__version__}, CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python")
    free, total = torch.cuda.mem_get_info()
    print(f"{torch.cuda.get_device_name()}, free {free / 2**20:.0f}/{total / 2**20:.0f} MiB")
    x = torch.ones((1, 1, 16, 16), device="cuda")
    offsets = torch.zeros((1, 18, 14, 14), device="cuda")
    weight = torch.ones((1, 1, 3, 3), device="cuda")
    print("CUDA matmul:", float((x[0, 0] @ x[0, 0]).mean()))
    print("CUDA deform_conv2d:", float(torchvision.ops.deform_conv2d(x, offsets, weight).mean()))
    try:
        import cupy
        print("CuPy NVRTC:", int(cupy.arange(8).sum().item()))
    except ImportError:
        print("CuPy not installed")


def parse_size(value):
    try:
        w, h = map(int, value.lower().split("x"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use WIDTHxHEIGHT") from exc
    if min(w, h) < 2 or w % 2 or h % 2:
        raise argparse.ArgumentTypeError("Use positive even dimensions")
    return w, h


def model_options(parser):
    parser.add_argument("--size", type=parse_size)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--cut-frame", type=int, action="append", default=[])
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--tile", type=int, default=192, help="Input tile edge for BasicVSR++; 0 uses a full frame")
    parser.add_argument("--basic-model", type=int, choices=(0, 1, 2, 3), default=0)
    parser.add_argument("--motion-scale", type=float, choices=(0.25, 0.5, 1.0), default=1.0)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Local video super-resolution and interpolation evaluator")
    subs = parser.add_subparsers(dest="action", required=True)
    metadata = subs.add_parser("probe")
    metadata.add_argument("input", type=Path)
    prep = subs.add_parser("prepare")
    prep.add_argument("source", type=Path)
    prep.add_argument("output", type=Path)
    metrics = subs.add_parser("score")
    metrics.add_argument("candidate", type=Path)
    metrics.add_argument("reference", type=Path)
    metrics.add_argument("--reference-start-frame", type=int, default=0)
    metrics.add_argument("--frames", type=int)
    metrics.add_argument("--inserted-only", action="store_true")
    subs.add_parser("doctor")
    job = subs.add_parser("render")
    job.add_argument("input", type=Path)
    job.add_argument("output", type=Path)
    job.add_argument("--engine", required=True, choices=sorted(SPATIAL | TEMPORAL))
    model_options(job)
    from jobs import add_commands, dispatch

    add_commands(subs)
    args = parser.parse_args()
    if args.action == "probe":
        print(json.dumps(probe(args.input, timing=True), ensure_ascii=False, indent=2))
    elif args.action == "prepare":
        print(prepare(args.source, args.output))
    elif args.action == "score":
        score(args)
    elif args.action == "doctor":
        doctor()
    elif args.action == "render":
        render(args)
    else:
        dispatch(args)


if __name__ == "__main__":
    main()
