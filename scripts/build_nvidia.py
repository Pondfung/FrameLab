"""Build the Windows NVIDIA single-file edition from the shared project sources."""

import argparse
from hashlib import file_digest
from pathlib import Path
import shutil
import struct
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".runtime"
MAGIC = b"FRAMELAB-NVIDIA!"


def tool_path(name):
    path = shutil.which(name)
    if not path:
        raise FileNotFoundError(f"{name} is missing from PATH")
    return Path(path)


def add_tree(archive, source, destination, excluded=()):
    if not source.is_dir():
        raise FileNotFoundError(source)
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if (path.is_file() and not set(relative.parts).intersection(excluded)
                and "__pycache__" not in relative.parts and path.suffix not in {".pyc", ".lock"}):
            archive.write(path, f"{destination}/{relative.as_posix()}")


def build_payload(destination, base, site, ffmpeg, ffprobe):
    with ZipFile(destination, "w", ZIP_DEFLATED, compresslevel=3) as archive:
        for name in ("engines.py", "jobs.py", "scene_detection.py", "vfg_review.py", "vfg_retry.py", "video_lab.py"):
            archive.write(ROOT / name, name)
        for name in ("python.exe", "python3.dll", "python313.dll", "vcruntime140.dll", "vcruntime140_1.dll",
                     "LICENSE.txt"):
            archive.write(base / name, f"python/{name}")
        excluded_dlls = {path.name for path in (base / "DLLs").iterdir()
                         if path.name.startswith(("_tkinter", "tcl", "tk", "_test"))}
        add_tree(archive, base / "DLLs", "python/DLLs", excluded_dlls)
        add_tree(archive, base / "Lib", "python/Lib", ("site-packages", "test", "tests", "idlelib", "tkinter", "ensurepip", "turtledemo"))

        libraries = "python/Lib/site-packages"
        for pattern in ("cupy", "cupyx", "cupy_backends", "cupy_cuda12x-*.dist-info", "numpy*", "cuda",
                        "cuda_pathfinder-*.dist-info", "nvidia", "nvidia_cuda_runtime_cu12-*.dist-info"):
            packages = sorted(path for path in site.glob(pattern) if path.is_dir())
            if not packages:
                raise FileNotFoundError(f"{pattern} is missing from {site}")
            for package in packages:
                add_tree(archive, package, f"{libraries}/{package.name}", ("tests", "direct_url.json"))
        # The retry gate only uses OpenCV's in-memory optical flow.
        cv2 = site / "cv2"
        videoio = tuple(path.name for path in cv2.glob("opencv_videoio_*.dll"))
        add_tree(archive, cv2, f"{libraries}/cv2", ("tests", "samples", "data", *videoio))
        for name in ("nvrtc64_120_0.dll", "nvrtc-builtins64_128.dll"):
            archive.write(site / "torch/lib" / name, f"cuda/bin/{name}")
        add_tree(archive, site / "nvidia/cuda_runtime/include", "cuda/include")
        add_tree(archive, RUNTIME / "vfx-deps", ".runtime/vfx-deps", ("direct_url.json",))
        for path in (ffmpeg, ffprobe):
            archive.write(path, f"bin/{path.name}")
        for name in ("LICENSE", "README.txt"):
            path = ffmpeg.parent.parent / name
            if path.is_file():
                archive.write(path, f"bin/{name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/FrameLab-NVIDIA.exe")
    parser.add_argument("--target-dir", type=Path, default=ROOT / "desktop/target")
    parser.add_argument("--offline", action="store_true", help="use only locally cached Rust dependencies")
    args = parser.parse_args()

    python = RUNTIME / "venv/Scripts/python.exe"
    base, site = map(Path, subprocess.check_output(
        [python, "-X", "utf8", "-c", "import sys, sysconfig; print(sys.base_prefix); print(sysconfig.get_path('purelib'))"],
        text=True, encoding="utf-8").splitlines())
    if not (RUNTIME / "vfx-deps/nvvfx/_ext.pyd").is_file():
        raise FileNotFoundError("Run python setup_runtime.py nvidia-vfx first")
    ffmpeg, ffprobe = tool_path("ffmpeg.exe"), tool_path("ffprobe.exe")
    target = args.target_dir.resolve()
    cargo = [tool_path("cargo"), "build", "--manifest-path", ROOT / "desktop/Cargo.toml", "--release", "--locked",
             "--features", "portable", "--target-dir", target]
    if args.offline:
        cargo.append("--offline")
    subprocess.run(cargo, check=True)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as temporary:
        archive = Path(temporary) / "payload.zip"
        executable = Path(temporary) / output.name
        print("Compressing the NVIDIA runtime", flush=True)
        build_payload(archive, base, site, ffmpeg, ffprobe)
        with archive.open("rb") as source:
            version = file_digest(source, "sha256").digest()[:16]
            source.seek(0)
            with executable.open("wb") as result, (target / "release/framelab.exe").open("rb") as stub:
                shutil.copyfileobj(stub, result)
                shutil.copyfileobj(source, result)
                result.write(struct.pack("<16sQ16s", MAGIC, archive.stat().st_size, version))
        executable.replace(output)
    print(f"{output} ({output.stat().st_size / 2**20:.2f} MiB, runtime {version.hex()})", flush=True)


if __name__ == "__main__":
    main()
