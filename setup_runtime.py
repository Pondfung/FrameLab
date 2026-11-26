import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import venv

ROOT = Path(__file__).resolve().parent / ".runtime"
PYTHON = ROOT / "venv" / "Scripts" / "python.exe"
COMFY = ROOT / "ComfyUI"
GIMM = COMFY / "custom_nodes" / "ComfyUI-GIMM-VFI"
NUMZ = ROOT / "SeedVR2-NumZ-4490bd1"
NUMZ_REVISION = "4490bd1f482e026674543386bb2a4d176da245b9"
SEED_7B = "seedvr2_ema_7b_fp16.safetensors"
SEED_7B_SHA = "7b8241aa957606ab6cfb66edabc96d43234f9819c5392b44d2492d9f0b0bbe4a"
TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
COMMITS = (
    (COMFY, "https://github.com/Comfy-Org/ComfyUI.git", "2fcb6cef9ab1b6dc556b7cff1765a0724ad5e884"),
    (GIMM, "https://github.com/kijai/ComfyUI-GIMM-VFI.git", "4c9a3123762af85e7c796e41737da0b70c75d72d"),
)
SEED_REPO = "https://huggingface.co/Comfy-Org/SeedVR2/resolve/main/"
GIMM_REPO = "https://huggingface.co/Kijai/GIMM-VFI_safetensors/resolve/main/"
WEIGHTS = (
    ("seedvr2", "diffusion_models/seedvr2_3b_int8_convrot.safetensors", SEED_REPO,
     "diffusion_models/", "c3dec8bcc5916843a8a858572970597462e1f2dc598d6dfd818f6cd40f53a157"),
    ("seedvr2", "vae/seedvr2_ema_vae_fp16.safetensors", SEED_REPO,
     "vae/", "20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1"),
    ("gimm-vfi", "interpolation/gimm-vfi/gimmvfi_r_arb_lpips_fp32.safetensors", GIMM_REPO,
     "", "897b892e7ee481f2e5c09e59661b645abbd6c9aba9c0c01346bbe31aa569e277"),
    ("gimm-vfi", "interpolation/gimm-vfi/raft-things_fp32.safetensors", GIMM_REPO,
     "", "ba7cb7781f9f67030e2f6acc4151ac8701b633d93d39e8d61991f2b472a0e45a"),
)


def execute(*args, cwd=None):
    subprocess.run(list(map(str, args)), check=True, cwd=cwd)


def checkout(path, url, revision):
    if (path / ".git").exists():
        current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()
        if current == revision:
            return
        raise RuntimeError(f"{path} is at {current}, expected {revision}")
    path.mkdir(parents=True, exist_ok=True)
    execute("git", "init", cwd=path)
    execute("git", "fetch", "--depth", "1", url, revision, cwd=path)
    execute("git", "checkout", "--detach", "FETCH_HEAD", cwd=path)


def dependencies(torch_wheel):
    for entry in COMMITS:
        checkout(*entry)
    if not PYTHON.exists():
        venv.EnvBuilder(with_pip=True).create(PYTHON.parent.parent)
    pip = (PYTHON, "-m", "pip", "--disable-pip-version-check", "install")
    execute(*pip, "--index-url", TORCH_INDEX, torch_wheel or "torch==2.10.0+cu128",
            "torchvision==0.25.0+cu128")
    # The engine imports ComfyUI nodes directly and does not run its web UI.
    frontend = ("comfyui-frontend-package", "comfyui-workflow-templates", "comfyui-embedded-docs")
    requirements = (COMFY / "requirements.txt").read_text(encoding="utf-8").splitlines()
    with TemporaryDirectory() as tmp:
        headless = Path(tmp) / "requirements-headless.txt"
        headless.write_text("\n".join(line for line in requirements if not line.startswith(frontend)),
                            encoding="utf-8")
        # Torch already ships the CUDA DLLs. CuPy also needs the matching headers for NVRTC.
        execute(*pip, "-r", headless, "-r", GIMM / "requirements.txt",
                "vsbasicvsrpp==3.0.0", "vsrife==5.7.0", "vapoursynth==82",
                "cupy-cuda12x==14.2.0", "nvidia-cuda-runtime-cu12==12.8.*", "opencv-python==5.1.0.95")
    execute(PYTHON, "-m", "pip", "check")


def checksum(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def weights(selection):
    for model, relative, base, prefix, digest in WEIGHTS:
        if selection != "all" and model != selection:
            continue
        path = COMFY / "models" / relative
        if path.exists():
            if digest and checksum(path) != digest:
                raise ValueError(f"Checksum mismatch: {path}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(path.name + ".download")
        print(f"Downloading {path.name}", flush=True)
        execute("curl.exe", "-fL", "--connect-timeout", "20", "--max-time", "3600", "--retry", "2",
                "--continue-at", "-", "-o", partial, base + prefix + path.name)
        if digest and checksum(partial) != digest:
            raise ValueError(f"Checksum mismatch: {partial}")
        partial.rename(path)


def seedvr2_7b():
    if not PYTHON.is_file():
        raise FileNotFoundError("Run python setup_runtime.py deps first")
    revision_file = NUMZ / ".revision"
    if not revision_file.is_file():
        checkout(NUMZ, "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git", NUMZ_REVISION)
        revision_file.write_text(NUMZ_REVISION + "\n", encoding="ascii")
    if revision_file.read_text(encoding="ascii").strip() != NUMZ_REVISION or not (NUMZ / "inference_cli.py").is_file():
        raise RuntimeError(f"Invalid SeedVR2 backend checkout: {NUMZ}")
    extras = ROOT / "seedvr2-7b-deps"
    execute(PYTHON, "-m", "pip", "install", "--no-deps", "--target", extras,
            "diffusers==0.43.0", "peft==0.22.0", "rotary_embedding_torch==0.9.2", "gguf==0.19.0", "accelerate==1.16.0")
    directory = ROOT / "seedvr2-7b-weights"
    directory.mkdir(exist_ok=True)
    vae = directory / "ema_vae_fp16.safetensors"
    if not vae.exists():
        weights("seedvr2")
        source = COMFY / "models/vae/seedvr2_ema_vae_fp16.safetensors"
        with TemporaryDirectory(dir=directory) as temp:
            staged_vae = Path(temp) / vae.name
            try:
                staged_vae.hardlink_to(source)
            except OSError:
                shutil.copyfile(source, staged_vae)
            staged_vae.rename(vae)
    if checksum(vae) != WEIGHTS[1][-1]:
        raise ValueError(f"Checksum mismatch: {vae}")
    model = directory / SEED_7B
    if not model.exists():
        partial = model.with_suffix(".download")
        execute("curl.exe", "-fL", "--connect-timeout", "20", "--retry", "4", "--continue-at", "-",
                "-o", partial, "https://www.modelscope.cn/models/SeasonQW/SeedVR2_comfyUI/resolve/master/" + model.name)
        if checksum(partial) != SEED_7B_SHA:
            raise ValueError(f"Checksum mismatch: {partial}")
        partial.rename(model)
    elif checksum(model) != SEED_7B_SHA:
        raise ValueError(f"Checksum mismatch: {model}")


def main():
    parser = argparse.ArgumentParser(description="Prepare repository-local video model dependencies")
    commands = parser.add_subparsers(dest="command", required=True)
    deps = commands.add_parser("deps")
    deps.add_argument("--torch-wheel", type=Path)
    models = commands.add_parser("weights")
    models.add_argument("model", choices=("all", "gimm-vfi", "seedvr2"), default="all", nargs="?")
    commands.add_parser("seedvr2-7b", help="install the optional SeedVR2 7B FP16 backend and weights")
    vfx = commands.add_parser("nvidia-vfx", help="install NVIDIA VSR Ultra and VFG High")
    vfx.add_argument("--wheel", type=Path)
    args = parser.parse_args()
    if args.command == "deps":
        dependencies(args.torch_wheel)
    elif args.command == "seedvr2-7b":
        seedvr2_7b()
    elif args.command == "nvidia-vfx":
        if not PYTHON.exists():
            raise FileNotFoundError("Run python setup_runtime.py deps first")
        target = ROOT / "vfx-deps"
        execute(PYTHON, "-m", "pip", "--disable-pip-version-check", "install", "--no-deps", "--only-binary=:all:",
                "--target", target, "--index-url", "https://pypi.nvidia.com", args.wheel or "nvidia-vfx==0.2.0.0")
    else:
        weights(args.model)


if __name__ == "__main__":
    main()
