from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import sys
import sysconfig

import numpy as np

RUNTIME = Path(__file__).resolve().parent / ".runtime"
COMFY = RUNTIME / "ComfyUI"
NUMZ = RUNTIME / "SeedVR2-NumZ-4490bd1"
SEED_7B_DIR = RUNTIME / "seedvr2-7b-weights"
SEED_7B_WEIGHTS = "seedvr2_ema_7b_fp16.safetensors"
NUMZ_VAE = "ema_vae_fp16.safetensors"
GIMM_WEIGHTS = "gimmvfi_r_arb_lpips_fp32.safetensors"
SEED_WEIGHTS = "seedvr2_3b_int8_convrot.safetensors"
SEED_VAE = "seedvr2_ema_vae_fp16.safetensors"


def model_files(engine, basic_model):
    packages = Path(sysconfig.get_path("purelib"))
    if engine == "basicvsrpp":
        names = (
            "basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth",
            "basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bi_20210305-4ef437e2.pth",
            "basicvsr_plusplus_c64n7_8x1_300k_vimeo90k_bd_20210305-ab315ab1.pth",
            "basicvsr_plusplus_c128n25_ntire_vsr_20210311-1ff35292.pth",
        )
        directory = packages / "vsbasicvsrpp/models"
        return [directory / names[basic_model], directory / "spynet_20210409-c6c1bd09.pth"]
    if engine == "rife":
        return [packages / "vsrife/models/flownet_v4.25.pkl"]
    if engine == "gimm-vfi":
        directory = COMFY / "models/interpolation/gimm-vfi"
        return [directory / GIMM_WEIGHTS, directory / "raft-things_fp32.safetensors"]
    if engine == "seedvr2":
        return [COMFY / "models/diffusion_models" / SEED_WEIGHTS, COMFY / "models/vae" / SEED_VAE]
    if engine == "seedvr2-7b":
        return [SEED_7B_DIR / SEED_7B_WEIGHTS, SEED_7B_DIR / NUMZ_VAE, NUMZ / "inference_cli.py"]
    return []


def vs_clip(vs, frames, rate, dtype):
    _, height, width, _ = frames.shape
    fmt = vs.RGBH if dtype == np.float16 else vs.RGBS
    blank = vs.core.std.BlankClip(width=width, height=height, length=len(frames), format=fmt,
                                  fpsnum=rate.numerator, fpsden=rate.denominator)

    def fill(n, f):
        frame = f.copy()
        pixels = frames[n].astype(np.float32) / 65535
        for plane in range(3):
            np.copyto(np.asarray(frame[plane]), pixels[:, :, plane], casting="same_kind")
        return frame

    return vs.core.std.ModifyFrame(blank, clips=blank, selector=fill)


def vs_as_array(frame):
    return np.stack([np.asarray(frame[plane]) for plane in range(3)], axis=-1).astype(np.float32)


def basicvsrpp(args, rate):
    import vapoursynth as vs
    from vsbasicvsrpp import basicvsrpp as upscale

    def render(frames):
        dtype = np.float16 if args.precision == "fp16" else np.float32
        clip = vs_clip(vs, frames, rate, dtype)
        result = upscale(clip, model=args.basic_model, length=len(frames), cpu_cache=True,
                         tile=[args.tile, args.tile], auto_download=True)
        return [vs_as_array(result.get_frame(i)) for i in range(len(frames))]

    return render


def rife(args, rate):
    import vapoursynth as vs
    from vsrife import rife as interpolate

    def render(frames):
        if len(frames) < 2:
            return []
        dtype = np.float16 if args.precision == "fp16" else np.float32
        clip = vs_clip(vs, frames, rate, dtype)
        result = interpolate(clip, model="4.25", factor_num=2, scale=args.motion_scale, auto_download=True)
        return [vs_as_array(result.get_frame(2*i+1)) for i in range(len(frames)-1)]

    return render


def require_weights(directory, *names):
    for name in names:
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Model weight missing: {directory / name}. Run python setup_runtime.py weights.")


def load_comfy():
    if not (COMFY / "nodes.py").is_file():
        raise FileNotFoundError("ComfyUI checkout missing. Run python setup_runtime.py deps.")
    # ComfyUI uses top-level imports for its own modules.
    sys.path.insert(0, str(COMFY))


def gimm(args):
    import torch

    require_weights(COMFY / "models/interpolation/gimm-vfi", GIMM_WEIGHTS, "raft-things_fp32.safetensors")
    load_comfy()
    plugin = COMFY / "custom_nodes/ComfyUI-GIMM-VFI"
    spec = spec_from_file_location("video_lab_gimm", plugin / "__init__.py",
                                   submodule_search_locations=[str(plugin)])
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from video_lab_gimm.nodes import DownloadAndLoadGIMMVFIModel, GIMMVFI_interpolate

    with torch.inference_mode():
        model, = DownloadAndLoadGIMMVFIModel().loadmodel(GIMM_WEIGHTS, precision=args.precision)

    def render(frames):
        if len(frames) < 2:
            return []
        with torch.inference_mode():
            tensor = torch.from_numpy(frames.astype(np.float32) / 65535)
            result, _ = GIMMVFI_interpolate().interpolate(model, tensor, args.motion_scale, 2, 0)
        if len(result) != 2 * len(frames) - 1:
            raise ValueError("GIMM-VFI returned an unexpected frame count")
        return list(result[1::2].float().cpu().numpy())

    return render


def seedvr2(args):
    import torch

    require_weights(COMFY / "models/diffusion_models", SEED_WEIGHTS)
    require_weights(COMFY / "models/vae", SEED_VAE)
    load_comfy()
    import nodes
    from comfy_extras.nodes_seedvr import SeedVR2Conditioning, SeedVR2PostProcessing, SeedVR2Preprocess

    with torch.inference_mode():
        vae, = nodes.VAELoader().load_vae(SEED_VAE)
        model, = nodes.UNETLoader().load_unet(SEED_WEIGHTS, "default")

    def render(frames):
        with torch.inference_mode():
            tensor = torch.from_numpy(frames.astype(np.float32) / 65535)
            resized, = nodes.ImageScale().upscale(tensor, "lanczos", *args.size, "disabled")
            padded, = SeedVR2Preprocess.execute(resized).result
            latent, = nodes.VAEEncodeTiled().encode(vae, padded, 512, 128)
            positive, negative = SeedVR2Conditioning.execute(model, latent).result
            restored, = nodes.KSampler().sample(model, 0, 1, 1, "euler", "simple", positive, negative, latent)
            decoded, = nodes.VAEDecodeTiled().decode(vae, restored, 512, 128)
            result, = SeedVR2PostProcessing.execute(decoded, resized, "none").result
        if float(result.max()) == 0 and int(frames.max()) > 4096:
            raise RuntimeError("SeedVR2 returned black frames for a non-black input")
        return result.float().cpu().numpy()

    return render


def seedvr2_7b(args):
    require_weights(SEED_7B_DIR, SEED_7B_WEIGHTS, NUMZ_VAE)
    if not (NUMZ / "inference_cli.py").is_file():
        raise FileNotFoundError("Install the 7B backend with python setup_runtime.py seedvr2-7b")
    sys.path[:0] = [str(NUMZ), str(RUNTIME / "seedvr2-7b-deps")]
    python_path = os.environ.get("PYTHONPATH")
    try:
        # The upstream backend selects its CUDA allocator before importing torch.
        import inference_cli as numz
    finally:
        if python_path is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = python_path
    import torch

    argv = sys.argv
    try:
        sys.argv = [str(NUMZ / "inference_cli.py"), "input.mkv", "--model_dir", str(SEED_7B_DIR),
                    "--dit_model", SEED_7B_WEIGHTS, "--resolution", str(min(args.size)),
                    "--batch_size", "5", "--seed", "0", "--color_correction", "none",
                    "--dit_offload_device", "cpu", "--vae_offload_device", "cpu",
                    "--blocks_to_swap", "36", "--swap_io_components", "--cache_dit", "--cache_vae",
                    "--vae_encode_tiled", "--vae_decode_tiled", "--vae_encode_tile_size", "512",
                    "--vae_decode_tile_size", "512"]
        config = numz.parse_arguments()
    finally:
        sys.argv = argv
    cache = {}

    def render(frames):
        with torch.inference_mode():
            tensor = torch.from_numpy(frames.astype(np.float32) / 65535).to(torch.float16)
            result = numz._process_frames_core(tensor, config, "0", numz.debug, cache)
        if float(result.max()) == 0 and int(frames.max()) > 4096:
            raise RuntimeError("SeedVR2 7B returned black frames for a non-black input")
        return result.float().cpu().numpy()

    return render


def make_engine(args, rate):
    if args.engine == "hold":
        return lambda frames: frames[:-1]
    if args.engine == "lanczos":
        return lambda frames: frames
    if args.engine == "basicvsrpp":
        return basicvsrpp(args, rate)
    if args.engine == "rife":
        return rife(args, rate)
    if args.engine == "gimm-vfi":
        return gimm(args)
    if args.engine == "seedvr2":
        return seedvr2(args)
    if args.engine == "seedvr2-7b":
        return seedvr2_7b(args)
    raise ValueError(f"Unknown engine {args.engine}")
