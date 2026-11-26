from contextlib import nullcontext
from importlib.util import module_from_spec, spec_from_file_location
from functools import lru_cache
import os
from pathlib import Path
import sys
import sysconfig

import numpy as np

from video_lab import BUNDLED_BIN

RUNTIME = Path(__file__).resolve().parent / ".runtime"
VFX_DEPS = RUNTIME / "vfx-deps"
COMFY = RUNTIME / "ComfyUI"
NUMZ = RUNTIME / "SeedVR2-NumZ-4490bd1"
SEED_7B_DIR = RUNTIME / "seedvr2-7b-weights"
SEED_7B_WEIGHTS = "seedvr2_ema_7b_fp16.safetensors"
NUMZ_VAE = "ema_vae_fp16.safetensors"
GIMM_WEIGHTS = "gimmvfi_r_arb_lpips_fp32.safetensors"
SEED_WEIGHTS = "seedvr2_3b_int8_convrot.safetensors"
SEED_VAE = "seedvr2_ema_vae_fp16.safetensors"


def model_files(engine, basic_model):
    if engine in ("nvidia-vsr", "nvidia-vfg"):
        records = list(VFX_DEPS.glob("nvidia_vfx-*.dist-info/RECORD"))
        if len(records) != 1 or not (VFX_DEPS / "nvvfx" / "__init__.py").is_file():
            raise RuntimeError("Install NVIDIA VSR/VFG with: python setup_runtime.py nvidia-vfx")
        return records
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


def load_gimm(args):
    import torch

    require_weights(COMFY / "models/interpolation/gimm-vfi", GIMM_WEIGHTS, "raft-things_fp32.safetensors")
    load_comfy()
    plugin = COMFY / "custom_nodes/ComfyUI-GIMM-VFI"
    spec = spec_from_file_location("video_lab_gimm", plugin / "__init__.py",
                                   submodule_search_locations=[str(plugin)])
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    from video_lab_gimm import nodes

    with torch.inference_mode():
        model, = nodes.DownloadAndLoadGIMMVFIModel().loadmodel(GIMM_WEIGHTS, precision=args.precision)
    return nodes, model


def gimm(args):
    import torch

    nodes, model = load_gimm(args)

    def render(frames):
        if len(frames) < 2:
            return []
        with torch.inference_mode():
            tensor = torch.from_numpy(frames.astype(np.float32) / 65535)
            result, _ = nodes.GIMMVFI_interpolate().interpolate(model, tensor, args.motion_scale, 2, 0)
        if len(result) != 2 * len(frames) - 1:
            raise ValueError("GIMM-VFI returned an unexpected frame count")
        return list(result[1::2].float().cpu().numpy())

    return render


def local_model_files(method, basic_model):
    if BUNDLED_BIN:
        raise ValueError("RIFE and GIMM-VFI need the full FrameLab runtime")
    paths = model_files(method, basic_model)
    for path in paths:
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Local repair model missing: {path}")
    return paths


def local_interpolator(method, args, rate):
    if method == "rife":
        import vapoursynth as vs
        from vsrife import rife as interpolate

        def render(pair, offset):
            dtype = np.float16 if args.precision == "fp16" else np.float32
            clip = vs_clip(vs, pair, rate, dtype)
            # Output frame 1 lands at factor_den / factor_num of this source pair.
            result = interpolate(clip, model="4.25", factor_num=offset.denominator,
                                 factor_den=offset.numerator, scale=args.motion_scale, auto_download=False)
            return vs_as_array(result.get_frame(1))

        return render
    if method != "gimm-vfi":
        raise ValueError(f"Unknown local interpolation method: {method}")

    import torch

    nodes, model = load_gimm(args)
    device = nodes.mm.get_torch_device()
    cast_device = nodes.mm.get_autocast_device(device)

    def render(pair, offset):
        with torch.inference_mode(), torch.autocast(device_type=cast_device, dtype=model.dtype,
                                                   enabled=model.dtype != torch.float32):
            images = torch.from_numpy(pair.astype(np.float32) / 65535).to(device).permute(0, 3, 1, 2)
            height, width = images.shape[-2:]
            # Keep the local GIMM candidate practical when VFG and VSR also occupy the 12 GB GPU.
            ratio = min(1.0, (1280 * 720 / (height * width)) ** 0.5)
            if ratio < 1:
                size = (max(2, int(height * ratio) & ~1), max(2, int(width * ratio) & ~1))
                images = torch.nn.functional.interpolate(images, size=size, mode="bilinear",
                                                         align_corners=False, antialias=True)
            a, b = images.split(1)
            padder = nodes.InputPadder(a.shape, 32)
            a, b = padder.pad(a, b)
            images = torch.stack((a, b), dim=2)
            time = float(offset)
            coord = model.sample_coord_input(1, images.shape[-2:], [time], device=device,
                                             upsample_ratio=args.motion_scale)
            prediction = model(images, [(coord, None)], t=[torch.full((1,), time, device=device)],
                               ds_factor=args.motion_scale)["imgt_pred"][0]
            prediction = padder.unpad(prediction).float()
            if ratio < 1:
                prediction = torch.nn.functional.interpolate(prediction, size=(height, width), mode="bilinear",
                                                             align_corners=False)
            return prediction[0].permute(1, 2, 0).cpu().numpy()

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


def load_nvidia_vfx(engine):
    model_files(engine, 0)
    if BUNDLED_BIN:
        import cupy as arrays
    else:
        import torch as arrays

    sys.path.insert(0, str(VFX_DEPS))
    try:
        import nvvfx
    except ImportError as exc:
        raise RuntimeError("NVIDIA VFX initialization failed; check the runtime and NVIDIA driver") from exc
    finally:
        sys.path.pop(0)
    return arrays, nvvfx


def nvidia_input(arrays, frame):
    rgb = np.ascontiguousarray(frame.transpose(2, 0, 1))
    if BUNDLED_BIN:
        return arrays.asarray(rgb, dtype=arrays.float32) / arrays.float32(65535)
    return arrays.from_numpy(rgb).to(device="cuda", dtype=arrays.float32).div_(65535)


def nvidia_image(arrays, result):
    # Consume/copy the SDK-owned buffer before another call reuses it.
    if BUNDLED_BIN:
        arrays.cuda.runtime.deviceSynchronize()
        return arrays.clip(arrays.from_dlpack(result.image), 0, 1)
    arrays.cuda.synchronize()
    return arrays.from_dlpack(result.image).clamp(0, 1)


def nvidia_output(arrays, result):
    return nvidia_rgb(arrays, nvidia_image(arrays, result))


def nvidia_rgb(arrays, image):
    if BUNDLED_BIN:
        rgb = arrays.rint(image * arrays.float32(65535)).astype(arrays.uint16)
        return arrays.ascontiguousarray(rgb.transpose(1, 2, 0)).get()
    rgb = image.mul_(65535).round_().to(arrays.uint16)
    return rgb.permute(1, 2, 0).contiguous().cpu().numpy()


def load_nvidia_vsr(sdk, size):
    effect = sdk.VideoSuperRes(quality=sdk.VideoSuperRes.QualityLevel.ULTRA)
    effect.output_width, effect.output_height = size
    effect.load()
    return effect


def nvidia_vsr(args):
    arrays, sdk = load_nvidia_vfx(args.engine)
    effect = load_nvidia_vsr(sdk, args.size)

    def render(frames):
        with nullcontext() if BUNDLED_BIN else arrays.inference_mode():
            return [nvidia_output(arrays, effect.run(nvidia_input(arrays, frame))) for frame in frames]

    return render


def nvidia_vfg(args, *, upscale_size=None):
    arrays, sdk = load_nvidia_vfx(args.engine)
    effect = sdk.VideoFrameGeneration(
        mode=sdk.VideoFrameGeneration.Mode.HIGH,
        automatic_shot_change_detection_enabled=getattr(args, "vfg_sdk_shot_detection", True),
    )
    upscale = load_nvidia_vsr(sdk, upscale_size) if upscale_size else None

    def render(frames, positions, *, review=None, selected=(), replacement=None):
        if not effect.is_loaded:
            effect.input_height, effect.input_width = frames.shape[1:3]
            effect.load()

        @lru_cache(maxsize=2)
        def uploaded(index):
            return nvidia_input(arrays, frames[index])

        @lru_cache(maxsize=1)
        def device_grid(rows, cols):
            if BUNDLED_BIN:
                return arrays.asarray(rows)[:, None], arrays.asarray(cols)
            return (arrays.as_tensor(rows, device="cuda")[:, None],
                    arrays.as_tensor(cols, device="cuda"))

        def thumbnail(image, rows, cols):
            from vfg_review import brightness

            y, x = device_grid(tuple(rows), tuple(cols))
            pixels = image[:, y, x].transpose(1, 2, 0) if BUNDLED_BIN else image[:, y, x].permute(1, 2, 0)
            return brightness(pixels.get() if BUNDLED_BIN else pixels.cpu().numpy(), 255)

        def inspect(image, index, position, offset, *, allow_retry):
            alternate = None

            def retry(rows, cols):
                nonlocal alternate
                result = effect.run_at_timestep(uploaded(index + 1), uploaded(index), float(1 - offset))
                alternate = nvidia_image(arrays, result)
                return thumbnail(alternate, rows, cols)

            # nvidia_image makes an owned clamped tensor. Sampling synchronizes it
            # before the alternate call can recycle the SDK output.
            accepted = review(frames, position, lambda y, x: thumbnail(image, y, x),
                              retry=retry if allow_retry and review.retry_version is not None else None)
            return alternate if accepted else image

        with nullcontext() if BUNDLED_BIN else arrays.inference_mode():
            for position in positions:
                index = int(position)
                offset = position - index
                if not offset or index + 1 == len(frames):
                    if review is not None:
                        review(frames, position, None)
                    if upscale is None:
                        yield frames[index]
                        continue
                    image = uploaded(index)
                else:
                    changed = position in selected
                    if changed and replacement is not None:
                        pixels = replacement(frames[index:index+2], offset)
                        image = nvidia_input(arrays, np.rint(np.clip(pixels, 0, 1) * 65535).astype(np.uint16))
                    else:
                        if changed:
                            result = effect.run_at_timestep(uploaded(index + 1), uploaded(index), float(1 - offset))
                        else:
                            result = effect.run_at_timestep(uploaded(index), uploaded(index + 1), float(offset))
                        image = nvidia_image(arrays, result)
                    if review is not None:
                        image = inspect(image, index, position, offset, allow_retry=not changed)
                    if upscale is None:
                        yield nvidia_rgb(arrays, image)
                        continue
                yield nvidia_output(arrays, upscale.run(image))

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
    if args.engine == "nvidia-vsr":
        return nvidia_vsr(args)
    if args.engine == "nvidia-vfg":
        return nvidia_vfg(args)
    raise ValueError(f"Unknown engine {args.engine}")
