"""Loading and text-to-video generation with Self Gradient Forcing+ (one resident model).

The steps are those of the official `inference.py` (T2V, `--use_ema`): the pipeline is `CausalInferencePipeline` built from the
release config, the generator gets the EMA weights of the SGF+ checkpoint, everything is cast to bfloat16, the text encoder is placed
on the GPU or behind the official dynamic-swap wrapper (chosen by free VRAM, as the official script does, unless set), then for each
video: seed, noise `[1, frames, 16, 60, 104]` in bfloat16, `pipeline.inference(...)`, decoded pixels in 0..1.
The model code is the vendored upstream package (vendor/UPSTREAM.md). Nothing here downloads anything.
"""

from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

VENDOR = Path(__file__).resolve().parent / "vendor" / "sgf"
VARIANTS = ("chunkwise", "framewise")
TEXT_ENCODER_MODES = ("auto", "gpu", "dynamic_swap")
LATENT_SHAPE = (16, 60, 104)  # Wan2.1 VAE latent of a 480 x 832 frame
FPS = 16

_LOCK = threading.Lock()
_SLOT = None


def set_seed(seed: int) -> None:
    """utils/misc.py set_seed of the official repository."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config(variant: str):
    from omegaconf import OmegaConf

    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    return OmegaConf.merge(OmegaConf.load(VENDOR / "configs" / "default_config.yaml"), OmegaConf.load(VENDOR / "configs" / f"sgf_plus_{variant}.yaml"))


@dataclass
class Loaded:
    key: tuple
    pipeline: torch.nn.Module
    device: torch.device
    variant: str
    info: dict = field(default_factory=dict)


def make_handle(wan_models: str, checkpoint: str, variant: str, text_encoder: str, keep_loaded: bool, device: str = "cuda") -> dict:
    """device is always "cuda" for the nodes; the test suite runs a tiny model on the CPU."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    if text_encoder not in TEXT_ENCODER_MODES:
        raise ValueError(f"text_encoder must be one of {TEXT_ENCODER_MODES}")
    return {"wan_models": wan_models, "checkpoint": checkpoint, "variant": variant, "text_encoder": text_encoder, "keep_loaded": bool(keep_loaded),
            "device": device}


def text_encoder_placement(mode: str, free_vram_gb: float) -> str:
    """The official low-memory rule (inference.py): below 40 GB of free VRAM the text encoder's weights stay in RAM and are streamed to
    the GPU when used ("dynamic_swap"); otherwise the encoder moves to the GPU ("gpu"). mode "gpu" / "dynamic_swap" forces one."""
    if mode not in TEXT_ENCODER_MODES:
        raise ValueError(f"text_encoder must be one of {TEXT_ENCODER_MODES}")
    if mode == "auto":
        return "dynamic_swap" if free_vram_gb < 40 else "gpu"
    return mode


def release() -> None:
    global _SLOT
    with _LOCK:
        _SLOT = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def acquire(handle: dict) -> Loaded:
    global _SLOT
    key = (handle["wan_models"], handle["checkpoint"], handle["variant"], handle["text_encoder"], handle.get("device", "cuda"))
    with _LOCK:
        if _SLOT is not None and _SLOT.key == key:
            return _SLOT
        _SLOT = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # ComfyUI runs nodes under torch.inference_mode(); build the weights as ordinary tensors, as the official script does.
        with torch.inference_mode(False):
            _SLOT = _load(key)
        return _SLOT


def _load(key) -> Loaded:
    wan_models, checkpoint, variant, text_encoder_mode, device_name = key
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("SGF+ needs a CUDA GPU (the official pipeline runs on CUDA only)")
    from .vendor.sgf.demo_utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb
    from .vendor.sgf.pipeline.causal_inference import CausalInferencePipeline
    from .vendor.sgf.utils import wan_wrapper

    started = time.perf_counter()
    device = torch.device("cuda", torch.cuda.current_device()) if device_name == "cuda" else torch.device(device_name)
    free_gb = get_cuda_free_memory_gb(device) if device.type == "cuda" else 0.0
    wan_wrapper.WAN_MODELS = str(Path(wan_models))  # the folder that holds Wan2.1-T2V-1.3B
    pipeline = CausalInferencePipeline(config(variant), device=device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    weights = state["generator_ema"]  # what the official launcher uses (USE_EMA=1, --use_ema)
    try:
        pipeline.generator.load_state_dict(weights)
    except RuntimeError:  # the official script's fallback for FSDP-wrapped key names
        weights = {k.replace("model._fsdp_wrapped_module.", "model.", 1) if k.startswith("model._fsdp_wrapped_module.") else k: v
                   for k, v in weights.items()}
        pipeline.generator.load_state_dict(weights, strict=True)
    del state, weights
    pipeline = pipeline.to(dtype=torch.bfloat16)
    swap = device.type == "cuda" and text_encoder_placement(text_encoder_mode, free_gb) == "dynamic_swap"
    if swap:
        DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)
    info = {
        "wan_models": Path(wan_models).name,
        "checkpoint": Path(checkpoint).name,
        "variant": variant,
        "text_encoder": "dynamic_swap" if swap else ("gpu" if device.type == "cuda" else device.type),
        "free_vram_gib_at_load": round(free_gb, 1),
        "load_seconds": round(time.perf_counter() - started, 2),
    }
    return Loaded(key=key, pipeline=pipeline, device=device, variant=variant, info=info)


def kv_cache_defaults(variant: str) -> dict:
    """The long-video KV-cache geometry of the official launcher (scripts/infer_self_gradient_forcing.sh)."""
    sink, fifo, current, train = (3, 6, 3, 12) if variant == "chunkwise" else (4, 16, 1, 21)
    return {"kv_cache_max_frames": sink + fifo + current, "kv_cache_sink": sink, "kv_cache_train_frames": train,
            "kv_cache_position_mode": "contiguous"}


def generator_calls(loaded: Loaded, latent_frames: int) -> int:
    """Generator evaluations of one video: per block, one per denoising step plus the clean-context KV-cache pass."""
    return latent_frames // int(loaded.pipeline.num_frame_per_block) * (len(loaded.pipeline.denoising_step_list) + 1)


@torch.no_grad()
def generate(loaded: Loaded, prompt: str, seed: int, latent_frames: int = 21, long_video: bool = True, progress=None) -> dict:
    """Returns {"images": [T, 480, 832, 3] float in 0..1 on the CPU, "video_255": the official `255 * video` tensor (for comparisons),
    "report": dict}. latent_frames: Wan latent frames (4 video frames each, plus 1); a multiple of the block size (3 chunkwise).
    progress(n) is called before every generator evaluation (it may raise to interrupt)."""
    if not prompt.strip():
        raise ValueError("prompt is empty")
    block = int(loaded.pipeline.num_frame_per_block)
    if latent_frames < block or latent_frames % block:
        raise ValueError(f"latent_frames must be a positive multiple of {block} for the {loaded.variant} model")
    device = loaded.device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    generator = loaded.pipeline.generator
    generator_forward = generator.forward
    calls = {"n": 0}

    def counted_forward(*args, **kwargs):
        calls["n"] += 1
        if progress is not None:
            progress(calls["n"])
        return generator_forward(*args, **kwargs)

    generator.forward = counted_forward  # the pipeline calls self.generator(...); an instance attribute is seen by Module.__call__
    try:
        set_seed(int(seed))
        noise = torch.randn([1, latent_frames, *LATENT_SHAPE], device=device, dtype=torch.bfloat16)
        kv = kv_cache_defaults(loaded.variant) if long_video else {}
        video, latents = loaded.pipeline.inference(noise=noise, text_prompts=[prompt], return_latents=True, initial_latent=None,
                                                   long_video=long_video, **kv)
    finally:
        del generator.forward  # back to the class method
        loaded.pipeline.vae.model.clear_cache()
    video = video[0].permute(0, 2, 3, 1).cpu()  # [T, H, W, C], as inference.py's rearrange 'b t c h w -> b t h w c'
    report = {
        "seed": int(seed),
        "latent_frames": int(latent_frames),
        "frames": int(video.shape[0]),
        "fps": FPS,
        "width": int(video.shape[2]),
        "height": int(video.shape[1]),
        "long_video": bool(long_video),
        "generator_calls": calls["n"],
        **({"kv_cache": kv} if long_video else {}),
        "seconds": round(time.perf_counter() - started, 2),
    }
    if device.type == "cuda":
        report["peak_vram_allocated_gib"] = round(torch.cuda.max_memory_allocated(device) / 1024**3, 2)
    return {"images": video.float(), "video_255": 255.0 * video, "latents": latents[0].cpu(), "report": report}
