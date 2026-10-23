"""A tiny SGF+ setup in the official file layout, so the real pipeline (CausalInferencePipeline, the causal Wan generator with its
dual expert, the KV caches, the scheduler) runs on the CPU in seconds.

- The generator keeps the official width (1536, 12 heads: the pipeline allocates its caches for that layout) but has one layer and a
  tiny FFN; it is written with `save_pretrained` into `<base>/Wan2.1-T2V-1.3B/` and its EMA weights into a checkpoint `.pt`.
- The 11 GB umT5 text encoder and the Wan VAE are replaced by small deterministic stand-ins (`StubTextEncoder`, `StubVAE`).
- The latent is 8 x 8 instead of 60 x 104 (16 tokens per frame instead of 1560); frames are 64 x 64.
`use_tiny(runtime, causal_inference, patcher)` applies these replacements to the given module objects (the ones the nodes use);
`patcher` is pytest's monkeypatch or anything with a `setattr(obj, name, value)` method.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch

LATENT = (16, 8, 8)
TOKENS_PER_FRAME = (LATENT[1] // 2) * (LATENT[2] // 2)
PIXEL_SCALE = 8
FRAME = (LATENT[1] * PIXEL_SCALE, LATENT[2] * PIXEL_SCALE)  # 64 x 64 pixels
WAN_CONFIG = {"dim": 1536, "eps": 1e-06, "ffn_dim": 64, "freq_dim": 256, "in_dim": 16, "model_type": "t2v", "num_heads": 12,
              "num_layers": 1, "out_dim": 16, "text_len": 512}
PLACEHOLDERS = ("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth", "google/umt5-xxl/tokenizer_config.json")


class StubTextEncoder(torch.nn.Module):
    """Deterministic [1, 512, 4096] prompt embedding derived from the prompt text."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.05))

    @property
    def device(self):
        return self.scale.device

    def forward(self, text_prompts):
        seed = int.from_bytes(hashlib.sha256("\n".join(text_prompts).encode()).digest()[:4], "little")
        g = torch.Generator().manual_seed(seed)
        emb = torch.randn(len(text_prompts), 512, 4096, generator=g).to(self.device, self.scale.dtype) * self.scale
        return {"prompt_embeds": emb}


class _StubVAEModel(torch.nn.Module):
    def clear_cache(self):
        pass


class StubVAE(torch.nn.Module):
    """decode_to_pixel: the first 3 latent channels squashed to [-1, 1] and enlarged PIXEL_SCALE times (nearest), one frame per latent
    frame (the real decoder gives 4 per latent frame, 8 times larger)."""

    def __init__(self):
        super().__init__()
        self.model = _StubVAEModel()
        self.gain = torch.nn.Parameter(torch.tensor(1.0))

    def encode_to_latent(self, pixels):
        """[B, 3, F, H, W] -> [B, 1 + (F - 1) // 4, 16, 8, 8] (the real VAE's latent count), deterministic."""
        b, c, f, h, w = pixels.shape
        frames = pixels[:, :, :1]
        if f > 1:
            frames = torch.cat([frames, pixels[:, :, 1:].reshape(b, c, (f - 1) // 4, 4, h, w).mean(3)], dim=2)
        pooled = torch.nn.functional.adaptive_avg_pool3d(frames.float(), (frames.shape[2], LATENT[1], LATENT[2]))
        return (pooled.repeat(1, 6, 1, 1, 1)[:, :LATENT[0]] * self.gain).permute(0, 2, 1, 3, 4).to(pixels.dtype)

    def decode_to_pixel(self, latent, use_cache=False):
        pixels = torch.tanh(latent[:, :, :3] * self.gain).float()  # the real wrapper returns float32 too
        return pixels.repeat_interleave(PIXEL_SCALE, dim=-2).repeat_interleave(PIXEL_SCALE, dim=-1)


def make_tiny_models(base: Path, variant: str = "chunkwise", seed: int = 0) -> tuple[Path, Path]:
    """Writes <base>/Wan2.1-T2V-1.3B/ (tiny generator + placeholder files) and <base>/sgf_plus_tiny_<variant>.pt; returns both."""
    from sgfplus_comfy import runtime
    from sgfplus_comfy.vendor.sgf.utils import wan_wrapper
    from sgfplus_comfy.vendor.sgf.wan.modules.causal_model import CausalWanModel

    torch.manual_seed(seed)
    wan = Path(base) / "Wan2.1-T2V-1.3B"
    wan.mkdir(parents=True, exist_ok=True)
    CausalWanModel(**WAN_CONFIG).save_pretrained(str(wan))
    for name in PLACEHOLDERS:
        (wan / name).parent.mkdir(parents=True, exist_ok=True)
        (wan / name).write_bytes(b"")
    previous = wan_wrapper.WAN_MODELS
    wan_wrapper.WAN_MODELS = str(base)
    try:
        generator = wan_wrapper.WanDiffusionWrapper(**runtime.config(variant).model_kwargs, is_causal=True)
    finally:
        wan_wrapper.WAN_MODELS = previous
    with torch.no_grad():
        for p in generator.parameters():
            p.copy_(torch.randn_like(p) * 0.02)
    state = {k: v.clone() for k, v in generator.state_dict().items()}
    checkpoint = Path(base) / f"sgf_plus_tiny_{variant}.pt"
    torch.save({"generator_ema": state, "generator": {k: v * 0 for k, v in state.items()}}, str(checkpoint))
    (Path(base) / "tiny.json").write_text(json.dumps({"variant": variant, "config": WAN_CONFIG}), encoding="utf-8")
    return wan.parent, checkpoint


def use_tiny(runtime, causal_inference, patcher) -> None:
    """Stand-ins for the text encoder and the VAE, the small latent, and the matching per-frame token count."""
    patcher.setattr(causal_inference, "WanTextEncoder", StubTextEncoder)
    patcher.setattr(causal_inference, "WanVAEWrapper", StubVAE)
    patcher.setattr(runtime, "LATENT_SHAPE", LATENT)
    original_init = causal_inference.CausalInferencePipeline.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.frame_seq_length = TOKENS_PER_FRAME
        self.num_transformer_blocks = WAN_CONFIG["num_layers"]

    patcher.setattr(causal_inference.CausalInferencePipeline, "__init__", init)
