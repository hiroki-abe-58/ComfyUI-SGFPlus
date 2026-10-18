"""Rebuild sgfplus_comfy/vendor/sgf from an upstream checkout and rewrite vendor/MANIFEST.sha256 (maintainer tool).

    python scripts/vendor_sync.py --upstream <Zihan-Su/Self_Gradient_Forcing_Plus checkout at UPSTREAM_COMMIT> [--check]

Only the files that text-to-video inference needs are vendored. COPIED files are byte for byte; PATCHED files get the edits of
`EDITS` below and start with a change notice (Apache-2.0, section 4(b)):
- package-relative imports instead of the top-level names `wan`, `utils`, `demo_utils` (those generic names would collide with other
  ComfyUI custom nodes if registered globally);
- the Wan2.1 model folder is a module variable (`WAN_MODELS`, default the upstream "wan_models") instead of a literal relative path;
- PyTorch SDPA in `flash_attention()` when flash-attn is not installed and no padding lengths are given (the attention is the same;
  flash-attn is not available on Windows);
- CPU-only machines: the CUDA device is not queried at import time.
SHIMS are this pack's own package markers (`__init__.py`) replacing upstream ones that import training / demo code.
tests/test_vendor.py rebuilds the tree from the pinned upstream commit in CI and compares every byte.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "sgfplus_comfy" / "vendor"
PKG = "sgf"
UPSTREAM_COMMIT = "7a7734f592f1bdb51edd28f1c42ca95b9f652453"
NOTICE = "# Changed by ComfyUI-SGFPlus; sgfplus_comfy/vendor/UPSTREAM.md lists the changes.\n"

COPIED = [
    "wan/modules/__init__.py", "wan/modules/model.py", "wan/modules/t5.py", "wan/modules/tokenizers.py", "wan/modules/vae.py",
    "utils/scheduler.py", "configs/default_config.yaml", "configs/sgf_plus_chunkwise.yaml", "configs/sgf_plus_framewise.yaml",
]
SHIMS = ["__init__.py", "wan/__init__.py", "utils/__init__.py", "pipeline/__init__.py", "demo_utils/__init__.py"]
LICENSE = "LICENSE"

_SDPA = '''

def _sdpa_flash_attention(q, k, v, q_lens, k_lens, dropout_p, softmax_scale, q_scale, causal, window_size, dtype):
    """flash_attention() with PyTorch SDPA, for machines without flash-attn (ComfyUI-SGFPlus). Same attention for the case inference
    uses: no padding lengths, no window, no dropout; inputs cast to half precision as flash_attention() does, output in q's dtype."""
    if q_lens is not None or k_lens is not None or tuple(window_size) != (-1, -1) or dropout_p:
        raise NotImplementedError("flash-attn is not installed; the SDPA fallback supports no padding lengths, windows or dropout")
    if causal and q.size(1) != k.size(1):
        raise NotImplementedError("flash-attn is not installed; the SDPA fallback supports causal attention only for equal lengths")
    half_dtypes = (torch.float16, torch.bfloat16)
    out_dtype = q.dtype
    q, k, v = [x if x.dtype in half_dtypes else x.to(dtype) for x in (q, k, v)]
    q, k = q.to(v.dtype), k.to(v.dtype)
    if q_scale is not None:
        q = q * q_scale
    x = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=causal, scale=softmax_scale)
    return x.transpose(1, 2).type(out_dtype)
'''

EDITS = {
    "wan/modules/attention.py": [
        ("\n\ndef flash_attention(\n", _SDPA + "\n\ndef flash_attention(\n"),
        ("    half_dtypes = (torch.float16, torch.bfloat16)\n    assert dtype in half_dtypes\n",
         "    if not (FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE):\n"
         "        return _sdpa_flash_attention(q, k, v, q_lens, k_lens, dropout_p, softmax_scale, q_scale, causal, window_size, dtype)\n"
         "    half_dtypes = (torch.float16, torch.bfloat16)\n    assert dtype in half_dtypes\n"),
    ],
    "wan/modules/causal_model.py": [
        ("from wan.modules.attention import attention, flash_attention\n", "from .attention import attention, flash_attention\n"),
        ("from wan.modules.model import (\n", "from .model import (\n"),
    ],
    "utils/wan_wrapper.py": [
        ("from utils.scheduler import ", "from .scheduler import "),
        ("from wan.modules.tokenizers import ", "from ..wan.modules.tokenizers import "),
        ("from wan.modules.model import ", "from ..wan.modules.model import "),
        ("from wan.modules.vae import ", "from ..wan.modules.vae import "),
        ("from wan.modules.t5 import ", "from ..wan.modules.t5 import "),
        ("from wan.modules.causal_model import ", "from ..wan.modules.causal_model import "),
        ("\n\nclass WanTextEncoder(", '\n\nWAN_MODELS = "wan_models"  # folder that holds Wan2.1-T2V-1.3B (set by the node pack)\n\n\nclass WanTextEncoder('),
        ('torch.load("wan_models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",',
         'torch.load(f"{WAN_MODELS}/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",'),
        ('name="wan_models/Wan2.1-T2V-1.3B/google/umt5-xxl/"', 'name=f"{WAN_MODELS}/Wan2.1-T2V-1.3B/google/umt5-xxl/"'),
        ("        return torch.cuda.current_device()\n",
         '        return torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")\n'),
        ('pretrained_path="wan_models/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"', 'pretrained_path=f"{WAN_MODELS}/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth"'),
        ('f"wan_models/{model_name}/", local_attn_size', 'f"{WAN_MODELS}/{model_name}/", local_attn_size'),
        ('WanModel.from_pretrained(f"wan_models/{model_name}/")', 'WanModel.from_pretrained(f"{WAN_MODELS}/{model_name}/")'),
    ],
    "pipeline/causal_inference.py": [
        ("from utils.wan_wrapper import ", "from ..utils.wan_wrapper import "),
        ("from demo_utils.memory import ", "from ..demo_utils.memory import "),
    ],
    "demo_utils/memory.py": [
        ("gpu = torch.device(f'cuda:{torch.cuda.current_device()}')\n",
         "gpu = torch.device(f'cuda:{torch.cuda.current_device()}') if torch.cuda.is_available() else cpu\n"),
    ],
}


def patch(rel: str, text: str) -> str:
    for old, new in EDITS[rel]:
        if text.count(old) != 1:
            raise ValueError(f"{rel}: expected exactly one occurrence of {old.strip()[:70]!r}, found {text.count(old)}")
        text = text.replace(old, new)
    return NOTICE + text


def expected_tree(upstream: Path) -> dict[str, bytes]:
    out = {f"{PKG}/{rel}": (upstream / rel).read_bytes() for rel in COPIED}
    for rel in EDITS:
        out[f"{PKG}/{rel}"] = patch(rel, (upstream / rel).read_bytes().decode("utf-8")).encode("utf-8")
    out[LICENSE] = (upstream / LICENSE).read_bytes()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--check", action="store_true", help="only compare, write nothing")
    args = ap.parse_args(argv)
    tree = expected_tree(args.upstream)
    bad = [rel for rel, data in tree.items() if not (VENDOR / rel).is_file() or (VENDOR / rel).read_bytes() != data]
    if args.check:
        print("\n".join(f"DIFFERS {rel}" for rel in bad) or "vendor tree matches the upstream checkout")
        return 1 if bad else 0
    for rel in bad:
        (VENDOR / rel).parent.mkdir(parents=True, exist_ok=True)
        (VENDOR / rel).write_bytes(tree[rel])
        print("wrote", rel)
    names = sorted(tree) + [f"{PKG}/{s}" for s in SHIMS]
    lines = [f"{hashlib.sha256((VENDOR / n).read_bytes()).hexdigest()}  {n}" for n in sorted(set(names))]
    (VENDOR / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")
    print(f"MANIFEST.sha256: {len(lines)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
