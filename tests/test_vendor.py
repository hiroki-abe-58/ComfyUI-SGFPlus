"""The vendored upstream code must equal the pinned upstream commit plus the documented edits (scripts/vendor_sync.py), the SDPA
fallback must compute the attention flash-attn computes, and the installed package must not contain code that reads environment
variables, opens network connections or starts processes."""

import ast
import hashlib
import math
import re
from pathlib import Path

import pytest
import torch

import vendor_sync

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "sgfplus_comfy" / "vendor"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _listed() -> dict:
    return dict(reversed(line.split("  ", 1)) for line in (VENDOR / "MANIFEST.sha256").read_text(encoding="ascii").splitlines())


def test_manifest_matches_every_vendored_file():
    actual = {p.relative_to(VENDOR).as_posix(): _sha(p) for p in (VENDOR / vendor_sync.PKG).rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    actual[vendor_sync.LICENSE] = _sha(VENDOR / vendor_sync.LICENSE)
    assert actual == _listed()


def test_vendor_notes_pin_the_commit_and_the_license():
    notes = (VENDOR / "UPSTREAM.md").read_text(encoding="utf-8")
    assert vendor_sync.UPSTREAM_COMMIT in notes and "Apache-2.0" in notes
    for rel in vendor_sync.EDITS:
        assert f"`{rel}`" in notes, rel
    assert "Apache License" in (VENDOR / "LICENSE").read_text(encoding="utf-8")


def test_patched_files_carry_the_change_notice():
    for rel in vendor_sync.EDITS:
        assert (VENDOR / vendor_sync.PKG / rel).read_text(encoding="utf-8").startswith(vendor_sync.NOTICE), rel
    for rel in vendor_sync.COPIED:
        assert not (VENDOR / vendor_sync.PKG / rel).read_bytes().startswith(vendor_sync.NOTICE.encode()), rel


def test_identical_to_the_upstream_checkout(upstream_dir):
    """CI checks out the pinned commit (LF line endings on every platform); copied files are byte for byte, patched files are upstream
    plus exactly the edits listed in vendor_sync.EDITS."""
    tree = vendor_sync.expected_tree(upstream_dir)
    assert set(tree) | {f"{vendor_sync.PKG}/{s}" for s in vendor_sync.SHIMS} == set(_listed())
    for rel, data in tree.items():
        assert (VENDOR / rel).read_bytes() == data, rel


def _without_imports(path: Path) -> str:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    return ast.dump(tree)


@pytest.mark.parametrize("rel", ["wan/modules/causal_model.py", "pipeline/causal_inference.py"])
def test_only_the_imports_change_in_the_model_and_the_pipeline(upstream_dir, rel):
    assert _without_imports(VENDOR / vendor_sync.PKG / rel) == _without_imports(upstream_dir / rel)


def test_the_vendored_pipeline_imports_without_cuda(tmp_path):
    """CI runs CPU-only torch builds; here CUDA is hidden from a fresh interpreter instead."""
    import os
    import subprocess
    import sys

    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1"}  # an empty value counts as unset on Windows
    code = "import torch; assert not torch.cuda.is_available(); import sgfplus_comfy.vendor.sgf.pipeline.causal_inference"
    done = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]


def _attention():
    from sgfplus_comfy.vendor.sgf.wan.modules import attention

    return attention


def _reference(q, k, v, scale, causal):
    """softmax(q k^T * scale) v in float64, from the half-precision inputs flash_attention() would see."""
    q, k, v = (x.to(torch.bfloat16).double() for x in (q, k, v))
    scores = torch.einsum("blhd,bshd->bhls", q, k) * scale
    if causal:
        scores = scores.masked_fill(torch.ones(q.shape[1], k.shape[1], dtype=torch.bool).triu(1), -math.inf)
    return torch.einsum("bhls,bshd->blhd", scores.softmax(-1), v)


@pytest.mark.parametrize("causal", [False, True])
def test_sdpa_fallback_is_the_same_attention(causal):
    attention = _attention()
    if attention.FLASH_ATTN_2_AVAILABLE or attention.FLASH_ATTN_3_AVAILABLE:
        pytest.skip("flash-attn is installed; the fallback is not used")
    g = torch.Generator().manual_seed(0)
    q, k, v = (torch.randn(1, 40, 12, 128, generator=g) for _ in range(3))
    out = attention.flash_attention(q, k, v, causal=causal)
    assert out.dtype == q.dtype and out.shape == q.shape
    ref = _reference(q, k, v, 128 ** -0.5, causal)
    assert (out.double() - ref).abs().max() < 2e-2  # bfloat16 inputs and output


def test_sdpa_fallback_refuses_what_it_does_not_implement():
    attention = _attention()
    if attention.FLASH_ATTN_2_AVAILABLE or attention.FLASH_ATTN_3_AVAILABLE:
        pytest.skip("flash-attn is installed; the fallback is not used")
    q = torch.randn(1, 8, 12, 128)
    with pytest.raises(NotImplementedError):
        attention.flash_attention(q, q, q, q_lens=torch.tensor([4]))
    with pytest.raises(NotImplementedError):
        attention.flash_attention(q, q, q, window_size=(2, 2))
    with pytest.raises(NotImplementedError):
        attention.flash_attention(q, q[:, :4], q[:, :4], causal=True)


PACKAGE_FILES = [p for p in [ROOT / "__init__.py", *(ROOT / "sgfplus_comfy").rglob("*.py"), *(ROOT / "tools").rglob("*.py")]
                 if "__pycache__" not in p.parts]
FORBIDDEN = re.compile(r"os\.environ|getenv|putenv|urllib|requests\.|http\.client|socket\.|subprocess|os\.system|importlib|(?<![.\w])exec\(|(?<![.\w])eval\(|__import__")


def test_package_has_no_environment_network_or_process_code():
    """The categories of the Comfy Registry's automated scan (scripts/ and tests/ are not part of a Registry package)."""
    hits = [f"{p.relative_to(ROOT).as_posix()}:{n}: {line.strip()}" for p in PACKAGE_FILES
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if FORBIDDEN.search(line)]
    assert not hits, "\n".join(hits)
