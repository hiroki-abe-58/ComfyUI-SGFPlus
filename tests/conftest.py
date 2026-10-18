"""Test setup. References come from environment variables, never from this repository:

SGFPLUS_COMFYUI_DIR     ComfyUI checkout (v0.38.0 or later)
SGFPLUS_UPSTREAM_DIR    Zihan-Su/Self_Gradient_Forcing_Plus checkout at the commit in sgfplus_comfy/vendor/UPSTREAM.md
SGFPLUS_REQUIRE_REFERENCES=1  fail instead of skip when one of them is missing (CI sets this)
No trained weights are used: the model tests build a tiny random SGF+ generator in the official file layout (tests/tiny_model.py) and
run the vendored pipeline on the CPU with small stand-ins for the text encoder and the VAE.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "scripts"))
REQUIRE = os.environ.get("SGFPLUS_REQUIRE_REFERENCES") == "1"


def _path(var):
    p = os.environ.get(var)
    return p if p and Path(p).exists() else None


COMFY = _path("SGFPLUS_COMFYUI_DIR")
UPSTREAM = _path("SGFPLUS_UPSTREAM_DIR")
if COMFY:
    sys.path.insert(1, COMFY)
    import comfy.options

    comfy.options.enable_args_parsing(False)
    import comfy.cli_args

    comfy.cli_args.args.cpu = True


def _need(value, what):
    if value:
        return value
    if REQUIRE:
        pytest.fail(f"{what} is required (SGFPLUS_REQUIRE_REFERENCES=1)")
    pytest.skip(f"{what} not available")


@pytest.fixture(scope="session")
def comfy_dir():
    return _need(COMFY, "ComfyUI checkout (SGFPLUS_COMFYUI_DIR)")


@pytest.fixture(scope="session")
def upstream_dir():
    return Path(_need(UPSTREAM, "Self_Gradient_Forcing_Plus checkout (SGFPLUS_UPSTREAM_DIR)"))


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory):
    """{"base": <dir holding Wan2.1-T2V-1.3B/>, "chunkwise": <checkpoint>, "framewise": <checkpoint>} (tiny random weights)."""
    from tiny_model import make_tiny_models

    base = tmp_path_factory.mktemp("models") / "sgf_plus"
    out = {"base": base}
    for variant in ("chunkwise", "framewise"):
        _, out[variant] = make_tiny_models(base, variant)
    return out
