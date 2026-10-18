# Vendored upstream code

Source: https://github.com/Zihan-Su/Self_Gradient_Forcing_Plus, commit `7a7734f592f1bdb51edd28f1c42ca95b9f652453` (Apache-2.0, see
`LICENSE` in this folder). The package is vendored as `sgf` (upstream's top-level names `wan`, `utils`, `pipeline`, `demo_utils` are
generic and would collide with other ComfyUI custom nodes if registered globally; here they are subpackages of `sgf`).

Only the files that text-to-video inference needs are vendored. `scripts/vendor_sync.py` rebuilds this folder from an upstream
checkout; `tests/test_vendor.py` repeats that in CI and compares every byte, and `MANIFEST.sha256` lists the SHA-256 of every file here.

## Copied unchanged (byte-identical)

- `sgf/wan/modules/`: `__init__.py`, `model.py`, `tokenizers.py`, `vae.py` (Copyright 2024-2025 The Alibaba Wan Team Authors,
  Apache-2.0, as their headers say)
- `sgf/utils/scheduler.py`
- `sgf/configs/`: `default_config.yaml`, `sgf_plus_chunkwise.yaml`, `sgf_plus_framewise.yaml` (the release configs; the pack reads
  the inference keys only)

## Patched

Each patched file starts with the line `# Changed by ComfyUI-SGFPlus; sgfplus_comfy/vendor/UPSTREAM.md lists the changes.` The exact
edits are the `EDITS` table of `scripts/vendor_sync.py`; the test applies them to the upstream files and compares bytes.

- `wan/modules/attention.py`: a new function `_sdpa_flash_attention` and two lines at the top of `flash_attention()`: when neither
  flash-attn 2 nor 3 is installed, the attention is computed with PyTorch `scaled_dot_product_attention` (inputs cast to half
  precision as `flash_attention()` does, output in the input dtype). It refuses padding lengths, windows, dropout and causal
  attention of unequal lengths, none of which inference uses. Without flash-attn, upstream stops at `assert FLASH_ATTN_2_AVAILABLE`
  (flash-attn has no official Windows build). With flash-attn installed nothing changes.
- `wan/modules/causal_model.py`: two imports become package-relative (`from .attention ...`, `from .model ...`). The test compares
  the parsed syntax trees without the import statements.
- `pipeline/causal_inference.py`: two imports become package-relative. Same syntax-tree test.
- `utils/wan_wrapper.py`: imports become package-relative; a module variable `WAN_MODELS = "wan_models"` (upstream's relative folder
  name) replaces that literal in the five model paths, so the pack can point it at `models/sgf_plus`; `WanTextEncoder.device`
  returns the CPU device when CUDA is not available (upstream calls `torch.cuda.current_device()` unconditionally).
- `demo_utils/memory.py` (copied by upstream from lllyasviel/FramePack, Apache-2.0, as its header says): the module-level `gpu`
  device falls back to the CPU when CUDA is not available, so the module can be imported on a CPU-only machine.
- `wan/modules/t5.py` (Wan team, Apache-2.0): the default `device` argument of `T5EncoderModel.__init__` (a class the pipeline does
  not use) is `torch.cuda.current_device()`, which Python evaluates when the module is imported; it falls back to `'cpu'` when
  CUDA is not available. Same reason as above. The nodes need CUDA anyway; this lets the test suite and CI import the code on the
  CPU.

## Pack shims (own files, not from upstream)

`sgf/__init__.py`, `sgf/wan/__init__.py`, `sgf/utils/__init__.py`, `sgf/pipeline/__init__.py`, `sgf/demo_utils/__init__.py`: package
markers. Upstream's versions import the training stack, the bidirectional pipelines and the demo helpers.

## Not vendored

Everything else: training (`trainer/`, `model/`, `train.py`), the other pipelines, the Gradio demo, `wan/configs`, `wan/distributed`,
`wan/text2video.py` and friends, the gradient-conflict experiments and the prompt lists. `inference.py` is not vendored; its
text-to-video steps are re-implemented in `sgfplus_comfy/runtime.py` and compared with the original script (docs/VERIFICATION.md).

## Provenance of the causal-generation code

Upstream's README says the repository builds on Self Gradient Forcing (zhuang2002/Self_Gradient_Forcing), Causal-Forcing
(thu-ml/Causal-Forcing) and Wan2.1; all three are published under Apache-2.0. Self Gradient Forcing descends from Self Forcing
(guandeh17/Self-Forcing, Apache-2.0), whose README says it is built on top of the open-source implementation of CausVid
(tianweiy/CausVid). CausVid's repository is published under CC BY-NC-SA 4.0, and modules of the same names exist there
(`causal_model.py`, `causal_inference.py`, `wan_wrapper.py`, `scheduler.py`). This pack redistributes the upstream files under the
license their authors declare (Apache-2.0) and does not settle that history; check it yourself before commercial use.
