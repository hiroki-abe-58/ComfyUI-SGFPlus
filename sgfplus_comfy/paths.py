"""Model locations under `<ComfyUI models>/sgf_plus/` (plus any path added for `sgf_plus` in extra_model_paths.yaml):

* `Wan2.1-T2V-1.3B/`: the official Wan-AI/Wan2.1-T2V-1.3B folder (config.json, diffusion_pytorch_model.safetensors,
  models_t5_umt5-xxl-enc-bf16.pth, Wan2.1_VAE.pth, google/umt5-xxl/). The folder name is fixed: the official code looks it up by name.
* SGF+ checkpoints: `*.pt` files (e.g. `sgf_plus_chunkwise.pt` = ZihanSu/Self_Gradient_Forcing_Plus chunkwise/model.pt).
Workflows choose names from these lists only; nothing is downloaded at run time.
"""

from __future__ import annotations

import os

FOLDER = "sgf_plus"
WAN = "Wan2.1-T2V-1.3B"
WAN_FILES = ("config.json", "diffusion_pytorch_model.safetensors", "models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth",
             os.path.join("google", "umt5-xxl", "tokenizer_config.json"))


def register() -> None:
    import folder_paths

    folder_paths.add_model_folder_path(FOLDER, os.path.join(folder_paths.models_dir, FOLDER), is_default=False)


def _bases() -> list[str]:
    import folder_paths

    return [b for b in folder_paths.get_folder_paths(FOLDER) if os.path.isdir(b)]


def wan_models_dir() -> str:
    """The first registered sgf_plus folder that holds a complete Wan2.1-T2V-1.3B folder."""
    for base in _bases():
        if all(os.path.isfile(os.path.join(base, WAN, name)) for name in WAN_FILES):
            return base
    raise FileNotFoundError(f"models/{FOLDER}/{WAN} is missing or incomplete (see the README for the download command)")


def checkpoints() -> list[str]:
    found = set()
    for base in _bases():
        found.update(entry.name for entry in os.scandir(base) if entry.is_file() and entry.name.lower().endswith(".pt"))
    return sorted(found)


def checkpoint_path(name: str) -> str:
    """The file of a listed name. Names come from the directory listing (no path separators); the file is looked up again here."""
    for base in _bases():
        if name in {entry.name for entry in os.scandir(base) if entry.is_file() and entry.name.lower().endswith(".pt")}:
            return os.path.join(base, name)
    raise FileNotFoundError(f"checkpoint {name!r} was not found in models/{FOLDER} (see the README for the download command)")
