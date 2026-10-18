"""Download the official SGF+ checkpoint(s) and the Wan2.1-T2V-1.3B files they need into ComfyUI's models/sgf_plus, and verify them.

    python tools/download_model.py --models-dir <ComfyUI>/models/sgf_plus [--variants chunkwise framewise] [--verify-only]

Sources (pinned to the revisions this node pack was verified with; both Apache-2.0):
  https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B               about 17.6 GB: base generator, umT5-XXL text encoder, VAE, tokenizer
  https://huggingface.co/ZihanSu/Self_Gradient_Forcing_Plus   about 11.4 GB per variant: chunkwise/model.pt, framewise/model.pt
Result:
  models/sgf_plus/Wan2.1-T2V-1.3B/...       (the folder name is fixed: the official code looks it up by name)
  models/sgf_plus/sgf_plus_chunkwise.pt     (= chunkwise/model.pt)
  models/sgf_plus/sgf_plus_framewise.pt     (= framewise/model.pt, only with --variants framewise)
Every file is checked against its SHA-256 in tools/model_files.sha256. Needs `huggingface_hub` (installed with ComfyUI's
transformers). Nothing is downloaded unless you run this script.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

WAN = ("Wan-AI/Wan2.1-T2V-1.3B", "37ec512624d61f7aa208f7ea8140a131f93afc9a")
SGF = ("ZihanSu/Self_Gradient_Forcing_Plus", "784c07876a43fb924a2e96095b33a5675282225b")
WAN_FOLDER = "Wan2.1-T2V-1.3B"
VARIANTS = ("chunkwise", "framewise")
MANIFEST = Path(__file__).resolve().with_name("model_files.sha256")


def expected_files() -> dict[str, str]:
    """{path below models/sgf_plus: sha256} from tools/model_files.sha256."""
    out = {}
    for line in MANIFEST.read_text(encoding="ascii").splitlines():
        if line and not line.startswith("#"):
            digest, _, name = line.partition("  ")
            out[name] = digest
    return out


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models-dir", type=Path, required=True, help="ComfyUI's models/sgf_plus folder")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=["chunkwise"], help="SGF+ checkpoints to fetch (default chunkwise)")
    parser.add_argument("--verify-only", action="store_true", help="check existing files, download nothing")
    args = parser.parse_args(argv)

    expected = expected_files()
    wanted = {name: digest for name, digest in expected.items()
              if name.startswith(WAN_FOLDER + "/") or name in {f"sgf_plus_{v}.pt" for v in args.variants}}
    models = args.models_dir
    if not args.verify_only:
        from huggingface_hub import hf_hub_download, snapshot_download

        wan_files = sorted(name.split("/", 1)[1] for name in wanted if name.startswith(WAN_FOLDER + "/"))
        snapshot_download(repo_id=WAN[0], revision=WAN[1], local_dir=str(models / WAN_FOLDER), allow_patterns=wan_files)
        for variant in args.variants:
            target = models / f"sgf_plus_{variant}.pt"
            if target.is_file() and sha256(target) == expected[target.name]:
                continue
            staged = hf_hub_download(repo_id=SGF[0], filename=f"{variant}/model.pt", revision=SGF[1], local_dir=str(models / ".sgf_plus_download"))
            os.replace(staged, target)
    ok = True
    for name, digest in wanted.items():
        path = models / name
        good = path.is_file() and sha256(path) == digest
        ok &= good
        print(f"{'ok      ' if good else ('MISMATCH' if path.is_file() else 'MISSING ')} {name}", flush=True)
    print("all files verified" if ok else "verification FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
