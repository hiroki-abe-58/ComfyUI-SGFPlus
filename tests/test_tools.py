"""The download tool's manifest covers every file the nodes require, and its verification reports what it finds."""

import hashlib
import importlib.util
from pathlib import Path

from sgfplus_comfy import paths

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_model", ROOT / "tools" / "download_model.py")
download_model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(download_model)


def test_manifest_covers_the_required_files_and_the_shipped_checkpoint_names():
    files = download_model.expected_files()
    assert all(len(d) == 64 and int(d, 16) >= 0 for d in files.values())
    for name in paths.WAN_FILES:
        assert f"{paths.WAN}/{Path(name).as_posix()}" in files, name
    assert {"sgf_plus_chunkwise.pt", "sgf_plus_framewise.pt"} <= set(files)
    assert download_model.WAN_FOLDER == paths.WAN


def test_verify_only_reports_missing_and_mismatching_files(tmp_path, capsys, monkeypatch):
    manifest = tmp_path / "model_files.sha256"
    good = tmp_path / "models" / "Wan2.1-T2V-1.3B" / "config.json"
    good.parent.mkdir(parents=True)
    good.write_bytes(b"{}")
    (tmp_path / "models" / "sgf_plus_chunkwise.pt").write_bytes(b"not the checkpoint")
    manifest.write_text(f"# test\n{hashlib.sha256(b'{}').hexdigest()}  Wan2.1-T2V-1.3B/config.json\n{'0' * 64}  sgf_plus_chunkwise.pt\n"
                        f"{'1' * 64}  Wan2.1-T2V-1.3B/Wan2.1_VAE.pth\n", encoding="ascii")
    monkeypatch.setattr(download_model, "MANIFEST", manifest)
    assert download_model.main(["--models-dir", str(tmp_path / "models"), "--verify-only"]) == 1
    out = capsys.readouterr().out
    assert "ok       Wan2.1-T2V-1.3B/config.json" in out and "MISMATCH sgf_plus_chunkwise.pt" in out
    assert "MISSING  Wan2.1-T2V-1.3B/Wan2.1_VAE.pth" in out
