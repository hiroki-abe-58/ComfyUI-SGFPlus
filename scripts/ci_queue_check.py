"""CPU "real queue" check for CI and for a clean-install test: start a real ComfyUI server with this node pack installed, and run the
shipped workflow through POST /prompt with a tiny random SGF+ generator in the official file layout.

    python scripts/ci_queue_check.py --comfyui <ComfyUI checkout> [--workdir DIR] [--keep-pack]

The trained model (1.3B generator, 11 GB text encoder) cannot run in CI, and the nodes are CUDA-only. The check therefore also
installs a small CI-only package (`zz_sgfplus_ci_tiny`, removed again at the end) with one node, "SGF+ CI tiny model", which the
check queues first: it switches the pack's runtime inside the server to the CPU and to the stand-ins of tests/tiny_model.py (tiny
generator, stub text encoder and VAE). Everything else is the real pack and the real vendored pipeline. This proves that the pack
installs into a stock ComfyUI, that Loader -> Text to Video -> Create Video -> Save Video executes through the queue, that the saved
video is reproducible for the same seed and differs for another, that a bad length fails cleanly, and that Unload works. The
trained-model runs are in docs/VERIFICATION.md. The server is started and stopped by this script only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests"))
import queue_client  # noqa: E402
from tiny_model import make_tiny_models  # noqa: E402

NODE_IDS = {"SGFPlusLoader", "SGFPlusTextToVideo", "SGFPlusUnload"}
HOOK = '''"""CI only (written by ComfyUI-SGFPlus/scripts/ci_queue_check.py): run the SGF+ nodes with the tiny test model on the CPU."""
import functools
import hashlib
import importlib
import importlib.util
import json
import os
import sys

RUNTIME = {runtime!r}
TINY = {tiny!r}
GENLOG = {genlog!r}


class _Set:
    def setattr(self, obj, name, value):
        setattr(obj, name, value)


class SGFPlusCITiny:
    @classmethod
    def INPUT_TYPES(cls):
        return {{"required": {{}}}}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "ci"

    @classmethod
    def IS_CHANGED(cls):
        return float("nan")

    def run(self):
        runtime = next(m for m in list(sys.modules.values())
                       if os.path.normcase(os.path.abspath(getattr(m, "__file__", "") or "")) == os.path.normcase(RUNTIME))
        if not getattr(runtime, "_ci_tiny", False):
            spec = importlib.util.spec_from_file_location("sgfplus_ci_tiny_model", TINY)
            tiny = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tiny)
            pipeline = importlib.import_module(runtime.__name__.rsplit(".", 1)[0] + ".vendor.sgf.pipeline.causal_inference")
            runtime.make_handle = functools.partial(runtime.make_handle, device="cpu")
            tiny.use_tiny(runtime, pipeline, _Set())
            generate = runtime.generate

            def logged(loaded, prompt, seed, *args, **kwargs):
                out = generate(loaded, prompt, seed, *args, **kwargs)
                digest = hashlib.sha256(out["video_255"].float().numpy().tobytes()).hexdigest()[:16]
                with open(GENLOG, "a", encoding="utf-8") as log:
                    log.write(json.dumps({{"seed": int(seed), "frames": int(out["video_255"].shape[0]), "sha": digest}}) + "\\n")
                return out

            runtime.generate = logged  # the nodes call runtime.generate(...), so they go through this
            runtime._ci_tiny = True
        text = "tiny model active in " + runtime.__name__
        return {{"ui": {{"text": [text]}}, "result": (text,)}}


NODE_CLASS_MAPPINGS = {{"SGFPlusCITiny": SGFPlusCITiny}}
NODE_DISPLAY_NAME_MAPPINGS = {{"SGFPlusCITiny": "SGF+ CI tiny model"}}
'''


def wait_up(server: str, proc: subprocess.Popen, timeout: float = 300) -> None:
    start = time.time()
    while time.time() - start < timeout:
        if proc.poll() is not None:
            raise SystemExit(f"ComfyUI exited early with code {proc.returncode}")
        try:
            urllib.request.urlopen(f"{server}/system_stats", timeout=3).read()
            return
        except Exception:
            time.sleep(2)
    raise SystemExit("ComfyUI did not start in time")


def decoded_frames(path: Path):
    import av
    import numpy as np

    with av.open(str(path)) as container:
        return np.stack([frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--comfyui", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8195)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--keep-pack", action="store_true", help="use the pack already installed in custom_nodes (clean-install test)")
    args = parser.parse_args(argv)

    comfy = args.comfyui.resolve()
    work = (args.workdir or Path(tempfile.mkdtemp(prefix="sgfplus-ci-"))).resolve()
    work.mkdir(parents=True, exist_ok=True)
    pack = comfy / "custom_nodes" / "ComfyUI-SGFPlus"
    if pack.is_symlink() or getattr(os.path, "isjunction", lambda p: False)(pack):  # never delete through a link into a working copy
        raise SystemExit(f"{pack} is a link; use a ComfyUI checkout without it")
    if not args.keep_pack:
        if pack.exists():
            shutil.rmtree(pack)
        shutil.copytree(ROOT, pack, ignore=shutil.ignore_patterns(".git", "__pycache__", ".pytest_cache", ".ruff_cache", "outputs"))
    elif not pack.is_dir():
        raise SystemExit(f"--keep-pack: {pack} is not installed")
    hook = comfy / "custom_nodes" / "zz_sgfplus_ci_tiny"
    hook.mkdir(exist_ok=True)
    (hook / "__init__.py").write_text(HOOK.format(runtime=str(pack / "sgfplus_comfy" / "runtime.py"), tiny=str(ROOT / "tests" / "tiny_model.py"),
                                                  genlog=str(work / "generated.jsonl")),
                                      encoding="utf-8")
    models = work / "models" / "sgf_plus"
    _, checkpoint = make_tiny_models(models, "chunkwise")
    shutil.copyfile(checkpoint, models / "sgf_plus_chunkwise.pt")  # the name the shipped workflow uses
    yaml = work / "extra_model_paths.yaml"
    yaml.write_text(f"sgfplus_ci:\n  base_path: {(work / 'models').as_posix()}\n  sgf_plus: sgf_plus\n", encoding="utf-8")
    output = work / "output"

    log = (work / "comfyui.log").open("w", encoding="utf-8")
    server = f"http://127.0.0.1:{args.port}"
    proc = subprocess.Popen(
        [sys.executable, "main.py", "--cpu", "--listen", "127.0.0.1", "--port", str(args.port), "--disable-auto-launch", "--disable-api-nodes",
         "--extra-model-paths-config", str(yaml), "--output-directory", str(output)],
        cwd=comfy, stdout=log, stderr=subprocess.STDOUT,
    )
    failures = []
    try:
        wait_up(server, proc)
        info = json.loads(urllib.request.urlopen(f"{server}/object_info", timeout=30).read())
        print("registered:", sorted(NODE_IDS & set(info)))
        if NODE_IDS - set(info):
            raise SystemExit(f"nodes missing from /object_info: {sorted(NODE_IDS - set(info))}")
        listed = info["SGFPlusLoader"]["input"]["required"]["checkpoint"]
        print("checkpoints:", listed)
        tiny = queue_client.run(server, {"1": {"class_type": "SGFPlusCITiny", "inputs": {}}}, None, 120)
        print(json.dumps({"workflow": "ci_tiny", "status": tiny["status"], "text": tiny.get("text"), "error": tiny.get("error")}))
        if tiny["status"] != "success":
            raise SystemExit(f"CI hook failed: {tiny.get('error')}")

        def run(prefix: str, expect_success: bool = True, **overrides) -> dict:
            wf = json.loads((ROOT / "workflows" / "api" / "t2v_basic.json").read_text(encoding="utf-8"))
            wf["4"]["inputs"]["filename_prefix"] = f"ci/{prefix}"
            for key, value in overrides.items():
                node, _, field = key.partition(".")
                wf[node]["inputs"][field] = value
            result = queue_client.run(server, wf, None, 900)
            error = result.get("error") or {}
            print(json.dumps({"run": prefix, "status": result["status"], "seconds": result["wall_seconds"], "files": result.get("files"),
                              "error": error.get("exception_message", "")[:200] if isinstance(error, dict) else error}))
            if (result["status"] == "success") != expect_success:
                failures.append(f"{prefix}: expected {'success' if expect_success else 'an error'}, got {result['status']} {error}")
            return result

        def video(prefix: str):
            files = sorted((output / "ci").glob(f"{prefix}*.mp4"))
            if len(files) != 1:
                failures.append(f"{prefix}: expected one saved .mp4, found {files}")
                return None
            return decoded_frames(files[0])

        # seed 1, seed 2, seed 1 again: the seed changes the inputs of node 2 every time, so ComfyUI's cache cannot serve the third run
        report = run("seed1_a", **{"2.seed": 1, "2.latent_frames": 6})
        run("seed2", **{"2.seed": 2, "2.latent_frames": 6})
        run("seed1_b", **{"2.seed": 1, "2.latent_frames": 6})
        a, b, other = video("seed1_a"), video("seed1_b"), video("seed2")
        if a is not None and b is not None and other is not None:
            print("frames:", a.shape)
            if a.shape != (6, 64, 64, 3):
                failures.append(f"unexpected video shape {a.shape}")
            generated = [json.loads(line) for line in (work / "generated.jsonl").read_text(encoding="utf-8").splitlines()]
            print("generated:", json.dumps(generated))  # seed1_a, seed2, seed1_b
            if len(generated) < 3 or generated[0]["sha"] != generated[2]["sha"]:
                failures.append("the same seed gave different generated frames")
            elif not (a == b).all():
                failures.append("the same seed gave the same generated frames but different saved MP4 files")
            if (a == other).all():
                failures.append("different seeds gave the same video")
        texts = [json.loads(v[0]) for v in (report.get("text") or {}).values() if v and v[0].lstrip().startswith("{")]
        if not texts or texts[0].get("generator_calls") != 10 or texts[0].get("frames") != 6:
            failures.append(f"unexpected report {texts}")
        run("bad_length", expect_success=False, **{"2.seed": 3, "2.latent_frames": 20})
        run("long_video", **{"2.seed": 4, "2.latent_frames": 15})  # longer than the 12-frame rolling cache of the default setting
        run("plain_cache", **{"2.seed": 5, "2.latent_frames": 6, "2.long_video": False, "1.keep_loaded": False})
        unload = queue_client.run(server, {"1": {"class_type": "SGFPlusUnload", "inputs": {}}}, None, 120)
        print(json.dumps({"workflow": "unload", "status": unload["status"], "text": unload.get("text")}))
        if unload["status"] != "success":
            failures.append(f"unload: {unload.get('error')}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        shutil.rmtree(hook, ignore_errors=True)
    if failures:
        print("FAILED:\n  " + "\n  ".join(failures))
        print((work / "comfyui.log").read_text(encoding="utf-8", errors="replace")[-4000:])
        return 1
    print("queue check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
