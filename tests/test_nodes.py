"""The node pack as ComfyUI loads it: registration, model discovery, ComfyUI's own validation of the shipped workflows, and node
execution with the tiny random generator on the CPU (the trained model runs in the real-queue checks of docs/VERIFICATION.md)."""

import asyncio
import functools
import importlib
import json
import shutil
from pathlib import Path

import pytest
import torch

import tiny_model

ROOT = Path(__file__).resolve().parents[1]
NODE_IDS = {"SGFPlusLoader", "SGFPlusTextToVideo", "SGFPlusImageToVideo", "SGFPlusExtendVideo", "SGFPlusUnload"}


def pack_module(comfy_nodes, name):
    """A module of the pack as ComfyUI imported it (under a path-based package name, so not the `sgfplus_comfy` the tests import)."""
    return importlib.import_module(comfy_nodes.NODE_CLASS_MAPPINGS["SGFPlusTextToVideo"].__module__.rsplit(".", 1)[0] + "." + name)


@pytest.fixture(scope="module")
def comfy_nodes(comfy_dir, tiny_models, tmp_path_factory):
    import folder_paths
    import nodes

    base = tmp_path_factory.mktemp("comfy")
    folder_paths.set_output_directory(str(base / "output"))
    folder_paths.set_input_directory(str(base / "input"))
    (base / "input").mkdir()
    from PIL import Image

    Image.new("RGB", (128, 72), (90, 120, 160)).save(base / "input" / "sgf_plus_start.png")  # the name the I2V workflow loads
    models = tiny_models["base"]
    shutil.copyfile(tiny_models["chunkwise"], models / "sgf_plus_chunkwise.pt")  # the name the shipped workflows use
    (models / "notes.txt").write_text("not a checkpoint", encoding="utf-8")
    folder_paths.add_model_folder_path("sgf_plus", str(models), is_default=True)

    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    assert asyncio.run(nodes.load_custom_node(str(ROOT))) is True
    folder_paths.filename_list_cache.clear()
    # the nodes are CUDA-only; here the same nodes run the tiny model on the CPU
    runtime = pack_module(nodes, "runtime")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(runtime, "make_handle", functools.partial(runtime.make_handle, device="cpu"))
        tiny_model.use_tiny(runtime, pack_module(nodes, "vendor.sgf.pipeline.causal_inference"), mp)
        yield nodes
        runtime.release()


def test_registered(comfy_nodes):
    assert NODE_IDS <= set(comfy_nodes.NODE_CLASS_MAPPINGS)


def test_only_checkpoint_files_are_listed_and_names_are_contained(comfy_nodes, tiny_models):
    paths = pack_module(comfy_nodes, "paths")
    assert paths.checkpoints() == ["sgf_plus_chunkwise.pt", "sgf_plus_tiny_chunkwise.pt", "sgf_plus_tiny_framewise.pt"]
    assert Path(paths.wan_models_dir()) == tiny_models["base"]
    for bad in ("notes.txt", "../sgf_plus_chunkwise.pt", "missing.pt", "..\\x.pt", "Wan2.1-T2V-1.3B"):
        with pytest.raises(FileNotFoundError):
            paths.checkpoint_path(bad)


def test_an_incomplete_wan_folder_is_reported(comfy_nodes, tmp_path, monkeypatch):
    paths = pack_module(comfy_nodes, "paths")
    (tmp_path / paths.WAN).mkdir()
    (tmp_path / paths.WAN / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(paths, "_bases", lambda: [str(tmp_path)])
    with pytest.raises(FileNotFoundError, match="incomplete"):
        paths.wan_models_dir()


@pytest.mark.parametrize("name", sorted(p.name for p in (ROOT / "workflows" / "api").glob("*.json")))
def test_api_workflows_pass_comfy_validation(comfy_nodes, name):
    import execution

    prompt = json.loads((ROOT / "workflows" / "api" / name).read_text(encoding="utf-8"))
    valid, error, outputs, node_errors = asyncio.run(execution.validate_prompt("t", prompt, None))
    assert valid, (error, node_errors)


def _handle(comfy_nodes, keep_loaded=True, checkpoint="sgf_plus_chunkwise.pt", variant="chunkwise"):
    return comfy_nodes.NODE_CLASS_MAPPINGS["SGFPlusLoader"].execute(checkpoint, variant, "auto", keep_loaded).args[0]


def _run(comfy_nodes, keep_loaded=True, prompt="a fox in the snow", seed=4, latent_frames=6, long_video=False):
    return comfy_nodes.NODE_CLASS_MAPPINGS["SGFPlusTextToVideo"].execute(_handle(comfy_nodes, keep_loaded), prompt, seed, latent_frames,
                                                                         long_video).args


def test_text_to_video_node_outputs(comfy_nodes):
    images, fps, report = _run(comfy_nodes)
    parsed = json.loads(report)
    assert images.shape == (6, *tiny_model.FRAME, 3) and images.dtype == torch.float32 and fps == 16.0
    assert parsed["model"] == {"checkpoint": "sgf_plus_chunkwise.pt", "variant": "chunkwise", "text_encoder": "cpu"}
    assert parsed["generator_calls"] == 10 and parsed["frames"] == 6 and parsed["seed"] == 4


def test_node_output_equals_the_runtime(comfy_nodes):
    runtime = pack_module(comfy_nodes, "runtime")
    images, _, _ = _run(comfy_nodes, seed=9)
    direct = runtime.generate(runtime.acquire(_handle(comfy_nodes)), "a fox in the snow", 9, latent_frames=6)
    assert torch.equal(images, direct["images"])


def test_bad_length_fails_before_generating(comfy_nodes):
    with pytest.raises(ValueError, match="multiple of 3"):
        _run(comfy_nodes, latent_frames=20)


def test_keep_loaded_false_releases_and_unload_is_safe(comfy_nodes):
    runtime = pack_module(comfy_nodes, "runtime")
    _run(comfy_nodes, keep_loaded=False)
    assert runtime._SLOT is None
    unload = comfy_nodes.NODE_CLASS_MAPPINGS["SGFPlusUnload"]
    assert json.loads(unload.execute().args[0])["unloaded"] is False
    _run(comfy_nodes)
    assert json.loads(unload.execute().args[0])["unloaded"] is True and runtime._SLOT is None


def test_interrupt_stops_between_generator_calls(comfy_nodes, monkeypatch):
    import comfy.model_management as mm

    before, _, _ = _run(comfy_nodes, seed=21)
    calls = {"n": 0}

    def interrupt():
        calls["n"] += 1
        if calls["n"] > 3:
            raise mm.InterruptProcessingException()

    monkeypatch.setattr(mm, "throw_exception_if_processing_interrupted", interrupt)
    with pytest.raises(mm.InterruptProcessingException):
        _run(comfy_nodes, seed=21)
    assert calls["n"] == 4  # the check before loading, then one per generator call
    monkeypatch.undo()
    after, _, _ = _run(comfy_nodes, seed=21)  # the resident model still works and gives the same video
    assert torch.equal(before, after)


def test_long_video_report(comfy_nodes):
    _, _, report = _run(comfy_nodes, latent_frames=15, long_video=True)
    parsed = json.loads(report)
    assert parsed["long_video"] is True and parsed["kv_cache"]["kv_cache_max_frames"] == 12 and parsed["frames"] == 15


def test_image_to_video_node(comfy_nodes):
    runtime = pack_module(comfy_nodes, "runtime")
    image = torch.rand(1, 72, 128, 3)
    node = comfy_nodes.NODE_CLASS_MAPPINGS["SGFPlusImageToVideo"]
    images, fps, report = node.execute(_handle(comfy_nodes), image, "a fox in the snow", 5, 9, False).args
    parsed = json.loads(report)
    assert parsed["context_latents"] == 3 and parsed["latent_frames"] == 6  # 9 in total: the image block + 6 generated
    direct = runtime.generate(runtime.acquire(_handle(comfy_nodes)), "a fox in the snow", 5, latent_frames=6, long_video=False,
                              context=runtime.image_context(image, 3))
    assert torch.equal(images, direct["images"]) and fps == 16.0
    with pytest.raises(ValueError, match="more than the image block"):
        node.execute(_handle(comfy_nodes), image, "a fox", 5, 3, False)


def test_extend_video_node(comfy_nodes):
    runtime = pack_module(comfy_nodes, "runtime")
    video = torch.rand(20, *tiny_model.FRAME, 3)
    node = comfy_nodes.NODE_CLASS_MAPPINGS["SGFPlusExtendVideo"]
    new, _, report = node.execute(_handle(comfy_nodes), video, "the fox turns", 6, 3, 6, False, "new frames").args
    direct = runtime.generate(runtime.acquire(_handle(comfy_nodes)), "the fox turns", 6, latent_frames=6, long_video=False,
                              context=runtime.video_context(video, 3))
    assert json.loads(report)["context_latents"] == 3
    assert torch.equal(new, direct["images"][-new.shape[0]:]) and new.shape[0] == 6  # the tiny stub decodes one frame per latent
    full, _, _ = node.execute(_handle(comfy_nodes), video, "the fox turns", 6, 3, 6, False, "input + new frames").args
    assert full.shape[0] == 20 + 6 and torch.equal(full[20:], new)
    # an input of another size is resized to the generated frames for joining
    small = torch.rand(20, 32, 48, 3)
    joined, _, _ = node.execute(_handle(comfy_nodes), small, "the fox turns", 6, 3, 6, False, "input + new frames").args
    assert joined.shape == (26, *new.shape[1:])


def test_shipped_gui_workflows_use_known_nodes():
    for path in sorted((ROOT / "workflows" / "gui").glob("*.json")):
        types = {n["type"] for n in json.loads(path.read_text(encoding="utf-8"))["nodes"]}
        assert types <= NODE_IDS | {"CreateVideo", "SaveVideo", "PreviewAny", "Note", "MarkdownNote", "LoadImage"}, path.name
