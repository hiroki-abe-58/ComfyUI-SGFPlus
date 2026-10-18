"""The runtime against the official steps, with the tiny random generator on the CPU (the trained model is compared with the official
inference script in docs/VERIFICATION.md)."""

import pytest
import torch

import tiny_model
from sgfplus_comfy import runtime
from sgfplus_comfy.vendor.sgf.pipeline import causal_inference


@pytest.fixture(autouse=True)
def tiny(monkeypatch):
    tiny_model.use_tiny(runtime, causal_inference, monkeypatch)
    yield
    runtime.release()


def _loaded(tiny_models, variant="chunkwise", text_encoder="auto"):
    return runtime.acquire(runtime.make_handle(str(tiny_models["base"]), str(tiny_models[variant]), variant, text_encoder, True, device="cpu"))


LAUNCHER_KV = {  # scripts/infer_self_gradient_forcing.sh: --kv_cache_max_frames sink+fifo+current, --kv_cache_sink, train frames
    "chunkwise": {"kv_cache_max_frames": 3 + 6 + 3, "kv_cache_sink": 3, "kv_cache_train_frames": 12, "kv_cache_position_mode": "contiguous"},
    "framewise": {"kv_cache_max_frames": 4 + 16 + 1, "kv_cache_sink": 4, "kv_cache_train_frames": 21, "kv_cache_position_mode": "contiguous"},
}


def _official(tiny_models, variant, prompt, seed, frames, long_video):
    """inference.py of the official repository (T2V, --use_ema; with --long_video and the launcher's KV-cache arguments when
    long_video), step by step, without this pack's runtime."""
    from sgfplus_comfy.vendor.sgf.utils import wan_wrapper

    wan_wrapper.WAN_MODELS = str(tiny_models["base"])
    pipeline = causal_inference.CausalInferencePipeline(runtime.config(variant), device=torch.device("cpu"))
    pipeline.generator.load_state_dict(torch.load(tiny_models[variant], map_location="cpu")["generator_ema"])
    pipeline = pipeline.to(dtype=torch.bfloat16)
    runtime.set_seed(seed)
    noise = torch.randn([1, frames, *tiny_model.LATENT], device="cpu", dtype=torch.bfloat16)
    kv = LAUNCHER_KV[variant] if long_video else {}
    video, latents = pipeline.inference(noise=noise, text_prompts=[prompt], return_latents=True, initial_latent=None, long_video=long_video, **kv)
    return 255.0 * video[0].permute(0, 2, 3, 1).cpu(), latents[0]


def test_release_configs():
    chunk, frame = runtime.config("chunkwise"), runtime.config("framewise")
    assert chunk.num_frame_per_block == 3 and frame.num_frame_per_block == 1
    for cfg in (chunk, frame):
        assert list(cfg.denoising_step_list) == [1000, 750, 500, 250] and cfg.warp_denoising_step
        assert cfg.model_kwargs.dual_full_expert.enabled and not cfg.independent_first_frame
    with pytest.raises(ValueError):
        runtime.config("other")


def test_handle_validation():
    with pytest.raises(ValueError):
        runtime.make_handle("w", "c", "blockwise", "auto", True)
    with pytest.raises(ValueError):
        runtime.make_handle("w", "c", "chunkwise", "cpu", True)


@pytest.mark.parametrize(("mode", "free", "placement"), [("auto", 39.9, "dynamic_swap"), ("auto", 40.0, "gpu"), ("auto", 80.0, "gpu"),
                                                         ("gpu", 10.0, "gpu"), ("dynamic_swap", 80.0, "dynamic_swap")])
def test_text_encoder_placement_is_the_official_rule(mode, free, placement):
    assert runtime.text_encoder_placement(mode, free) == placement


def test_cuda_is_required_for_the_nodes(monkeypatch, tiny_models):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        runtime.acquire(runtime.make_handle(str(tiny_models["base"]), str(tiny_models["chunkwise"]), "chunkwise", "auto", True))
    assert runtime._SLOT is None


def test_the_ema_weights_are_loaded_in_bfloat16(tiny_models):
    loaded = _loaded(tiny_models)
    ema = torch.load(tiny_models["chunkwise"], map_location="cpu")["generator_ema"]
    state = loaded.pipeline.generator.state_dict()
    assert state.keys() == ema.keys()
    assert all(state[k].dtype == torch.bfloat16 and torch.equal(state[k], ema[k].to(torch.bfloat16)) for k in ema)
    assert loaded.info["text_encoder"] == "cpu" and loaded.info["variant"] == "chunkwise"


def test_fsdp_key_names_load_the_same_weights(tiny_models, tmp_path):
    state = torch.load(tiny_models["chunkwise"], map_location="cpu")
    wrapped = {k.replace("model.", "model._fsdp_wrapped_module.", 1): v for k, v in state["generator_ema"].items()}
    path = tmp_path / "fsdp.pt"
    torch.save({"generator_ema": wrapped}, str(path))
    plain = {k: v.clone() for k, v in _loaded(tiny_models).pipeline.generator.state_dict().items()}
    runtime.release()
    fsdp = runtime.acquire(runtime.make_handle(str(tiny_models["base"]), str(path), "chunkwise", "auto", True, device="cpu"))
    assert all(torch.equal(plain[k], v) for k, v in fsdp.pipeline.generator.state_dict().items())


@pytest.mark.parametrize(("variant", "frames", "long_video"), [("chunkwise", 6, False), ("chunkwise", 18, True), ("framewise", 6, False),
                                                              ("framewise", 24, True)])
def test_generate_equals_the_official_steps(tiny_models, variant, frames, long_video):
    """The long cases are longer than the rolling cache (12 / 21 latent frames), so the FIFO eviction runs."""
    out = runtime.generate(_loaded(tiny_models, variant), "a fox in the snow", 3, latent_frames=frames, long_video=long_video)
    runtime.release()
    video_255, latents = _official(tiny_models, variant, "a fox in the snow", 3, frames, long_video)
    assert torch.equal(out["video_255"], video_255) and torch.equal(out["latents"], latents)
    images = out["images"]
    assert images.shape == (frames, *tiny_model.FRAME, 3) and images.dtype == torch.float32 and 0 <= images.min() and images.max() <= 1
    report = out["report"]
    block = 3 if variant == "chunkwise" else 1
    assert report["generator_calls"] == frames // block * 5 and report["frames"] == frames and report["fps"] == 16
    assert "peak_vram_allocated_gib" not in report
    assert report.get("kv_cache") == (LAUNCHER_KV[variant] if long_video else None)


def test_the_default_is_the_official_launcher_setting():
    import inspect

    assert inspect.signature(runtime.generate).parameters["long_video"].default is True
    for variant in runtime.VARIANTS:
        assert runtime.kv_cache_defaults(variant) == LAUNCHER_KV[variant]


def test_generator_calls_matches_the_pipeline(tiny_models):
    loaded = _loaded(tiny_models)
    assert runtime.generator_calls(loaded, 9) == 15
    seen = []
    runtime.generate(loaded, "a fox", 0, latent_frames=9, progress=seen.append)
    assert seen == list(range(1, 16))


def test_seed_and_prompt_change_the_video_and_repeat_exactly(tiny_models):
    loaded = _loaded(tiny_models)
    a = runtime.generate(loaded, "a fox", 1, latent_frames=3)["video_255"]
    assert torch.equal(a, runtime.generate(loaded, "a fox", 1, latent_frames=3)["video_255"])
    assert not torch.equal(a, runtime.generate(loaded, "a fox", 2, latent_frames=3)["video_255"])
    assert not torch.equal(a, runtime.generate(loaded, "a cat", 1, latent_frames=3)["video_255"])


@pytest.mark.parametrize("bad", [0, 2, 4, 20])
def test_latent_frames_must_fill_whole_blocks(tiny_models, bad):
    with pytest.raises(ValueError, match="multiple of 3"):
        runtime.generate(_loaded(tiny_models), "a fox", 0, latent_frames=bad)


def test_empty_prompt_is_rejected(tiny_models):
    with pytest.raises(ValueError, match="prompt"):
        runtime.generate(_loaded(tiny_models), "  ", 0, latent_frames=3)


def test_interrupt_leaves_a_working_model(tiny_models):
    loaded = _loaded(tiny_models)
    first = runtime.generate(loaded, "a fox", 5, latent_frames=6)["video_255"]

    class Stop(Exception):
        pass

    def stop(n):
        if n == 7:
            raise Stop

    with pytest.raises(Stop):
        runtime.generate(loaded, "a fox", 5, latent_frames=6, progress=stop)
    assert "forward" not in vars(loaded.pipeline.generator)  # the counting wrapper is gone
    assert torch.equal(first, runtime.generate(loaded, "a fox", 5, latent_frames=6)["video_255"])


def test_long_video_uses_the_official_rolling_cache(tiny_models):
    loaded = _loaded(tiny_models)
    out = runtime.generate(loaded, "a river", 7, latent_frames=18, long_video=True)  # 18 > 12 cached frames: the FIFO evicts
    report = out["report"]
    assert report["kv_cache"] == {"kv_cache_max_frames": 12, "kv_cache_sink": 3, "kv_cache_train_frames": 12,
                                  "kv_cache_position_mode": "contiguous"}
    assert out["images"].shape[0] == 18 and torch.isfinite(out["images"]).all()
    assert loaded.pipeline.kv_cache1[0]["k"].shape[1] == 12 * tiny_model.TOKENS_PER_FRAME
    short = runtime.generate(loaded, "a river", 7, latent_frames=6, long_video=False)  # back to the plain cache
    assert loaded.pipeline.kv_cache1[0]["k"].shape[1] == 9 * tiny_model.TOKENS_PER_FRAME  # local_attn_size 9 of the chunkwise config
    assert "kv_cache" not in short["report"]


def test_one_resident_model(tiny_models):
    a = _loaded(tiny_models)
    assert _loaded(tiny_models) is a
    b = _loaded(tiny_models, "framewise")
    assert b is not a and runtime._SLOT is b
    runtime.release()
    assert runtime._SLOT is None


def test_loading_under_inference_mode_gives_ordinary_tensors(tiny_models):
    with torch.inference_mode():
        loaded = _loaded(tiny_models)
    assert not any(p.is_inference() for p in loaded.pipeline.parameters())
