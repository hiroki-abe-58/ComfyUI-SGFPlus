"""ComfyUI nodes for Self Gradient Forcing+ text-to-video (runs in the ComfyUI process; nothing is downloaded at run time).

The output is an IMAGE batch of the frames (832 x 480, 16 fps): connect Create Video (fps from the node) and Save Video."""

from __future__ import annotations

import json

import torch
from comfy_api.latest import ComfyExtension, io

from . import paths, runtime

SGFPlusModel = io.Custom("SGF_PLUS_MODEL")


def _progress(total: int):
    import comfy.model_management as mm
    import comfy.utils

    bar = comfy.utils.ProgressBar(max(1, int(total)))

    def tick(done: int):
        mm.throw_exception_if_processing_interrupted()
        bar.update_absolute(min(done, total), total)

    return tick


class SGFPlusLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SGFPlusLoader",
            display_name="SGF+ Loader",
            category="loaders/sgf_plus",
            description=("Loads Self Gradient Forcing+ (an autoregressive Wan2.1-T2V-1.3B with a separate context-writing expert) with the "
                         "official Wan2.1 text encoder (umT5-XXL) and VAE from models/sgf_plus. One model stays resident; loading another "
                         "frees it. Needs a CUDA GPU."),
            inputs=[
                io.Combo.Input("checkpoint", options=paths.checkpoints(), tooltip="An SGF+ checkpoint (.pt) in models/sgf_plus"),
                io.Combo.Input("variant", options=list(runtime.VARIANTS), default="chunkwise",
                               tooltip="Must match the checkpoint: chunkwise (3 latent frames per block) or framewise (1)."),
                io.Combo.Input("text_encoder", options=list(runtime.TEXT_ENCODER_MODES), default="auto",
                               tooltip="Where the 11 GB text encoder lives. auto = the official rule: below 40 GB of free VRAM its weights stay "
                                       "in RAM and are streamed to the GPU when used (dynamic_swap); otherwise gpu."),
                io.Boolean.Input("keep_loaded", default=True, tooltip="Keep the model in memory after the job. Turn off to free VRAM and RAM after every run."),
            ],
            outputs=[SGFPlusModel.Output(display_name="model")],
        )

    @classmethod
    def execute(cls, checkpoint, variant, text_encoder, keep_loaded) -> io.NodeOutput:
        return io.NodeOutput(runtime.make_handle(paths.wan_models_dir(), paths.checkpoint_path(checkpoint), variant, text_encoder, keep_loaded))


class SGFPlusTextToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SGFPlusTextToVideo",
            display_name="SGF+ Text to Video",
            category="video/sgf_plus",
            description=("Generates a video autoregressively, block by block, with 4 denoising steps per block (the official few-step "
                         "setting). 21 latent frames = 81 frames = about 5 s at 16 fps, 832 x 480; longer videos cost the same per block."),
            inputs=[
                SGFPlusModel.Input("model"),
                io.String.Input("prompt", multiline=True, default="A red fox trotting through fresh snow in a pine forest, morning light, cinematic."),
                io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFF, control_after_generate=True,
                             tooltip="Same model, prompt, seed, settings and environment give the same video."),
                io.Int.Input("latent_frames", default=21, min=1, max=255, step=1,
                             tooltip="Video length in Wan latent frames (frames = 4 x latent - 3). Must be a multiple of the block size "
                                     "(3 for chunkwise). 21 = 81 frames, the official 5 s setting."),
                io.Boolean.Input("long_video", default=True,
                                 tooltip="On (the official SGF+ launcher, scripts/infer_sgf_plus.sh): rolling KV cache with the release "
                                         "geometry (attention sink + FIFO, positions kept in the trained range), for any length. Off: the "
                                         "plain cache of inference.py without --long_video (the 5 s training window)."),
            ],
            outputs=[io.Image.Output(display_name="images"), io.Float.Output(display_name="fps"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, prompt, seed, latent_frames, long_video) -> io.NodeOutput:
        import comfy.model_management as mm

        mm.throw_exception_if_processing_interrupted()
        loaded = runtime.acquire(model)
        info = loaded.info
        try:
            out = runtime.generate(loaded, prompt, seed, latent_frames, long_video, progress=_progress(runtime.generator_calls(loaded, latent_frames)))
        finally:
            del loaded  # drop this reference first, otherwise release() cannot free the weights
            if not model.get("keep_loaded", True):
                runtime.release()
        report = json.dumps({"model": {k: info[k] for k in ("checkpoint", "variant", "text_encoder")}, **out["report"]}, indent=1)
        return io.NodeOutput(out["images"], float(runtime.FPS), report)


def _run(model, prompt, seed, latent_frames, long_video, context):
    import comfy.model_management as mm

    mm.throw_exception_if_processing_interrupted()
    loaded = runtime.acquire(model)
    info = loaded.info
    try:
        context_latents = (context.shape[2] - 1) // 4 + 1
        out = runtime.generate(loaded, prompt, seed, latent_frames, long_video, context=context,
                               progress=_progress(runtime.generator_calls(loaded, latent_frames, context_latents)))
    finally:
        del loaded  # drop this reference first, otherwise release() cannot free the weights
        if not model.get("keep_loaded", True):
            runtime.release()
    out["report"] = {"model": {k: info[k] for k in ("checkpoint", "variant", "text_encoder")}, **out["report"]}
    return out


class SGFPlusImageToVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SGFPlusImageToVideo",
            display_name="SGF+ Image to Video",
            category="video/sgf_plus",
            description=("Starts the video from an image, as the official inference.py --i2v does: the image (resized to 832 x 480) "
                         "fills the first block, which is encoded and cached instead of generated. The first 9 frames (chunkwise) show "
                         "the image. SGF+ was trained for text to video, so the image is a strong hint rather than a guarantee."),
            inputs=[
                SGFPlusModel.Input("model"),
                io.Image.Input("image"),
                io.String.Input("prompt", multiline=True, default="The camera slowly pushes in while leaves drift in the wind."),
                io.Int.Input("seed", default=42, min=0, max=0xFFFFFFFF, control_after_generate=True),
                io.Int.Input("latent_frames", default=21, min=2, max=255, step=1,
                             tooltip="Total length in latent frames including the image block (official --num_output_frames). 21 = 81 frames."),
                io.Boolean.Input("long_video", default=True, tooltip="As in SGF+ Text to Video."),
            ],
            outputs=[io.Image.Output(display_name="images"), io.Float.Output(display_name="fps"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, image, prompt, seed, latent_frames, long_video) -> io.NodeOutput:
        block = runtime.config(model["variant"]).num_frame_per_block
        if latent_frames <= block:
            raise ValueError(f"latent_frames must be more than the image block ({block})")
        out = _run(model, prompt, seed, latent_frames - block, long_video, runtime.image_context(image, block))
        return io.NodeOutput(out["images"], float(runtime.FPS), json.dumps(out["report"], indent=1))


class SGFPlusExtendVideo(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SGFPlusExtendVideo",
            display_name="SGF+ Extend Video",
            category="video/sgf_plus",
            description=("Continues a video: its last frames (1 + 4 x (context_latents - 1), e.g. 9) are encoded and cached as the start, "
                         "then new blocks are generated with this prompt. Chain several to make a long video with changing prompts; each "
                         "run starts from the frames of the last one (the model's memory of earlier frames is not carried over)."),
            inputs=[
                SGFPlusModel.Input("model"),
                io.Image.Input("images", tooltip="the video so far (832 x 480 at 16 fps fits the model; other sizes are resized)"),
                io.String.Input("prompt", multiline=True, default="The fox stops, looks back over its shoulder, then runs on."),
                io.Int.Input("seed", default=43, min=0, max=0xFFFFFFFF, control_after_generate=True),
                io.Int.Input("context_latents", default=3, min=1, max=60, tooltip="Latent frames of context; a multiple of the block size (3 chunkwise)."),
                io.Int.Input("latent_frames", default=21, min=1, max=255, tooltip="New latent frames to generate (4 frames each); a multiple of the block size."),
                io.Boolean.Input("long_video", default=True, tooltip="As in SGF+ Text to Video."),
                io.Combo.Input("output", options=["input + new frames", "new frames"], default="input + new frames"),
            ],
            outputs=[io.Image.Output(display_name="images"), io.Float.Output(display_name="fps"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, images, prompt, seed, context_latents, latent_frames, long_video, output) -> io.NodeOutput:
        out = _run(model, prompt, seed, latent_frames, long_video, runtime.video_context(images, context_latents))
        # keep the frames of the new latents: the decoder gives 1 frame for the first latent and the same number for each later one
        frames, latents = out["images"].shape[0], out["latents"].shape[0]
        per_latent = (frames - 1) // (latents - 1) if latents > 1 else 1
        new = out["images"][frames - per_latent * latent_frames:]
        if output == "input + new frames":
            import comfy.utils

            previous = images[..., :3].float().cpu()
            if previous.shape[1:3] != new.shape[1:3]:  # bring the input to the generated size to join them
                previous = comfy.utils.common_upscale(previous.movedim(-1, 1), new.shape[2], new.shape[1], "bilinear", "disabled").movedim(1, -1)
            new = torch.cat([previous, new], dim=0)
        return io.NodeOutput(new, float(runtime.FPS), json.dumps(out["report"], indent=1))


class SGFPlusUnload(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SGFPlusUnload",
            display_name="SGF+ Unload",
            category="loaders/sgf_plus",
            description="Frees the resident SGF+ model (GPU and RAM).",
            inputs=[io.Image.Input("after", optional=True, tooltip="Optional: connect the images so this runs after them")],
            outputs=[io.String.Output(display_name="report")],
            is_output_node=True,
            not_idempotent=True,
        )

    @classmethod
    def execute(cls, after=None) -> io.NodeOutput:
        was = runtime._SLOT is not None
        runtime.release()
        text = json.dumps({"unloaded": was, "cuda_allocated_mib": torch.cuda.memory_allocated() // 2**20 if torch.cuda.is_available() else None}, indent=1)
        return io.NodeOutput(text, ui={"text": [text]})


class SGFPlusExtension(ComfyExtension):
    async def get_node_list(self):
        return [SGFPlusLoader, SGFPlusTextToVideo, SGFPlusImageToVideo, SGFPlusExtendVideo, SGFPlusUnload]


async def comfy_entrypoint() -> SGFPlusExtension:
    paths.register()
    return SGFPlusExtension()
