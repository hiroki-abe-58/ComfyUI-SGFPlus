# Verification

What was run to check this node pack, with the raw numbers. Everything below used the real released weights unless it says
"tiny model".

## 1. Setup

- Windows 11 Pro (build 26200), Intel Core Ultra 9 285K, 64 GB RAM, NVIDIA RTX 5090 32 GB (driver 610.88).
- ComfyUI v0.38.0 (commit `6b747c0428c343e1417219641db93a4fb7cb69ae`), started with `--listen 127.0.0.1 --disable-auto-launch
  --disable-api-nodes`.
- Upstream code: Zihan-Su/Self_Gradient_Forcing_Plus at `7a7734f592f1bdb51edd28f1c42ca95b9f652453`.
- Weights (SHA-256 checked, `tools/model_files.sha256`):
  - `Wan-AI/Wan2.1-T2V-1.3B` @ `37ec512624d61f7aa208f7ea8140a131f93afc9a`
  - `ZihanSu/Self_Gradient_Forcing_Plus` @ `784c07876a43fb924a2e96095b33a5675282225b`: `chunkwise/model.pt` (`53e07965…`),
    `framewise/model.pt` (`6c18725b…`)
- Two Python environments:

  | | Reference ("ref") | ComfyUI ("comfy") |
  | --- | --- | --- |
  | Python | 3.11.9 | 3.12.13 |
  | torch | 2.8.0+cu128 | 2.14.1+cu130 |
  | diffusers | 0.31.0 (upstream pin) | 0.41.0 |
  | transformers | 4.49.0 (upstream minimum) | 5.20.1 |
  | numpy | 1.24.4 (upstream pin) | 2.5.2 |

  flash-attn is not installed in either (there is no official Windows build).

## 2. Pack vs the official script (ref environment)

**Official side.** `inference.py` of the pinned commit, run unchanged through a small harness that (a) replaces
`flash_attention()` in the Wan modules with PyTorch SDPA computing the same attention (bf16 inputs, no mask, output in the input
dtype; it refuses padding lengths, windows and dropout, none of which occur) and (b) records the SHA-256 of the float32 bytes of the
`video` tensor that `inference.py` hands to `write_video` (`255 * video`, [T, 480, 832, 3]). Arguments: `--config_path
configs/sgf_plus_<variant>.yaml --checkpoint_path <variant>/model.pt --data_path <prompt file> --num_output_frames N --seed S
--use_ema`, plus for the rolling cache exactly what `scripts/infer_sgf_plus.sh` passes: `--long_video --kv_cache_max_frames 12
--kv_cache_sink 3 --kv_cache_train_frames 12 --kv_cache_position_mode contiguous` (chunkwise) or `21 / 4 / 21 / contiguous`
(framewise). `inference.py` seeds prompt number `idx` with `seed + idx`.

**Pack side.** `scripts/run_jobs.py`: the pack's runtime (the code behind the nodes), in-process, same environment, hashing the
same tensor.

| # | Variant | Cache | Latent frames | Frames | Prompt, seed | Official | Pack | Peak VRAM (both) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1 | chunkwise | plain | 21 | 81 | fox, 42 | `80d7fe59e6e41928` | `80d7fe59e6e41928` | 12.40 GiB |
| A2 | chunkwise | plain | 21 | 81 | boat, 43 | `e8b71b69b309aa2b` | `e8b71b69b309aa2b` | 12.40 GiB |
| A3 | chunkwise | rolling | 48 | 189 | fox, 42 | `89dd48c25a28219b` | `89dd48c25a28219b` | 13.45 GiB |
| A4 | framewise | plain | 21 | 81 | fox, 42 | `598be219e636f405` | `598be219e636f405` | 14.53 GiB |
| A5 | framewise | rolling | 42 | 165 | fox, 42 | `d4f736eba44524f0` | `d4f736eba44524f0` | 15.80 / 15.79 GiB |

fox = "A red fox trotting through fresh snow in a pine forest, morning light, cinematic."; boat = "A sailboat gliding across a calm
turquoise bay at sunset, gentle waves." (A1 and A2 were one official run with both prompts and `--seed 42`). A3 and A5 are longer
than the rolling cache (12 and 21 latent frames), so the cache eviction ran. **5 of 5 identical.** Generation time in this
environment: A1 13.6 s, A3 34.6 s, A4 16.3 s, A5 45.2 s (pack side).

## 3. Real ComfyUI queue (comfy environment)

ComfyUI started from a checkout with this pack in `custom_nodes`, the models in an `extra_model_paths.yaml` folder, and jobs submitted
through `POST /prompt` (`scripts/queue_client.py`). To compare exact pixels, the queue workflow saved every frame with SaveImage in
addition to Save Video; the in-process run converts frames to uint8 the same way.

| Job | Setting | In-process (same env) | Queue | Frames equal |
| --- | --- | --- | --- | --- |
| fox, 42, chunkwise | default (rolling cache), 21 | 14.89 s, 13.21 GiB | 14.84 s, 13.14 GiB | 81 / 81, max diff 0 |
| fox, 42, framewise | default (rolling cache), 21 | 19.15 s, 15.62 GiB | 18.25 s, 15.55 GiB | 81 / 81, max diff 0 |
| fox, 42 and boat, 43, chunkwise | plain cache, 21 | `06d5061c42b21950`, `ec8ec5728e71ffdd` | same frames | 2 / 2 |

Shipped workflow `t2v_long.json` (chunkwise, 81 latent frames, rolling cache): 321 frames, 56.55 s, peak 13.72 GiB, saved as MP4.
Loading took 57.7 s (chunkwise) and 60.9 s (framewise), text encoder `dynamic_swap` (30.3 GB free at load, below the official
40 GB threshold).

### Why the two environments give different videos

In the comfy environment the pack gives a different video than in the ref environment for the same seed (fox, 42, plain cache:
`06d5061c42b21950` vs `80d7fe59e6e41928`). The scene, subject and composition are the same; motion details differ. Measured:

- the prompt's token ids are identical in both environments;
- the umT5 prompt embedding (bf16 on the GPU) differs by about 1.9 % (relative) between the environments;
- the generated latent differs by about 8 % (relative L2) in the first block and the difference grows block by block to about 47 %
  in the last one, as expected when an autoregressive bf16 model conditions every block on the previous ones.

The code path is the same (section 2 shows the pack equals the official script, and within one environment the pack, the queue and
repeated runs are identical). The difference comes from the library versions (torch kernels, transformers' T5 numerics).

## 4. Real jobs and edge cases (comfy environment, real queue, chunkwise)

| Case | Result |
| --- | --- |
| 48 latent frames, rolling cache | 189 frames, 33.8 s, success |
| 81 latent frames, rolling cache | 321 frames, 56.7 s, peak 13.72 GiB, success |
| Interrupt (`POST /interrupt` during generation) | status interrupted between two generator calls |
| Next job after the interrupt | success, model still resident |
| `latent_frames` = 20 (not a multiple of 3) | the node fails before generating: "latent_frames must be a positive multiple of 3 for the chunkwise model" |
| `keep_loaded` off | success; model released after the job |
| SGF+ Unload | success, report `unloaded: true`, 32 MiB still allocated by PyTorch |

## 5. Tests without the weights (tiny model)

- `tests/` (51 tests, CPU): the vendored files against the pinned upstream commit (copied files byte for byte, patched files =
  upstream + the listed edits, syntax trees of the model and the pipeline unchanged apart from imports), the SDPA fallback against
  a float64 reference, the package scan for environment / network / process code, the download manifest, the runtime against the
  official steps re-implemented in the test (chunkwise and framewise, plain and rolling cache, with eviction), seeds, interrupt,
  bad lengths, one resident model, loading under `torch.inference_mode()`, and the nodes as ComfyUI loads them (validation of the
  shipped workflows by ComfyUI's own `validate_prompt`). The tiny model keeps the official width (1536, 12 heads; the pipeline's
  caches are sized for it) with one layer, an 8 x 8 latent, and small stand-ins for the text encoder and the VAE.
- `scripts/ci_queue_check.py`: a stock ComfyUI server with the pack, the tiny model on the CPU, the shipped workflow through the
  queue: same seed gives the same decoded video, another seed a different one, a bad length fails, rolling and plain cache run,
  Unload works.
- Both also pass with CUDA hidden (`CUDA_VISIBLE_DEVICES=-1`), as on the CPU-only CI runners.

## 6. CI

GitHub Actions (`.github/workflows/ci.yml`), CPU only, ComfyUI v0.38.0 and the upstream commit checked out at their pinned commits:
Ubuntu 24.04 with Python 3.12 / torch 2.14.1 and Python 3.10 / torch 2.8.0, Windows with Python 3.12 / torch 2.14.1. Each job runs
ruff, a scan for absolute local paths, the test suite with the references required (nothing may be skipped: 51 tests) and the
real-queue check with the tiny model. The first run failed on all three jobs because a vendored module queried the CUDA device at
import time on the CPU-only torch builds; fixed in `3e1181d` (section 5), all three jobs pass since.

## 7. Clean install

Fresh ComfyUI v0.38.0 clone, fresh Python 3.12 virtual environment (torch 2.14.1+cu130, then ComfyUI's `requirements.txt`, then
this pack's `requirements.txt`), `git clone` of this repository into `custom_nodes`:

- `tools/download_model.py --models-dir ComfyUI/models/sgf_plus` downloaded the Wan2.1 files (17.6 GB, 9.5 min) and verified all
  10 files against `tools/model_files.sha256` (the SGF+ checkpoint was placed beforehand as a hard link to the already downloaded
  file; the tool verified it and skipped the download).
- Real queue, default settings, fox / seed 42, chunkwise: 81 frames, 14.9 s, peak 13.14 GiB; **frames identical** to the
  development environment's queue run (81 / 81).
- Both shipped workflows unchanged: `t2v_basic` (its generation was served from ComfyUI's cache of the identical job before it)
  and `t2v_long` (321 frames); the saved `t2v_long` MP4 is byte-identical to the one from the development environment.
- Installing this pack's requirements moved `huggingface-hub` from 2.2.0 to 1.33.0 (diffusers 0.41.0 requires `<2.0`); ComfyUI ran
  normally.
- After the CPU-import fix (commit `3e1181d`, no change on CUDA machines) the clean installation was updated with `git pull` and
  `scripts/ci_queue_check.py --keep-pack` passed against it.
