# EvDiff: Event-Based Video Reconstruction using One-Step Diffusion Models

[![Project Page](https://img.shields.io/badge/Project_Page-EvDiff-161616)](https://liweitu.github.io/EvDiff/)
[![arXiv](https://img.shields.io/badge/arXiv-2511.17492-b31a1c)](https://arxiv.org/abs/2511.17492)
[![ECCV 2026](https://img.shields.io/badge/ECCV-2026-0B4E78)](https://eccv.ecva.net/)

Official code release for **EvDiff**, a one-step diffusion model that
reconstructs high-quality color videos from monochrome event-camera streams.

> This repository provides training, inference, data preparation, and
> [evaluation](eval/README.md) for event-based video reconstruction.

---

## Table of contents
- [Method overview](#method-overview)
- [Installation](#installation)
- [Data format](#data-format)
- [Preparing data (DSEC / BS-ERGB)](#preparing-data-dsec--bs-ergb)
- [Download weights](#download-weights)
- [Training](TRAINING.md)
- [Inference](#inference)
- [Expected output](#expected-output)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Method overview

EvDiff is a one-step event-to-video diffusion model built on Stable Diffusion 3.
It reconstructs color videos from monochrome event streams while keeping both
**high fidelity** and **temporal consistency**, and removes the dependence on
paired event–image datasets via a Surrogate Training Framework.

**Training (3 stages; see [TRAINING.md](TRAINING.md) for code and commands):**

| Stage | What | Data |
|-------|------|------|
| 1 | DiT training — synthesize E2VID-style degradation as a surrogate, train the one-step diffusion model on HQ/LQ image pairs | Places365 (1.8M images) |
| 2 | EvEncoder distillation — distill the event encoder against a frozen E2VID+ teacher (public training code) so events map into the diffusion latent space | REDS (synthetic events via v2e) |
| 3 | Joint finetune — EvEncoder + DiT + RAFT temporal loss | REDS |

**At inference**, reconstruction is **one-step** — a single diffusion forward
pass (no iterative denoising). Per frame:

1. Build a 5-bin event voxel from the events between consecutive frames.
2. Encode it through the recurrent **EvEncoder** → a 16-channel latent.
3. One **SD3 transformer** forward pass (with the LoRA adapter) at a fixed timestep.
4. Flow-matching update, then **VAE decode** → RGB frame.

The EvEncoder is recurrent, so its hidden state is warm-started across frames
within each sequence (and reset at each sequence boundary).

---

> **Training teacher:** The original experiments and released EvDiff reconstruction weights used an unpublished internal teacher. The public training code uses the publicly available [E2VID+ teacher from EVREAL](https://github.com/ercanburak/EVREAL/tree/main/pretrained/E2VID%2B) instead. The internal teacher architecture and weights are not included. Training with this substitution has not been validated to reproduce the reported results.

## Installation

```bash
conda create -n evdiff python=3.11 -y
conda activate evdiff

# 1. Install torch FIRST — pick the wheel that matches your CUDA:
#    https://pytorch.org/get-started/locally/
pip install torch torchvision  # e.g. CUDA 12.x: see pytorch.org index

# 2. Everything else:
pip install -r requirements.txt
```

**Runtime dependencies:** `torch`, `torchvision`, `diffusers>=0.30`,
`peft>=0.10`, `numpy`, `scipy`, `opencv-python`, `h5py`, `hdf5plugin`,
`omegaconf`, `tqdm`, `torchinfo`. See [`requirements.txt`](requirements.txt)
for the full rationale.

---

## Data format

Inference reads a **mem-map dataset**: a directory of sequence subfolders,
each containing `.npy` arrays. A "dataset root" looks like:

```
<data_root>/
├── interlaken_00_a/          # one folder per sequence
│   ├── events_ts.npy         # float64, [N_ev]     event timestamps, normalized seconds (first event = 0)
│   ├── events_xy.npy         # int32,  [N_ev, 2]   (x, y) coordinates
│   ├── events_p.npy          # int64,  [N_ev]      polarity, stored 0/1 (loader maps to -1/+1)
│   ├── images.npy            # uint8,  [N_img, H, W, 3]   frame images (RGB), memory-mapped
│   ├── images_ts.npy         # float64,[N_img]     frame timestamps (same time base as events_ts)
│   └── image_event_indices.npy  # int64, [N_img, 1]  searchsorted(events_ts, frame_ts) per frame
├── zurich_city_01_a/
│   └── ...
└── ...
```

Key conventions (the loader enforces / assumes these):
- **`events_p.npy` is 0/1** — the loader converts to bipolar (`p*2-1`).
- **`events_xy.npy` is `(x, y)`** column order — `data_mem.py` does `xs=xy[:,0]; ys=xy[:,1]`.
- **`events_ts.npy` is normalized to seconds** with the first event at `t=0`.
- **`image_event_indices.npy`** = `searchsorted(events_ts, frame_ts, side='right')` — the count of events *before* each frame.
- The three `events_*.npy` files are **mandatory**; the `images*.npy` files are needed for the `between_frames` voxel method used by inference.
- **Spatial resolution** must be a multiple of 16 (the VAE requires it). For DSEC this is **464 × 640** (after cropping). The loader will auto-crop to a multiple of 16 if needed.

The current public inference script builds voxels with `get_voxel_grid_fast`
in [`data/data_mem.py`](data/data_mem.py). It normalizes timestamps within each
event window, scales them to the temporal bin range, and truncates to integer
bin indices. Each event contributes its signed polarity to one bin. The default
is five bins.

---

## Preparing data (DSEC / BS-ERGB)

EvDiff reads both datasets in the
[EVREAL](https://github.com/ercanburak/EVREAL#preparing-datasets) memmap format.
The preparation steps differ by dataset.

### DSEC

Use the 12 sequences in the [official DSEC test split](https://dsec.ifi.uzh.ch/dsec-datasets/download/).
For `interlaken_00_a`, download the left events, rectified left images, left
image exposure timestamps, and calibration. Arrange the files under the
sequence directory as shown below. Save the downloaded left exposure
timestamps as `images/left/exposure_timestamps.txt`.

```text
data/DSEC/test/interlaken_00_a/
├── calibration/cam_to_cam.yaml
├── events/left/events.h5
└── images/left/
    ├── exposure_timestamps.txt
    └── rectified/*.png
```

From the repository root, call the Python API in
[`tools/dsec/convert_small_align_rgb.py`](tools/dsec/convert_small_align_rgb.py):

```bash
python - <<'PY'
from tools.dsec.convert_small_align_rgb import DSECToHQFConverter

DSECToHQFConverter(
    dsec_path="data/DSEC/test/interlaken_00_a",
    output_path="data/DSEC_mem/test/interlaken_00_a",
).convert()
PY
```

The converter remaps the rectified left RGB frames to the left event-camera
view, applies the same crop to the frames and event coordinates, and uses each
frame's exposure midpoint to index the events. The remap can leave a blank
band near the bottom of an aligned frame.

The [inference command below](#inference) uses `./data/DSEC_mem/test` as its
input root. Add `--sequences interlaken_00_a` to run that sequence alone. To
convert the full test split, arrange all sequence directories under
`data/DSEC/test` and call `batch_convert("data/DSEC/test", "data/DSEC_mem/test")`
from the same module.

### BS-ERGB

Obtain the data from the [BS-ERGB download page](https://github.com/uzh-rpg/timelens-pp)
and use the ten sequences in the
[EVREAL handheld split](https://github.com/ercanburak/EVREAL/blob/main/config/dataset/BS_ERGB_handheld.json).
Place the prepared sequences in the memmap layout above and pass their parent
directory as `--input_dir`. Keep the GT frames and events in the same sensor
coordinates. For the evaluation crop, see [Evaluation](#evaluation).

---

## Download weights

See [`WEIGHTS.md`](WEIGHTS.md) for the full manifest. You need **two** things:

1. **Stable Diffusion 3 Medium** (base diffusion model) → passed as `--sd3_path`.
   ```bash
   huggingface-cli download stabilityai/stable-diffusion-3-medium-diffusers \
       --local-dir ./checkpoints/sd3-medium
   ```
   *(gated repo — accept the license on the model page first)*

2. **EvDiff trained checkpoint** (`voxel_vae.pth` + `transformer.safetensors`)
   → passed as `--checkpoint`.
   ```bash
   huggingface-cli download litutu135/EvDiff --local-dir ./checkpoints/evdiff
   ```

The **prompt embeddings** (`dataset/default/prompt_embeds.pt`,
`dataset/default/pool_embeds.pt`) are **already in this repo** — precomputed
against SD3 Medium's text encoders (CLIP-L, CLIP-G, T5-XXL), so the text
encoders are never loaded at inference.

---

## Inference

Run from the repo root (the script does `sys.path.append(os.getcwd())`, so run
from the directory that contains `models/`, `data/`, `utils/`):

```bash
python test/infer_e2vid.py \
    --checkpoint   ./checkpoints/evdiff/checkpoint \
    --sd3_path     ./checkpoints/sd3-medium \
    --embedding_dir dataset/default \
    --input_dir    ./data/DSEC_mem/test \
    --output_dir   ./outputs \
    --model_name   evdiff
```

### Arguments

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | *(required)* | EvDiff checkpoint dir with `voxel_vae.pth` + `transformer.safetensors` |
| `--sd3_path` | *(required)* | SD3 Medium dir (with `scheduler/`, `transformer/`, `vae/`) |
| `--input_dir` | *(required)* | mem-map dataset root (sequence subfolders) |
| `--output_dir` | *(required)* | prediction root; PNGs → `<output_dir>/<model_name>/<seq>/` |
| `--embedding_dir` | `dataset/default` | dir with `prompt_embeds.pt` + `pool_embeds.pt` |
| `--model_name` | `vae_color_raft_base` | subfolder name under `--output_dir` |
| `--input_channels` | `5` | event voxel bins (must match training) |
| `--output_channels` | `16` | EvEncoder latent channels |
| `--base_channels` | `96` | EvEncoder base width |
| `--temporal_fusion` | `gated` | EvEncoder temporal fusion type |
| `--rank` | `64` | transformer LoRA rank |
| `--num_bins` | `5` | voxel bins (matches `--input_channels`) |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--inference_timestep` | `680.0` | fixed flow-matching timestep (matches training) |
| `--max_sequences` | *(none)* | optional cap on #sequences (smoke test) |
| `--sequences` | *(none)* | comma-separated sequence names to run (default: all) |

> **Architecture flags must match the checkpoint.** The defaults above are the
> EvDiff config — only change them if you trained a different variant.

### Notes
- **Inference runs in fp32.**
- **Models stay in train mode** (not `.eval()`). This is intentional — the
  recurrent EvEncoder's norm/dropout/state logic depends on `self.training`,
  and calling `.eval()` produces washed-out reconstructions. The script sets
  `requires_grad_(False)` and wraps the forward in `torch.no_grad()`.
- **One-step diffusion**: only one transformer forward pass (no iterative
  denoising).

---

## Expected output

```
<output_dir>/<model_name>/<seq>/<i:04d>.png
```

For example, the first output is
`./outputs/evdiff/interlaken_00_a/0000.png`. For `i > 0`, output `i` uses the
events between frames `i-1` and `i` and corresponds to `images.npy[i]`.
Output `0000.png` uses the events up to the first frame.

---

## Evaluation

See the [evaluation guide](eval/README.md) for the MSE, SSIM, LPIPS, FID,
and FVD command, dataset splits, and reference results. The evaluator pairs
predictions by frame index and scores the same image region for all methods.

For BS-ERGB, the legacy EvDiff evaluator used a center crop of the reference
frames that was five pixels to the right of the prediction region. The current
evaluation uses a shared region in event-camera coordinates. Its reference
scores use the protocol described in the evaluation guide.

---

## Citation

If you use this work, please cite:

```bibtex
@misc{li2025evdiffhighqualityvideo,
      title        = {EvDiff: High Quality Video with an Event Camera},
      author       = {Weilun Li and Lei Sun and Ruixi Gao and Qi Jiang and Yuqin Ma and
                      Kaiwei Wang and Ming-Hsuan Yang and Luc Van Gool and Danda Pani Paudel},
      year         = {2025},
      eprint       = {2511.17492},
      archivePrefix= {arXiv},
      primaryClass = {cs.CV},
      url          = {https://arxiv.org/abs/2511.17492}
}
```

**Paper:** [arXiv:2511.17492](https://arxiv.org/abs/2511.17492)

> **Note.** The arXiv version currently uses the earlier CVPR title
> *“EvDiff: High Quality Video with an Event Camera”*; the camera-ready ECCV
> version is titled *“EvDiff: Event-Based Video Reconstruction using One-Step
> Diffusion Models”*. The content is essentially the same. The arXiv entry will
> be updated to the ECCV version in due course; the BibTeX above is the
> citable reference for now.

## Acknowledgements

- The diffusion model is built on [Stable Diffusion 3](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
  by Stability AI.
- The memmap data format follows the [EVREAL](https://github.com/ercanburak/EVREAL) benchmark.
- Event data and evaluation use the [DSEC](https://github.com/uzh-rpg/DSEC) and [BS-ERGB](https://github.com/uzh-rpg/timelens-pp) datasets.
- The SD3 MMDiT adapter wiring uses [🤗 PEFT](https://github.com/huggingface/peft)
  and [🤗 Diffusers](https://github.com/huggingface/diffusers).

---

## License

This project is released under the [MIT License](LICENSE).

> **The base diffusion model (SD3 Medium) is governed by Stability AI's own
> license** — you must accept it on the [HF model page](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers)
> before downloading. This MIT license covers only the code and trained
> checkpoint released in *this* repository.
