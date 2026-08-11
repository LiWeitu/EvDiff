# Weight Manifest

This repo ships **no model weights**. You need to obtain two things to run
inference:

1. The **base diffusion model** (Stable Diffusion 3 Medium).
2. The **trained EvDiff checkpoint** (our weights).

The prompt embeddings (`dataset/default/*.pt`, `dataset/null/*.pt`) **are**
included in this repo — you do **not** need to download them.

---

## Quick reference

| # | Weight | Where it goes | How it's passed at inference |
|---|--------|---------------|------------------------------|
| 1 | Stable Diffusion 3 Medium | `--sd3_path <dir>` | a HF-format dir with `scheduler/`, `transformer/`, `vae/` subfolders |
| 2 | EvDiff trained checkpoint dir | `--checkpoint <dir>` | a dir containing **both** `voxel_vae.pth` and `transformer.safetensors` |

The EvDiff checkpoint already bundles the EvEncoder (voxel VAE) and the SD3
transformer LoRA, so you need nothing else for inference.

---

## 1. Stable Diffusion 3 Medium (base diffusion model)

EvDiff is hard-bound to the SD3 architecture: `SD3Transformer2DModel` (MMDiT),
a 16-channel latent VAE, and `FlowMatchEulerDiscreteScheduler`. The prompt
embeddings shipped in `dataset/default/` were generated against SD3 Medium's
three text encoders (CLIP-L, CLIP-G, T5-XXL), so use **SD3 Medium** to match.

| | |
|---|---|
| **Official source** | [`stabilityai/stable-diffusion-3-medium-diffusers`](https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers) (Hugging Face, gated — accept the license) |
| **Download** | `huggingface-cli download stabilityai/stable-diffusion-3-medium-diffusers --local-dir ./checkpoints/sd3-medium` |

The local dir must contain at least:
```
<sd3_path>/
├── scheduler/scheduler_config.json
├── transformer/                 # SD3Transformer2DModel weights
└── vae/                         # AutoencoderKL weights (16 latent channels)
```
(The text encoders are **not** loaded at inference — we use the precomputed
embeddings — so you can skip `text_encoder/`, `text_encoder_2/`, `tokenizer*/`
to save space.)

---

## 2. EvDiff trained checkpoint

Our trained reconstruction model. The checkpoint dir contains **both**:

- `voxel_vae.pth` — the EvEncoder (event-voxel → latent VAE).
- `transformer.safetensors` — the SD3 transformer LoRA weights (rank 64).

| | |
|---|---|
| **Hugging Face repo** | [`litutu135/EvDiff`](https://huggingface.co/litutu135/EvDiff) |
| **Layout** | `checkpoint/voxel_vae.pth`, `checkpoint/transformer.safetensors` |
| **Download** | `huggingface-cli download litutu135/EvDiff --local-dir ./checkpoints/evdiff` |

After download, pass the checkpoint dir to inference:

```bash
python test/infer_e2vid.py \
    --checkpoint ./checkpoints/evdiff/checkpoint \
    --sd3_path   ./checkpoints/sd3-medium \
    ...
```

> Note: `huggingface-cli download` puts the files under
> `<local-dir>/checkpoint/`, so `--checkpoint` points at that subfolder.

Architecture defaults the inference script assumes (must match training —
change via CLI flags if your checkpoint differs):
`--input_channels 5 --output_channels 16 --base_channels 96
--temporal_fusion gated --rank 64 --num_bins 5`.

---

## Verification after placement

Quick sanity check that the two paths resolve to the right files:

```bash
# 1. SD3 base model — should list scheduler/, transformer/, vae/
ls <sd3_path>/{scheduler,transformer,vae}/

# 2. EvDiff checkpoint — should list BOTH files:
ls <checkpoint>/voxel_vae.pth <checkpoint>/transformer.safetensors
```

Then run inference (see README §Inference). If `--checkpoint` is missing
`voxel_vae.pth`, the script crashes at `load_state_dict`; if the LoRA rank or
voxel-VAE channel counts don't match, you'll see a shape-mismatch error —
adjust the corresponding `--rank` / `--output_channels` / `--base_channels` /
`--temporal_fusion` flag to match how the checkpoint was trained.
