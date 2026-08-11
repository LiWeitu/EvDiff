#!/usr/bin/env python
# coding=utf-8
"""
Clean step-3 inference for the E2VID color-reconstruction model.

Loads a step-3 checkpoint dir (which contains BOTH `voxel_vae.pth` and
`transformer.safetensors`, as written by train_color_vae_raft_v5_EVDiff.py's save hook),
runs single-step flow-matching inference over every sequence in `--input_dir`, and writes
reconstructed PNGs to `<output_dir>/<model_name>/<seq>/<i:04d>.png` — the exact layout that
`eval/eval_all_gray.py` expects.

The inference math is copied verbatim from the TRAINING VALIDATION loop
(train/train_color_vae_raft_v5_EVDiff.py:1030-1062): recurrent voxel-VAE state warm-up ->
single transformer forward at timestep=680 -> get_sigmas -> Euler update -> VAE decode.
The only difference from the val loop is that we reconstruct and save EVERY frame of EVERY
sequence (the val loop only keeps the last frame of a random segment for a quick metric).

Voxel construction reuses data.data_mem.MemMapDatasetVal (its `.sequences` metadata +
`.get_voxel_grid_fast`), so voxels are built identically to training. We never invoke the
random-segment sampler.
"""
import argparse
import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch
from torchvision.transforms import ToPILImage
from tqdm import tqdm

from peft import LoraConfig
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)

from models.autoencoder_kl import AutoencoderKL
from models.voxel_model import OptimizedEventVAEEncoderv5
from utils.util import load_lora_state_dict
from data.data_mem import MemMapDatasetVal


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="step-3 checkpoint dir containing voxel_vae.pth AND transformer.safetensors")
    p.add_argument("--sd3_path", type=str, required=True,
                   help="SD3 model dir (--pretrained_model_name_or_path)")
    p.add_argument("--embedding_dir", type=str, default="dataset/default",
                   help="dir with prompt_embeds.pt and pool_embeds.pt")
    p.add_argument("--input_dir", type=str, required=True,
                   help="dataset root whose subfolders are sequences")
    p.add_argument("--output_dir", type=str, required=True,
                   help="prediction root; PNGs go to <output_dir>/<model_name>/<seq>/")
    p.add_argument("--model_name", type=str, default="vae_color_raft_base",
                   help="folder name under output_dir (the 'model' dimension eval_all_gray expects)")
    # model architecture (must match training)
    p.add_argument("--input_channels", type=int, default=5)
    p.add_argument("--output_channels", type=int, default=16, help="voxel-VAE latent channels")
    p.add_argument("--base_channels", type=int, default=96)
    p.add_argument("--temporal_fusion", type=str, default="gated")
    p.add_argument("--rank", type=int, default=64, help="transformer LoRA rank")
    p.add_argument("--num_bins", type=int, default=5)
    # runtime
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--inference_timestep", type=float, default=680.0,
                   help="fixed flow-matching timestep (matches training val loop)")
    p.add_argument("--max_sequences", type=int, default=None,
                   help="optional cap on #sequences (smoke test)")
    p.add_argument("--sequences", type=str, default=None,
                   help="只跑这些序列，逗号分隔（如 thun_00_a,zurich_city_01_a）。"
                        "不传则跑 input_dir 下全部序列")
    return p.parse_args()


def get_sigmas(noise_scheduler, timesteps, n_dim=4, dtype=torch.float32, device="cuda"):
    """Copied from train_color_vae_raft_v5_EVDiff.py:853-866."""
    sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(device)
    timesteps = timesteps.to(device)
    step_indices = []
    for t in timesteps:
        idx = (schedule_timesteps - t).abs().argmin().item()
        step_indices.append(idx)
    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def build_voxel_for_frame(dataset, filehandle, event_indices, frame_idx, sensor_resolution):
    """Reproduce the between_frames voxel build from data_mem.py:341-378 for a single frame."""
    if frame_idx > 0:
        _, idx0 = event_indices[frame_idx - 1]
    else:
        idx0 = 0
    _, idx1 = event_indices[frame_idx]

    num_bins = dataset.num_bins
    if idx1 > idx0:
        xy = filehandle["xy"][idx0:idx1]
        ts = filehandle["t"][idx0:idx1]
        ps = filehandle["p"][idx0:idx1]

        xs = xy[:, 0].astype(np.float32)
        ys = xy[:, 1].astype(np.float32)
        ps = ps * 2.0 - 1.0

        ts_0 = ts[0]
        xs = torch.from_numpy(xs)
        ys = torch.from_numpy(ys)
        ts_torch = torch.from_numpy((ts - ts_0).astype(np.float32))
        ps = torch.from_numpy(ps.astype(np.float32))

        voxel = dataset.get_voxel_grid_fast(xs, ys, ts_torch, ps, sensor_resolution)
    else:
        voxel = torch.zeros((num_bins, *sensor_resolution), dtype=torch.float32)
    return voxel


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.float32  # fp32 inference (training used --mixed_precision="no")

    # ---------------- models ----------------
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.sd3_path, subfolder="scheduler")
    transformer = SD3Transformer2DModel.from_pretrained(args.sd3_path, subfolder="transformer", torch_dtype=weight_dtype)
    vae = AutoencoderKL.from_pretrained(args.sd3_path, subfolder="vae", torch_dtype=weight_dtype)
    voxel_vae = OptimizedEventVAEEncoderv5(
        input_channels=args.input_channels,
        latent_channels=args.output_channels,
        base_channels=args.base_channels,
        temporal_fusion=args.temporal_fusion,
    )

    # load voxel-VAE weights
    state = torch.load(os.path.join(args.checkpoint, "voxel_vae.pth"), map_location="cpu", weights_only=False)
    state = state["state_dict"] if "state_dict" in state else state
    voxel_vae.load_state_dict(state, strict=True)

    # add + load transformer LoRA
    transformer.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0", "add_q_proj",
                            "add_k_proj", "add_v_proj", "proj", "linear", "proj_out"],
        )
    )
    transformer.enable_adapters()
    lora_state = StableDiffusion3Pipeline.lora_state_dict(args.checkpoint, weight_name="transformer.safetensors")
    load_lora_state_dict(lora_state, transformer)

    # prompt embeddings
    prompt_embeds = torch.load(os.path.join(args.embedding_dir, "prompt_embeds.pt"), map_location="cpu").to(weight_dtype)
    pooled_embeds = torch.load(os.path.join(args.embedding_dir, "pool_embeds.pt"), map_location="cpu").to(weight_dtype)
    if prompt_embeds.dim() == 2:
        prompt_embeds = prompt_embeds.unsqueeze(0)  # [1, L, C]
    if pooled_embeds.dim() == 1:
        pooled_embeds = pooled_embeds.unsqueeze(0)  # [1, P]

    # IMPORTANT: do NOT call .eval() on these models. The reference inference
    # (TSD_main/test/test_h5_batch_e2vid.py:256-262) and the training val loop both
    # run transformer/vae/voxel_vae in TRAIN mode -- only requires_grad_(False) + the
    # outer torch.no_grad(). voxel_vae is a recurrent net whose norm/dropout/state
    # logic depends on self.training; calling .eval() switches it to (uncalibrated)
    # running stats and yields washed-out, blurry reconstructions even with a good
    # checkpoint. Keep them in train mode to match the validated path.
    transformer = transformer.to(device, dtype=weight_dtype)
    vae = vae.to(device, dtype=weight_dtype)
    voxel_vae = voxel_vae.to(device, dtype=weight_dtype)
    for m in (transformer, vae, voxel_vae):
        m.requires_grad_(False)
    transformer.train(); vae.train(); voxel_vae.train()
    prompt_embeds = prompt_embeds.to(device, dtype=weight_dtype)
    pooled_embeds = pooled_embeds.to(device, dtype=weight_dtype)

    timesteps = torch.tensor([args.inference_timestep], device=device)

    # ---------------- data (sequence metadata only) ----------------
    dataset = MemMapDatasetVal(
        data_path=args.input_dir,
        num_bins=args.num_bins,
        voxel_method={"method": "between_frames"},
        segment_length=2,   # dummy: only .sequences + get_voxel_grid_fast are used
        num_samples=1,
        seed=42,
    )

    sequences = dataset.sequences
    if args.max_sequences is not None:
        sequences = sequences[: args.max_sequences]
    if args.sequences:
        want = {s.strip() for s in args.sequences.split(",") if s.strip()}
        sequences = [s for s in sequences if s["name"] in want]
        missing = want - {s["name"] for s in sequences}
        if missing:
            print(f"[warn] --sequences 里这些序列在数据集中没找到: {sorted(missing)}")

    print(f"Inference over {len(sequences)} sequences -> {args.output_dir}/{args.model_name}/")

    for seq in tqdm(sequences, desc="sequences"):
        seq_name = seq["name"]
        length = seq["length"]                 # num_frames - 1
        event_indices = seq["event_indices"]
        filehandle = seq["filehandle"]
        sensor_resolution = seq["sensor_resolution"]

        out_dir = os.path.join(args.output_dir, args.model_name, seq_name)
        os.makedirs(out_dir, exist_ok=True)

        voxel_vae.reset_states()

        for frame_idx in range(length):
            voxel = build_voxel_for_frame(dataset, filehandle, event_indices, frame_idx, sensor_resolution)
            events_t = voxel.unsqueeze(0).to(device, dtype=weight_dtype)  # [1, num_bins, H, W]

            with torch.no_grad():
                model_input, _, _ = voxel_vae(events_t, sample=True)

                model_pred = transformer(
                    hidden_states=model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_embeds,
                    return_dict=False,
                )[0]

                sigmas = get_sigmas(noise_scheduler, timesteps, n_dim=model_input.ndim, dtype=model_input.dtype, device=device)
                latent_stu = model_pred * (-sigmas) + model_input

                image = vae.decode(latent_stu / vae.config.scaling_factor, return_dict=False)[0].clamp(-1, 1)
                image = image * 0.5 + 0.5  # -> [0, 1]

            ToPILImage()(image[0].cpu()).save(os.path.join(out_dir, f"{frame_idx:04d}.png"))

    print("Done.")


if __name__ == "__main__":
    main()