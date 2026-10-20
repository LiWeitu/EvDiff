# Training EvDiff

This release contains the three training stages with a public E2VID+ teacher. The training code lives
in `training/`, with its own model and dataset modules so that the existing
inference code remains independent. The launchers use Accelerate on one CUDA GPU.
No Slurm submission scripts or REDS event-simulation pipeline are included.

Training and inference are provided in the same EvDiff repository. Run the
commands below from the repository root. The training package uses the same
released prompt embeddings and leaves the inference modules unchanged.

**Difference from the original experiments:** The original experiments and the
released EvDiff reconstruction weights used an unpublished internal teacher.
This public training code substitutes the publicly available E2VID+ teacher.
The internal teacher architecture, configuration, and weights are not included.
The existing EvDiff inference weights remain available and unchanged. Results
after retraining with E2VID+ may differ; equivalence to the reported results has
not been established.

## Installation and inputs

Use the existing EvDiff repository, install a matching PyTorch/torchvision CUDA
build, then install the additional training dependencies:

```bash
pip install -r requirements-training.txt
```

The dependency file lists the required packages; it is not a tested version lock.
The launchers use TensorBoard by default. To use W&B, install `wandb`, set
`REPORT_TO=wandb`, and configure authentication in your environment.

You will need:

- Stable Diffusion 3 Medium in Diffusers format. Training loads the three text
  encoders and tokenizers in addition to the transformer, VAE, and scheduler.
  An inference-only SD3 download may therefore be incomplete for training.
- Places365 images under `<PLACES365_PATH>/gt/`. Class subfolders are discovered
  recursively. Per-image text prompts are optional with the provided launcher.
- Precomputed REDS event H5 sequences under `small1/event/*.h5` and
  `small2/event/*.h5`. Both roots are required by the supplied configuration.
- The public E2VID+ teacher checkpoint from EVREAL for stage 2 (see below).
- BS-ERGB validation sequences in the memory-mapped layout used by the existing
  data-preparation tools, for stage 3.
- The existing repository's `dataset/default/` and `dataset/null/` prompt
  embeddings (`prompt_embeds.pt` and `pool_embeds.pt`).

Datasets, SD3, and the E2VID+ teacher weights are supplied separately. Stage 3
uses torchvision's pretrained RAFT-Large; its weights must be downloadable or
already present in the torch cache.

## Configure paths

```bash
cp script/training/paths.example.sh script/training/paths.local.sh
# Edit paths.local.sh for your machine.
```

All dataset paths are environment variables. `paths.local.sh` is ignored by Git.
Keep `REDS` in the REDS H5 paths: the baseline readers use this case-sensitive
substring to enable sequence merging. Online voxelization, augmentation, and
frame merging are part of the dataset readers. These readers consume existing
events; REDS event generation and its simulation settings are not distributed.

The H5 readers expect an `events` group with one-dimensional `xs`, `ys`, `ts`,
and `ps` datasets. The `images` group contains grayscale frames in frame-key
order; each image carries an `event_idx` attribute delimiting the event window.
Stage 3 also reads matching `rgb_images` frames and scalar entries in `tags`.
This describes the reader interface only; it does not provide a simulation or
conversion recipe.

Each BS-ERGB sequence folder needs `events_ts.npy`, `events_xy.npy`, `events_p.npy`,
`images.npy`, `images_ts.npy`, and `image_event_indices.npy`. Optional
`metadata.json` contains `sensor_resolution` as `[height, width]`.

## Run the three stages

Run in order. The launchers execute in the foreground with `tee` and `pipefail`,
and stop if CUDA is unavailable. They respect the GPU visibility assigned by
your environment. Arguments appended to the launcher override its defaults.

```bash
bash script/training/step1.sh
# Select a saved stage-1 checkpoint and set STEP1_CHECKPOINT.
bash script/training/step2.sh
# Select a saved stage-2 voxel_vae.pth and set STEP2_VOXEL_VAE.
bash script/training/step3.sh
```

| Stage | Training | Input checkpoints | Saved model files |
|---|---|---|---|
| 1 | Grayscale surrogate degradation; SD3 transformer and VAE encoder LoRA | SD3 Medium | `transformer.safetensors`, `vae.safetensors` |
| 2 | Event encoder distillation against frozen E2VID+ | SD3, stage-1 LoRA, public E2VID+ teacher | `voxel_vae.pth` |
| 3 | Event encoder and transformer LoRA joint finetuning with RAFT temporal loss | SD3, stage-1 transformer LoRA, stage-2 event encoder | `transformer.safetensors`, `voxel_vae.pth` |

Stage 2 uses `STEP1_CHECKPOINT` as `--vae_lora_dir`. Stage 3 uses that directory
as `--lora_dir` and loads the separate stage-2 file via `--init_voxel_vae_path`.
There is no need to copy or assemble intermediate checkpoint directories.

Outputs default to `outputs/step1`, `outputs/step2`, and `outputs/step3`; logs go
to `logs/`. Resume defaults to the latest checkpoint in that output directory.
Use a fresh output directory for a new run. For example, a short launch is:

```bash
OUTPUT_DIR=outputs/smoke_step1 OUTPUT_LOG=logs/smoke_step1.log \
  bash script/training/step1.sh --max_train_steps 2 --train_batch_size 1
```

The supplied defaults retain the original launchers' budgets and hyperparameters
(400k/60k/60k maximum steps, rank 64, learning rate 5e-6). They are configurable
training budgets, not a claim that the published reconstruction weights were
selected at those exact endpoints. Full training is memory intensive; the
release has not established a minimum GPU-memory requirement. Adjust the batch
size and accumulation for your hardware.

## Public E2VID+ teacher checkpoint

Download `pretrained/E2VID+/model.pth` from the public
[EVREAL repository](https://github.com/ercanburak/EVREAL/tree/main/pretrained/E2VID%2B)
and set `E2VIDPLUS_CHECKPOINT` in `paths.local.sh` to that file. A pinned download:

```bash
mkdir -p checkpoints/e2vidplus
curl -L 'https://raw.githubusercontent.com/ercanburak/EVREAL/398f03551b7f5d150ae10aa89e4e731557fce240/pretrained/E2VID%2B/model.pth' \
  -o checkpoints/e2vidplus/model.pth
```

The E2VID+ architecture and necessary layers are included under
`training/teachers/e2vidplus/`, with upstream attribution and the MIT license.
The loader reads the official legacy `ConfigParser` checkpoint directly, checks
the architecture, and loads its state dictionary strictly. No private checkpoint
or private model module is required. The old custom-teacher export tool has been
removed from this release.

The teacher runs frozen in fp32, with its recurrent state reset per sequence.
Inputs are five-bin event voxels. Right/bottom padding to the encoder stride is
cropped from the output. Following EVREAL's E2VID+ method configuration, the
adapter does not normalize event tensors or normalize/clamp reconstruction
outputs. The existing stage-2 image-to-VAE transform and losses are retained.
Training stability and reconstruction quality with this teacher substitution
still require validation on the training data.

## Validation scope

This addition is prepared from the baseline research sources and public EVREAL
teacher sources. Python syntax,
local imports, shell syntax, launcher arguments, and publication exclusions are
checked during packaging. The official E2VID+ checkpoint also passes strict
weight loading, a three-frame CPU forward check, recurrent-state reset, and
padding/cropping checks. For stride-aligned inputs, the adapter's reconstruction
matches the unmodified public reference implementation exactly in that check.
A CUDA training smoke run and a full reproduction with the external training
datasets remain to be performed; these CPU checks do not establish quality
equivalence after the teacher substitution.
