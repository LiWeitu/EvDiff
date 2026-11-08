# Reconstruction evaluation

This evaluator scores saved PNG reconstructions against the `images.npy` files
from the DSEC or BS-ERGB data preparation step. It uses the same scoring frames
and image region for each method evaluated on a given split.

## Run

Install PyTorch as described in the [main README](../README.md), then install the
evaluation dependencies from the repository root.

```bash
pip install -r eval/requirements.txt
```

For FVD, download the
[I3D TorchScript checkpoint](https://www.dropbox.com/s/ge9e5ujwgetktms/i3d_torchscript.pt?dl=1)
from the [StyleGAN-V evaluation implementation](https://github.com/cvpr2022-stylegan-v/stylegan-v/blob/main/src/metrics/frechet_video_distance.py).
Its SHA-256 is `bec6519f66ea534e953026b4ae2c65553c17bf105611c746d904657e5860a5e2`.
PyIQA and pytorch-fid fetch their pretrained weights on first use.

```bash
python eval/evaluate.py \
  --dataset DSEC \
  --gt-root ./data/DSEC_mem/test \
  --pred-root ./outputs/evdiff \
  --i3d-path ./checkpoints/i3d_torchscript.pt \
  --output ./outputs/dsec_metrics.json
```

The prediction root is the model folder inside the inference output directory.
For example, it contains `interlaken_00_a/0001.png`, and the matching reference
is `interlaken_00_a/images.npy[1]` under the reference root. Numeric names and
`frame_`-prefixed numeric names are supported. Use `--dataset BS_ERGB` for the
BS-ERGB handheld split. For a single-sequence check, add
`--sequences interlaken_00_a`. Use `--image-metrics-only` to run MSE, SSIM and
LPIPS without an I3D checkpoint.

## Frame pairing and image regions

[splits.json](splits.json) lists the reference frame ranges and evaluation
rectangles for DSEC (12 sequences) and BS-ERGB (10 sequences). End indices are
exclusive. For a sequence with `T` reference frames, the evaluator scores
frames `1` through `T-2`. For EvDiff inference the PNG ID equals the reference
index. For other outputs, `--index-offset` specifies the PNG ID minus the
reference index, and `--pred-origin TOP LEFT` specifies the prediction's origin
in the reference coordinate system. Use the same split and sequence subset for
all methods. Missing or duplicate prediction IDs and incomplete image regions
raise errors.

The legacy EvDiff BS-ERGB evaluator center-cropped the reference to match a
`624 × 960` prediction. For top-left-aligned predictions, this shifted the
reference crop five pixels to the right. This evaluator selects the same
region in both the reference and prediction using their coordinates. The
BS-ERGB rectangles account for small border differences across methods. The
historical EvDiff and EVREAL-baseline entry points handled the reference crop
differently.

## Metrics and reference results

MSE, SSIM and LPIPS are computed on grayscale frames. RGB is converted with
OpenCV and replicated to three channels for PyIQA. The scores are averaged over
frames within each sequence and then averaged equally across sequences.

FID uses RGB Inception pool3 features from pytorch-fid, with up to 512 uniformly
sampled frames per sequence. FVD uses I3D features from non-overlapping
contiguous 16-frame RGB clips, with up to 128 uniformly selected clips per
sequence. Each distance is computed from features pooled across the dataset
or selected subset.

The table below records EvDiff-base scores from archived predictions evaluated
with this common-region protocol. Inference settings for those predictions may
differ from the current public runner, including latent sampling seeds. The
historical paper scores used a separate evaluator with different geometry or
feature aggregation.

| Dataset | Frames scored | MSE ↓ | SSIM ↑ | LPIPS ↓ | FID ↓ | FVD-16 ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BS-ERGB | 4,581 | 0.05866 | 0.35264 | 0.43580 | 91.83 | 694.79 |
| DSEC | 11,180 | 0.05895 | 0.36770 | 0.45851 | 69.04 | 741.36 |
