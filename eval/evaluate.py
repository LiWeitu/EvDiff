"""Score saved event-to-video reconstructions on a fixed dataset split."""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def png_indices(directory):
    paths = {}
    for path in directory.glob("*.png"):
        stem = path.stem.removeprefix("frame_")
        if not stem.isascii() or not stem.isdecimal():
            raise ValueError(f"Expected a numeric PNG frame ID: {path}")
        index = int(stem)
        if index in paths:
            raise ValueError(f"Duplicate prediction ID {index} in {directory}")
        paths[index] = path
    return paths


def rgb(frame):
    if frame.ndim == 2:
        frame = frame[..., None]
    if frame.shape[-1] == 1:
        return np.repeat(frame, 3, axis=-1)
    return frame


def sequence_readers(args, sequence, view):
    top, left, bottom, right = view["roi"]
    frames = np.load(args.gt_root / sequence / "images.npy", mmap_mode="r", allow_pickle=False)
    if frames.dtype != np.uint8 or frames.ndim not in (3, 4) or (
        frames.ndim == 4 and frames.shape[-1] not in (1, 3)
    ):
        raise ValueError(f"{sequence}: expected uint8 grayscale or RGB reference frames")
    start, stop = view["frames"]
    if len(frames) != stop + 1 or bottom > frames.shape[1] or right > frames.shape[2]:
        raise ValueError(f"{sequence}: reference length or size differs from the split")

    paths = png_indices(args.pred_root / sequence)
    missing = [index for index in range(start, stop) if index + args.index_offset not in paths]
    if missing:
        raise ValueError(f"{sequence}: missing {len(missing)} predictions, first GT IDs {missing[:8]}")
    origin_y, origin_x = args.pred_origin
    pred_rect = (left - origin_x, top - origin_y, right - origin_x, bottom - origin_y)
    if min(pred_rect) < 0:
        raise ValueError(f"{sequence}: prediction origin does not cover the evaluation region")

    def reference(index):
        return rgb(frames[index, top:bottom, left:right])

    def prediction(index):
        path = paths[index + args.index_offset]
        with Image.open(path) as image:
            if image.width < pred_rect[2] or image.height < pred_rect[3]:
                raise ValueError(f"{path}: prediction does not cover the evaluation region")
            return np.asarray(image.convert("RGB").crop(pred_rect))

    return reference, prediction


def tensor(frames, device, gray=False):
    if gray:
        frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in frames]
        frames = np.repeat(np.stack(frames)[..., None], 3, axis=-1)
    else:
        frames = np.stack(frames)
    return torch.from_numpy(frames).permute(0, 3, 1, 2).to(device, torch.float32) / 255


def sampling(start, stop):
    fid_ids = np.linspace(start, stop - 1, min(512, stop - start), dtype=int)
    clip_starts = np.arange(start, stop - 15, 16, dtype=int)
    if len(clip_starts) > 128:
        clip_starts = clip_starts[np.linspace(0, len(clip_starts) - 1, 128, dtype=int)]
    return fid_ids, clip_starts


def i3d_features(clip, detector):
    # Match the RGB preprocessing of the shared evaluation. The detector
    # receives one C x T x H x W clip in [-1, 1] after a spatial center crop.
    height, width = clip.shape[-2:]
    size = 224
    scale = size / min(height, width)
    target_size = (size, math.ceil(width * scale)) if height <= width else (math.ceil(height * scale), size)
    resized = F.interpolate(
        clip.permute(1, 0, 2, 3), size=target_size,
        mode="bilinear", align_corners=False,
    )
    top, left = (resized.shape[-2] - size) // 2, (resized.shape[-1] - size) // 2
    cropped = resized[:, :, top:top + size, left:left + size]
    video = ((cropped - 0.5) * 2).unsqueeze(0).contiguous()
    return detector(video, rescale=False, resize=False, return_features=True).cpu().numpy()


def covariance_factor(features, mean):
    centered = (features - mean) / math.sqrt(len(features) - 1)
    if centered.shape[0] > centered.shape[1]:
        return np.linalg.qr(centered, mode="r")
    return centered


def frechet(reference, prediction):
    reference, prediction = np.asarray(reference, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if min(len(reference), len(prediction)) < 2:
        raise ValueError("Frechet distance needs at least two samples per side")
    real_mean, pred_mean = reference.mean(axis=0), prediction.mean(axis=0)
    real_factor = covariance_factor(reference, real_mean)
    pred_factor = covariance_factor(prediction, pred_mean)
    # For sample covariances R.T @ R and P.T @ P, the square-root trace
    # is the sum of singular values of R @ P.T. This also handles low rank.
    cross_trace = np.linalg.svd(real_factor @ pred_factor.T, compute_uv=False).sum()
    covariance_trace = np.square(real_factor).sum() + np.square(pred_factor).sum()
    delta = real_mean - pred_mean
    value = float(delta @ delta + covariance_trace - 2 * cross_trace)
    if not np.isfinite(value) or value < -1e-8 * max(1.0, covariance_trace):
        raise ValueError(f"Invalid Frechet distance: {value}")
    return max(0.0, value)


def frame_metrics(readers, start, stop, metrics, args):
    sums = dict.fromkeys(("mse", *metrics), 0.0)
    for first in range(start, stop, args.batch_size):
        indices = range(first, min(first + args.batch_size, stop))
        reference, prediction = [
            tensor([read(index) for index in indices], args.device, gray=True) for read in readers
        ]
        sums["mse"] += (reference - prediction).square().mean(dim=(1, 2, 3)).sum().item()
        for name, metric in metrics.items():
            values = metric(prediction, reference).reshape(-1)
            if len(values) != len(indices):
                raise ValueError(f"{name} did not return one value per frame")
            sums[name] += values.sum().item()
    if not all(math.isfinite(value) for value in sums.values()):
        raise ValueError("Nonfinite image metrics")
    return {name: total / (stop - start) for name, total in sums.items()}


def feature_arrays(read, fid_ids, clip_starts, inception, i3d, args):
    fid = []
    for first in range(0, len(fid_ids), args.batch_size):
        indices = fid_ids[first:first + args.batch_size]
        batch = tensor([read(int(index)) for index in indices], args.device)
        fid.append(inception(batch)[0].flatten(1).cpu().numpy())
    fvd = []
    for start in clip_starts:
        clip = tensor([read(index) for index in range(int(start), int(start) + 16)], args.device)
        fvd.append(i3d_features(clip, i3d))
    features = {"fid": np.concatenate(fid), "fvd16": np.concatenate(fvd)}
    for name, shape in (("fid", (len(fid_ids), 2048)), ("fvd16", (len(clip_starts), 400))):
        if features[name].shape != shape or not np.isfinite(features[name]).all():
            raise ValueError(f"Invalid {name} features, expected shape {shape}")
    return features


def evaluate(args, views):
    import pyiqa

    selected = {name: views[name] for name in args.sequences} if args.sequences else views
    readers = {name: sequence_readers(args, name, view) for name, view in selected.items()}
    metrics = {name: pyiqa.create_metric(name, device=args.device).eval() for name in ("ssim", "lpips")}
    if not args.image_metrics_only:
        from pytorch_fid.inception import InceptionV3

        inception = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(args.device).eval()
        i3d = torch.jit.load(str(args.i3d_path), map_location=args.device).eval()

    per_sequence = {}
    pooled = [[], []]
    with torch.inference_mode():
        for name, view in selected.items():
            print(f"Scoring {name}", flush=True)
            start, stop = view["frames"]
            pair = readers[name]
            scores = frame_metrics(pair, start, stop, metrics, args)
            per_sequence[name] = {**scores, "frames": stop - start, "roi": view["roi"]}
            if not args.image_metrics_only:
                fid_ids, clip_starts = sampling(start, stop)
                for arrays, read in zip(pooled, pair):
                    arrays.append(feature_arrays(read, fid_ids, clip_starts, inception, i3d, args))
    summary = {
        name: float(np.mean([sequence[name] for sequence in per_sequence.values()]))
        for name in ("mse", "ssim", "lpips")
    }
    if not args.image_metrics_only:
        for name in ("fid", "fvd16"):
            reference, prediction = [np.concatenate([sequence[name] for sequence in side]) for side in pooled]
            summary[name] = frechet(reference, prediction)
            summary[f"{name}_samples"] = len(reference)
    return {
        "protocol": "evdiff-common-roi-v1",
        "dataset": args.dataset,
        "prediction_index_offset": args.index_offset,
        "prediction_origin": args.pred_origin,
        "summary": summary,
        "per_sequence": per_sequence,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("BS_ERGB", "DSEC"))
    parser.add_argument("--gt-root", required=True, type=Path)
    parser.add_argument("--pred-root", required=True, type=Path, help="Directory containing sequence folders")
    parser.add_argument("--i3d-path", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--index-offset", type=int, default=0, help="PNG ID minus GT ID")
    parser.add_argument("--pred-origin", type=int, nargs=2, default=(0, 0), metavar=("TOP", "LEFT"))
    parser.add_argument("--sequences", nargs="+", help="Optional subset of the dataset split")
    parser.add_argument("--image-metrics-only", action="store_true", help="Skip FID and FVD")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not args.image_metrics_only and (args.i3d_path is None or not args.i3d_path.is_file()):
        parser.error("--i3d-path must point to the I3D TorchScript checkpoint")
    splits = json.loads(Path(__file__).with_name("splits.json").read_text(encoding="utf-8"))
    views = splits[args.dataset]
    if args.sequences and (len(set(args.sequences)) != len(args.sequences) or set(args.sequences) - views.keys()):
        parser.error("--sequences must list distinct members of the selected dataset split")
    result = evaluate(args, views)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
