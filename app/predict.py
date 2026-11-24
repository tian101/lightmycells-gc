"""Inference + evaluation for a BF -> {Nucleus, Mitochondria} pix2pix model.

* runs the trained generator on whole 2048x2048 bright-field fields of view
  (overlapping 512 tiles with a Hann window, so there are no seams),
* writes BF / ground-truth / prediction comparison images,
* scores the held-out fields (PSNR, SSIM, Pearson r, foreground IoU).

Examples:
    python predict.py --target nucleus       --epoch 80
    python predict.py --target mitochondria  --epoch 80
    python predict.py --target mitochondria --name mitochondria_pix2pix --out ../results/mitochondria
"""
import argparse
import json
import os
import time

import numpy as np
import tifffile
import torch
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.metrics import structural_similarity as ssim

from models import networks

SRC = os.path.join("..", "Study_01")
TILE = 512
STRIDE = 384  # 128 px overlap
SUFFIX = {"nucleus": "Nucleus", "mitochondria": "Mitochondria"}


def rel_path(p):
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), p)


def norm8(arr, lo, hi):
    a = (arr.astype(np.float32) - lo) / max(hi - lo, 1e-6)
    return (np.clip(a, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def hann(n):
    w = np.hanning(n + 2)[1:-1]
    return np.outer(w, w).astype(np.float32)


def tiled_infer(net, img8, device, tile_batch=8, ret_float=False):
    """img8: uint8 HxW -> HxW prediction (same size). Tiles are batched.

    ret_float: return float32 on the 0..255 scale instead of uint8, so the caller can
    store the model output at 16-bit / float precision without a second pass.
    """
    h, w = img8.shape
    pad_h, pad_w = max(0, TILE - h), max(0, TILE - w)
    if pad_h or pad_w:
        img8 = np.pad(img8, ((0, pad_h), (0, pad_w)), mode="reflect")
    H, W = img8.shape
    acc = np.zeros((H, W), np.float32)
    wsum = np.zeros((H, W), np.float32)
    win = hann(TILE)
    ys = list(range(0, max(H - TILE, 0) + 1, STRIDE))
    xs = list(range(0, max(W - TILE, 0) + 1, STRIDE))
    if ys[-1] != H - TILE:
        ys.append(H - TILE)
    if xs[-1] != W - TILE:
        xs.append(W - TILE)
    coords = [(y, x) for y in ys for x in xs]
    for i in range(0, len(coords), tile_batch):
        chunk = coords[i:i + tile_batch]
        arr = np.stack([img8[y:y + TILE, x:x + TILE] for y, x in chunk]).astype(np.float32)
        arr = arr / 127.5 - 1.0
        t = torch.from_numpy(arr)[:, None].to(device)
        with torch.inference_mode():
            out = net(t).float().cpu().numpy()[:, 0]
        for (y, x), o in zip(chunk, out):
            pred = (np.clip(o, -1, 1) + 1.0) * 127.5
            acc[y:y + TILE, x:x + TILE] += pred * win
            wsum[y:y + TILE, x:x + TILE] += win
    out = np.clip(acc / np.maximum(wsum, 1e-6), 0, 255)
    if ret_float:
        return out[:h, :w].astype(np.float32)
    return out[:h, :w].astype(np.uint8)


def full_infer(net, img8, device, use_amp=False, ret_float=False):
    """One forward pass on the whole image (fast on >=12 GB cards)."""
    h, w = img8.shape
    assert h % 256 == 0 and w % 256 == 0, "full mode needs sizes divisible by 256"
    t = torch.from_numpy(img8.astype(np.float32) / 127.5 - 1.0)[None, None].to(device)
    with torch.inference_mode():
        if use_amp and device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                out = net(t)
        else:
            out = net(t)
    out = out[0, 0].float().cpu().numpy()
    out = np.clip((np.clip(out, -1, 1) + 1.0) * 127.5, 0, 255)
    if ret_float:
        return out.astype(np.float32)
    return out.astype(np.uint8)


def infer_image(net, img8, device, mode="auto", tile_batch=8, use_amp=False,
                ret_float=False):
    """mode=auto: whole image when possible (fast), falling back to tiles on OOM.

    ret_float: float32 0..255 output, for lossless 16-bit / float32 saving.
    """
    h, w = img8.shape
    if mode in ("auto", "full") and h % 256 == 0 and w % 256 == 0:
        try:
            return full_infer(net, img8, device, use_amp, ret_float)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if mode == "full":
                raise
            print(f"  full-image pass failed ({type(exc).__name__}), falling back to tiles")
            torch.cuda.empty_cache()
    return tiled_infer(net, img8, device, tile_batch, ret_float)


def to_depth(arr, bits):
    """Map an array on the 0..255 scale to the requested storage dtype."""
    a = np.asarray(arr, dtype=np.float32)
    if bits == 8:
        return np.clip(a, 0, 255).astype(np.uint8)
    if bits == 16:
        return np.round(np.clip(a, 0, 255) * (65535.0 / 255.0)).astype(np.uint16)
    return a


def write_tiff(path, arr):
    """Lossless TIFF; 2-D = single channel, HxWx3 = RGB composite."""
    tifffile.imwrite(path, arr, photometric="rgb" if arr.ndim == 3 else "minisblack")


def metrics(pred, gt, ssim_max=0):
    p = pred.astype(np.float32) / 255.0
    g = gt.astype(np.float32) / 255.0
    mse = float(np.mean((p - g) ** 2))
    psnr = 99.0 if mse == 0 else float(10 * np.log10(1.0 / mse))
    r = float(np.corrcoef(p.ravel(), g.ravel())[0, 1])
    try:
        th_p, th_g = threshold_otsu(pred), threshold_otsu(gt)
        mask_p, mask_g = pred > th_p, gt > th_g
        inter = np.logical_and(mask_p, mask_g).sum()
        union = np.logical_or(mask_p, mask_g).sum()
        iou = float(inter / max(union, 1))
    except Exception:
        iou = float("nan")
    ps, gs = p, g
    if ssim_max and max(p.shape) > ssim_max:  # SSIM is CPU bound; shrink for speed
        size = (ssim_max, int(ssim_max * p.shape[0] / p.shape[1]))
        ps = np.array(Image.fromarray(pred).resize(size[::-1], Image.BILINEAR), np.float32) / 255.0
        gs = np.array(Image.fromarray(gt).resize(size[::-1], Image.BILINEAR), np.float32) / 255.0
    return {"psnr": psnr, "ssim": float(ssim(ps, gs, data_range=1.0)), "pearson": r, "foreground_iou": iou}


def load_generator(ckpt, ngf, device, benchmark=False, norm="batch", antialias=False,
                   sfm_levels=0, sfm_base_size=512, sfm_start_level=1, netG="unet_256",
                   sfm_msau=0, sfm_fan=0, sfm_fan_out=0):
    # NOTE: cudnn.benchmark=True triggers a very long autotune on the first
    # 2048x2048 forward (~80 s on a GTX 1650), so it is opt-in only.
    torch.backends.cudnn.benchmark = bool(benchmark)
    state = torch.load(ckpt, map_location=device)
    if "module." in list(state.keys())[0]:
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    if norm == "auto":
        # instance norm has no running statistics, so the state dict reveals it
        norm = "batch" if any(k.endswith("running_mean") for k in state) else "instance"
    net = networks.define_G(1, 1, ngf, netG, norm, True, "normal", 0.02, antialias=antialias,
                            sfm_levels=sfm_levels, sfm_base_size=sfm_base_size,
                            sfm_start_level=sfm_start_level,
                            sfm_msau=int(sfm_msau or 0), sfm_fan=int(sfm_fan or 0),
                            sfm_fan_out=int(sfm_fan_out or 0)).to(device)
    net.load_state_dict(state)
    net.eval()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=sorted(SUFFIX), default="nucleus")
    ap.add_argument("--name", default=None, help="checkpoint folder (default: <target>_pix2pix)")
    ap.add_argument("--epoch", default="latest")
    ap.add_argument("--checkpoints_dir", default="../check")
    ap.add_argument("--ngf", type=int, default=64)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mode", choices=("auto", "full", "tile"), default="tile",
                    help="tile (default, matches training, best metrics); full: one forward on the whole image (fastest, needs size %% 256 == 0); auto: full with tiling fallback")
    ap.add_argument("--tile_batch", type=int, default=8, help="tiles per forward in tile mode")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast for the full-image pass")
    ap.add_argument("--ssim_max", type=int, default=0, help="downscale to this size before SSIM (0 = full size)")
    ap.add_argument("--skip_metrics", action="store_true")
    ap.add_argument("--out_bits", type=int, default=16, choices=(8, 16, 32),
                    help="bit depth of the saved TIFF: 8 = uint8, 16 = uint16 (default), "
                         "32 = float32 on 0..255")
    ap.add_argument("--benchmark", action="store_true", help="cudnn autotune (only for many images of the same size)")
    args = ap.parse_args()

    name = args.name or f"{args.target}_pix2pix"
    out_dir = rel_path(args.out or os.path.join("..", "results", args.target))
    stats_path = rel_path(os.path.join("..", "datasets", args.target, "norm_stats.json"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else "  <-- CPU fallback is very slow!"))
    ckpt = os.path.join(rel_path(args.checkpoints_dir), name, f"{args.epoch}_net_G.pth")
    t0 = time.time()
    net = load_generator(ckpt, args.ngf, device, args.benchmark)
    print(f"loaded {ckpt}  ({time.time() - t0:.2f}s)")

    with open(stats_path) as fh:
        stats = json.load(fh)["images"]

    os.makedirs(os.path.join(out_dir, "full"), exist_ok=True)
    rows, panels = [], []
    for key, meta in stats.items():
        if meta["phase"] != "test":
            continue
        i = int(key.split("_")[1])
        suffix = SUFFIX[args.target]
        bf = tifffile.imread(os.path.join(rel_path(SRC), f"image_{i}_BF_z0.ome.tiff"))
        gt = tifffile.imread(os.path.join(rel_path(SRC), f"image_{i}_{suffix}_z0.ome.tiff"))
        bf8 = norm8(bf, meta["input"]["p_lo"], meta["input"]["p_hi"])
        gt8 = norm8(gt, meta["target"]["p_lo"], meta["target"]["p_hi"])

        t0 = time.time()
        pred_f = infer_image(net, bf8, device, args.mode, args.tile_batch, args.amp,
                             ret_float=True)
        dt_infer = time.time() - t0
        pred8 = np.clip(pred_f, 0, 255).astype(np.uint8)
        t0 = time.time()
        m = metrics(pred8, gt8, args.ssim_max) if not args.skip_metrics else {"psnr": float("nan"), "ssim": float("nan"), "pearson": float("nan"), "foreground_iou": float("nan")}
        dt_metric = time.time() - t0
        m["image"] = f"image_{i:02d}"
        rows.append(m)

        t0 = time.time()
        write_tiff(os.path.join(out_dir, "full", f"image_{i:02d}_pred.tif"), to_depth(pred_f, args.out_bits))
        write_tiff(os.path.join(out_dir, "full", f"image_{i:02d}_gt.tif"), to_depth(gt8, args.out_bits))
        write_tiff(os.path.join(out_dir, "full", f"image_{i:02d}_input.tif"), to_depth(bf8, args.out_bits))
        rgb = np.dstack([bf8, gt8, pred8])  # R=BF, G=GT, B=prediction
        write_tiff(os.path.join(out_dir, "full", f"image_{i:02d}_rgb.tif"), rgb)
        thick = np.full((bf8.shape[0], 6), 255, np.uint8)
        compare = np.array(Image.fromarray(
            np.hstack([bf8, thick, gt8, thick, pred8])).resize((1536, 512), Image.BILINEAR))
        write_tiff(os.path.join(out_dir, "full", f"image_{i:02d}_compare.tif"), compare)
        dt_save = time.time() - t0
        panels.append((i, bf8, gt8, pred8))
        print(f"image_{i:02d} psnr={m['psnr']:.2f} ssim={m['ssim']:.3f} r={m['pearson']:.3f} iou={m['foreground_iou']:.3f}"
              f"  [infer {dt_infer:.2f}s, metrics {dt_metric:.2f}s, save {dt_save:.2f}s]")

    # summary montage: every test field, BF | GT | prediction (downscaled)
    cell = 256
    blocks = []
    for i, bf8, gt8, pred8 in panels:
        ims = []
        for a in (bf8, gt8, pred8):
            ims.append(np.array(Image.fromarray(a).resize((cell, cell), Image.BILINEAR)))
            ims.append(np.full((cell, 4), 255, np.uint8))
        blocks.append(np.hstack(ims[:-1]))
        blocks.append(np.full((4, blocks[-1].shape[1]), 255, np.uint8))
    write_tiff(os.path.join(out_dir, "summary.tif"), np.vstack(blocks[:-1]))
    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("psnr", "ssim", "pearson", "foreground_iou")}
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump({"target": args.target, "per_image": rows, "mean": mean, "epoch": args.epoch}, fh, indent=2)
    with open(os.path.join(out_dir, "metrics.csv"), "w") as fh:
        fh.write("image,psnr,ssim,pearson,foreground_iou\n")
        for r in rows:
            fh.write(f"{r['image']},{r['psnr']:.3f},{r['ssim']:.4f},{r['pearson']:.4f},{r['foreground_iou']:.4f}\n")
        fh.write(f"MEAN,{mean['psnr']:.3f},{mean['ssim']:.4f},{mean['pearson']:.4f},{mean['foreground_iou']:.4f}\n")
    print("MEAN", {k: round(v, 4) for k, v in mean.items()})


if __name__ == "__main__":
    main()
