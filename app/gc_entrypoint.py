#!/usr/bin/env python3
"""Grand Challenge inference container: transmitted light -> four fluorescence channels.

Interface
---------
in   /input/images/organelles-transmitted-light-ome-tiff/<uuid>.tif
out  /output/images/nucleus-fluorescence-ome-tiff/<uuid>.tif
     /output/images/mitochondria-fluorescence-ome-tiff/<uuid>.tif
     /output/images/actin-fluorescence-ome-tiff/<uuid>.tif
     /output/images/tubulin-fluorescence-ome-tiff/<uuid>.tif

The platform invokes this as ``gc_entrypoint.py <interface.json> <output_dir>``.
The JSON is read to learn the real socket slugs (defaults are used if absent)
and the output directory is only a fallback for /output. Set INPUT_DIR /
OUTPUT_DIR to test locally without Docker.

The four models are independent single-channel pix2pix U-Nets (ngf 64, 8 down
levels) trained on the LightMyCells index v2.2 split. Inference reproduces the
validated evaluation path exactly: 1/99.5 percentile normalisation of the whole
plane, then overlapping 512 px tiles (stride 384) blended with a Hann window.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import tifffile
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from models import networks  # noqa: E402

IN_SLUG = "organelles-transmitted-light-ome-tiff"
CHANNELS = ("nucleus", "mitochondria", "actin", "tubulin")
OUT_SLUG = {c: f"{c}-fluorescence-ome-tiff" for c in CHANNELS}


def load_interface(json_path):
    """Read the platform's interface description to learn the real socket slugs.

    Falls back to the hard-coded defaults when no readable JSON is supplied, so
    local testing and slug changes on the platform both keep working.
    """
    in_slug, out_slugs = IN_SLUG, dict(OUT_SLUG)
    if not json_path or not os.path.isfile(json_path):
        return in_slug, out_slugs
    try:
        with open(json_path) as fh:
            spec = json.load(fh)
    except (OSError, ValueError) as exc:
        log(f"warning: cannot parse {json_path} ({exc}); using default slugs")
        return in_slug, out_slugs

    def slugs(items):
        found = []
        for item in items or []:
            slug = (item.get("socket") or {}).get("slug") or item.get("slug")
            if slug:
                found.append(slug)
        return found

    ins = slugs(spec.get("inputs"))
    outs = slugs(spec.get("outputs"))
    if len(ins) == 1:
        in_slug = ins[0]
    for ch in CHANNELS:
        for slug in outs:
            if ch in slug:
                out_slugs[ch] = slug
                break
    return in_slug, out_slugs

# Architecture is identical for all four checkpoints (read from train_opt.txt).
ARCH = dict(ngf=64, norm="batch", netG="unet_256", antialias=False,
            sfm_levels=5, sfm_start_level=2, sfm_msau=1, sfm_fan=1,
            sfm_fan_out=0, sfm_base_size=512)

TILE = 512
STRIDE = 384  # 128 px overlap
PERC_LO, PERC_HI = 1.0, 99.5


def log(msg):
    print(msg, flush=True)


# --------------------------------------------------------------------------- io
def _decode(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        return np.asarray(tifffile.imread(path))
    try:
        import SimpleITK as sitk
    except ImportError:
        raise RuntimeError(f"cannot read {path}: SimpleITK is not installed")
    return sitk.GetArrayFromImage(sitk.ReadImage(path))


def to_gray(plane):
    """Drop a colour axis; the models were trained on grayscale transmitted light."""
    if plane.ndim == 2:
        return plane
    if plane.ndim == 3 and plane.shape[-1] in (3, 4) and min(plane.shape[:2]) > 8:
        w = np.array([0.299, 0.587, 0.114, 0.0][:plane.shape[-1]], np.float32)
        return (plane.astype(np.float32) * w).sum(-1)
    if plane.ndim == 3 and plane.shape[0] in (3, 4) and min(plane.shape[1:]) > 8:
        w = np.array([0.299, 0.587, 0.114, 0.0][:plane.shape[0]], np.float32)
        return (plane.astype(np.float32) * w[:, None, None]).sum(0)
    raise RuntimeError(f"cannot interpret shape {plane.shape} as a 2-D image")


def read_planes(path):
    """Decode an image into one 2-D grayscale plane per z slice.

    Handles a single plane, a z-stack, a colour image, and the (Z, H, W, C) mix,
    because the platform forwards whatever microscopy vendor format was uploaded.
    """
    arr = _decode(path)
    while arr.ndim > 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 2:
        return [arr]
    if arr.ndim == 3:
        colour_last = arr.shape[-1] in (3, 4) and min(arr.shape[:2]) > 8
        colour_first = arr.shape[0] in (3, 4) and min(arr.shape[1:]) > 8
        if colour_last or colour_first:
            log(f"  treating shape {arr.shape} as one colour plane")
            return [to_gray(arr)]
        return [to_gray(p) for p in arr]
    if arr.ndim == 4:
        return [to_gray(p) for p in arr]
    raise RuntimeError(f"unsupported shape {arr.shape} in {path}")


def sharpness(plane):
    """Variance of the 4-neighbour Laplacian: a focus proxy for stack inputs."""
    a = plane.astype(np.float32)
    lap = (a[:-2, 1:-1] + a[2:, 1:-1] + a[1:-1, :-2] + a[1:-1, 2:]) - 4.0 * a[1:-1, 1:-1]
    return float(lap.var())


def normalise(arr, mode="percentile"):
    """Per-plane intensity normalisation to 8-bit.

    Verbatim copy of ``normalise`` in data/lmc_dataset.py; that module imports
    torchvision, so it is inlined to keep the image small. test/verify.py asserts
    bit-identical output for uint8 / uint16 / float32 input.
    """
    if mode == "meanstd":
        a = arr.astype(np.float32)
        scale = max(3.0 * float(a.std()), 1e-6)
        a = (a - float(a.mean())) / scale + 0.5
        return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if arr.dtype.kind in "iu" and np.iinfo(arr.dtype).max < 2 ** 16:
        sample = arr[::2, ::2]
        lo, hi = np.percentile(sample, (PERC_LO, PERC_HI))
        top = np.iinfo(arr.dtype).max + 1
        src = np.clip(np.arange(top, dtype=np.float32), lo, hi)
        lut = ((src - lo) * (255.0 / max(float(hi - lo), 1e-6)) + 0.5).astype(np.uint8)
        return np.take(lut, arr)
    if arr.dtype.kind in "iu":
        a = arr.astype(np.float32)
        sample = a[::2, ::2]
        lo, hi = np.percentile(sample, (PERC_LO, PERC_HI))
        a = (a - lo) * (1.0 / max(float(hi - lo), 1e-6))
        return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def write_ome_tiff(path, arr):
    """Write a single-channel uint16 OME-TIFF named after the input uuid."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tifffile.imwrite(path, arr, ome=True, photometric="minisblack",
                     metadata={"axes": "YX"})


# ---------------------------------------------------------------------- network
def load_generator(ckpt, device):
    """Rebuild one generator and load its weights (mirrors predict.load_generator)."""
    state = torch.load(ckpt, map_location=device)
    if next(iter(state)).startswith("module."):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    net = networks.define_G(1, 1, ARCH["ngf"], ARCH["netG"], ARCH["norm"], True,
                            "normal", 0.02,
                            antialias=ARCH["antialias"],
                            sfm_levels=ARCH["sfm_levels"],
                            sfm_base_size=ARCH["sfm_base_size"],
                            sfm_start_level=ARCH["sfm_start_level"],
                            sfm_msau=ARCH["sfm_msau"],
                            sfm_fan=ARCH["sfm_fan"],
                            sfm_fan_out=ARCH["sfm_fan_out"]).to(device)
    net.load_state_dict(state)
    net.eval()
    return net


def hann(n):
    w = np.hanning(n + 2)[1:-1]
    return np.outer(w, w).astype(np.float32)


def tiled_infer(net, img8, device, tile_batch=8):
    """uint8 HxW -> float32 HxW on the 0..255 scale (Hann-blended 512 px tiles)."""
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
    return np.clip(acc / np.maximum(wsum, 1e-6), 0, 255)[:h, :w].astype(np.float32)


def predict_channel(net, planes8, device, z_select, tile_batch):
    """Run one channel and return a single float32 prediction on the 0..255 scale.

    z_select=sharp/mid predicts on the one plane already chosen by the caller.
    z_select=all predicts on every plane and keeps the sharpest prediction, which
    is the best available proxy for that channel's own best focus.
    """
    if z_select == "all" and len(planes8) > 1:
        best, best_s = None, -1.0
        for p in planes8:
            pred = tiled_infer(net, p, device, tile_batch)
            s = sharpness(pred)
            if s > best_s:
                best, best_s = pred, s
        return best
    return tiled_infer(net, planes8[0], device, tile_batch)


# ------------------------------------------------------------------------- main
def find_inputs(in_dir, in_slug=IN_SLUG):
    d = os.path.join(in_dir, "images", in_slug)
    if not os.path.isdir(d):
        d = in_dir
    files = []
    for e in ("*.tif", "*.tiff", "*.mha", "*.mhd"):
        files.extend(glob.glob(os.path.join(d, e)))
    return sorted(files)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("positional", nargs="*", help="ignored: the platform passes interface.json")
    ap.add_argument("--weights", default=os.path.join(HERE, "..", "weights"))
    ap.add_argument("--input_dir", default=os.environ.get("INPUT_DIR", "/input"))
    ap.add_argument("--output_dir", default=os.environ.get("OUTPUT_DIR", "/output"))
    ap.add_argument("--z_select", default="sharp", choices=("sharp", "mid", "all"),
                    help="reduce a z-stack input to inference plane(s)")
    ap.add_argument("--tile_batch", type=int, default=8)
    ap.add_argument("--save_input", action="store_true",
                    help="also dump the normalised input plane next to the outputs")
    args = ap.parse_args()

    in_slug, out_slug = load_interface(args.positional[0] if args.positional else None)
    log(f"interface: in={in_slug} out={out_slug}")

    # the platform mounts /output; the second positional argument holds the same
    # path, so fall back to it only if /output is not there
    if not os.path.isdir(args.output_dir) and len(args.positional) > 1:
        if os.path.isdir(args.positional[1]):
            args.output_dir = args.positional[1]
    log(f"input_dir={args.input_dir} output_dir={args.output_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device: {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    nets, t_load = {}, time.time()
    for ch in CHANNELS:
        ckpt = os.path.join(args.weights, f"{ch}.pth")
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"weight file missing: {ckpt}")
        nets[ch] = load_generator(ckpt, device)
        log(f"  loaded {ch:<13} {os.path.getsize(ckpt) / 2**20:.1f} MB")
    log(f"models ready in {time.time() - t_load:.1f} s")

    files = find_inputs(args.input_dir, in_slug)
    if not files:
        raise RuntimeError(f"no input images under {os.path.join(args.input_dir, 'images', in_slug)}")
    log(f"{len(files)} input image(s)")

    for path in files:
        uuid = os.path.splitext(os.path.basename(path))[0]
        planes = read_planes(path)
        raw_shape, raw_dtype = np.shape(planes[0]), planes[0].dtype
        if args.z_select == "mid":
            planes = [planes[len(planes) // 2]]
        elif args.z_select == "sharp":
            planes = [planes[int(np.argmax([sharpness(p) for p in planes]))]]
        planes8 = [normalise(p) for p in planes]
        log(f"{uuid}: {len(planes8)} inference plane(s), {raw_shape} {raw_dtype}")

        for ch in CHANNELS:
            t0 = time.time()
            pred = predict_channel(nets[ch], planes8, device, args.z_select, args.tile_batch)
            out = np.round(np.clip(pred, 0, 255) * (65535.0 / 255.0)).astype(np.uint16)
            dst = os.path.join(args.output_dir, "images", out_slug[ch], f"{uuid}.tif")
            write_ome_tiff(dst, out)
            log(f"  {ch:<13} {out.shape} uint16 -> {dst}  [{time.time() - t0:.2f} s]")

        if args.save_input:
            dst = os.path.join(args.output_dir, "debug", f"{uuid}_input.tif")
            write_ome_tiff(dst, planes8[0])
            log(f"  input dump -> {dst}")

    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
