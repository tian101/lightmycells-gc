#!/usr/bin/env python3
"""Robustness + quality check for the container entrypoint.

Feeds the entrypoint the awkward input shapes the platform might forward
(z-stack, colour plane, .mha, odd size) and scores the plain case against the
ground truth so a silently broken model cannot pass as "it produced a file".

    python test/test_inputs.py --app /tmp/gc/app --weights /tmp/gc/weights \
                               --index /mnt/data0/lixt/pix/datasets/lmc/index.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import tifffile

CHANNELS = ("nucleus", "mitochondria", "actin", "tubulin")


def load_field(index_path, n=1):
    with open(index_path) as fh:
        index = json.load(fh)
    src = index["src"]
    out = []
    for f in index["fields"]:
        t = {k.lower(): v for k, v in f["targets"].items()}
        if f["split"] != "test" or "nucleus" not in t:
            continue
        z = int(f["input_best_z"])
        if str(z) not in f["input_z"]:
            z = sorted(int(k) for k in f["input_z"])[len(f["input_z"]) // 2]
        out.append({"uuid": f"{f['study']}_img{f['image']:05d}",
                    "src": os.path.join(src, f["input_z"][str(z)]),
                    "gt": {c: os.path.join(src, v["path"]) for c, v in t.items()}})
        if len(out) >= n:
            break
    return out


def normalise8(arr):
    """Same 1/99.5 percentile stretch the training pipeline applies to the target."""
    if arr.dtype.kind in "iu" and np.iinfo(arr.dtype).max < 2 ** 16:
        lo, hi = np.percentile(arr[::2, ::2], (1.0, 99.5))
        top = np.iinfo(arr.dtype).max + 1
        src = np.clip(np.arange(top, dtype=np.float32), lo, hi)
        lut = ((src - lo) * (255.0 / max(float(hi - lo), 1e-6)) + 0.5).astype(np.uint8)
        return np.take(lut, arr)
    a = arr.astype(np.float32)
    lo, hi = np.percentile(a[::2, ::2], (1.0, 99.5))
    return (np.clip((a - lo) / max(float(hi - lo), 1e-6), 0, 1) * 255.0 + 0.5).astype(np.uint8)


def score(pred_u16, gt_raw):
    """PSNR / Pearson / SSIM on the 0..255 scale, matching tools/make_report.py."""
    from skimage.metrics import structural_similarity as ssim
    p = np.clip(np.asarray(pred_u16, np.float32) / 65535.0 * 255.0, 0, 255)
    g = normalise8(gt_raw).astype(np.float32)
    mse = float(np.mean((p - g) ** 2))
    psnr = 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)
    r = float(np.corrcoef(p.ravel(), g.ravel())[0, 1])
    s = float(ssim(p / 255.0, g / 255.0, data_range=1.0))
    return psnr, r, s


def run_case(app, weights, work, name, write):
    """write(dir) -> list of expected output basenames; returns psnr dict."""
    in_dir = os.path.join(work, name, "input")
    out_dir = os.path.join(work, name, "output")
    slug = os.path.join(in_dir, "images", "organelles-transmitted-light-ome-tiff")
    os.makedirs(slug, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    uuids = write(slug)

    cmd = [sys.executable, os.path.join(app, "gc_entrypoint.py"), "x", out_dir,
           "--weights", weights, "--input_dir", in_dir, "--output_dir", out_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout[-2000:], proc.stderr[-2000:])
        raise SystemExit(f"case {name}: entrypoint failed")

    bad = []
    for uuid in uuids:
        for ch in CHANNELS:
            p = os.path.join(out_dir, "images", f"{ch}-fluorescence-ome-tiff", f"{uuid}.tif")
            if not os.path.isfile(p):
                bad.append(f"missing {os.path.relpath(p, out_dir)}")
                continue
            arr = tifffile.imread(p)
            if arr.dtype != np.uint16 or arr.ndim != 2:
                bad.append(f"{os.path.basename(p)}: {arr.dtype} {arr.shape}")
    print(f"  {name:<22} {'OK' if not bad else 'FAIL: ' + '; '.join(bad)}")
    return uuids, out_dir, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app", default="/tmp/gc/app")
    ap.add_argument("--weights", default="/tmp/gc/weights")
    ap.add_argument("--index", default="/mnt/data0/lixt/pix/datasets/lmc/index.json")
    ap.add_argument("--only", default="",
                    help="comma separated case names to run (default: all)")
    args = ap.parse_args()

    fields = load_field(args.index)
    field = fields[0]
    plane = np.asarray(tifffile.imread(field["src"]))
    h, w = plane.shape
    print(f"source plane {field['uuid']} {plane.shape} {plane.dtype}")

    work = tempfile.mkdtemp(prefix="gcvariants_")
    problems = []
    try:
        def plain(slug):
            p = os.path.join(slug, "case-plain.tif")
            tifffile.imwrite(p, plane, ome=True, photometric="minisblack",
                             metadata={"axes": "YX"})
            return ["case-plain"]

        def zstack(slug):
            stack = np.stack([np.roll(plane, k * 3, axis=1) for k in range(4)])
            p = os.path.join(slug, "case-stack.tif")
            tifffile.imwrite(p, stack, ome=True, photometric="minisblack",
                             metadata={"axes": "ZYX"})
            return ["case-stack"]

        def colour(slug):
            rgb = np.stack([plane, np.roll(plane, 7, axis=0), plane], -1).astype(np.uint8)
            p = os.path.join(slug, "case-rgb.tif")
            tifffile.imwrite(p, rgb, ome=True, photometric="rgb", metadata={"axes": "YXS"})
            return ["case-rgb"]

        def odd(slug):
            small = plane[:301, :407]
            p = os.path.join(slug, "case-odd.tif")
            tifffile.imwrite(p, small, ome=True, photometric="minisblack",
                             metadata={"axes": "YX"})
            return ["case-odd"]

        def two(slug):
            for i, u in enumerate(("case-a", "case-b")):
                tifffile.imwrite(os.path.join(slug, f"{u}.tif"),
                                 np.roll(plane, i * 5, 0), ome=True,
                                 photometric="minisblack", metadata={"axes": "YX"})
            return ["case-a", "case-b"]

        def mha(slug):
            import SimpleITK as sitk
            p = os.path.join(slug, "case-mha.mha")
            sitk.WriteImage(sitk.GetImageFromArray(plane), p)
            return ["case-mha"]

        cases = [("plain 2D", plain), ("z-stack (4 planes)", zstack),
                 ("colour RGB", colour), ("odd 301x407", odd),
                 ("two inputs", two)]
        try:
            import SimpleITK  # noqa: F401
            cases.append((".mha", mha))
        except ImportError:
            pass

        if args.only:
            want = {s.strip() for s in args.only.split(",")}
            cases = [c for c in cases if c[0].split()[0] in want or c[0] in want]

        plain_out = None
        for name, fn in cases:
            uuids, out_dir, bad = run_case(args.app, args.weights, work, name, fn)
            problems += bad
            if name.startswith("plain"):
                plain_out = out_dir
        if not any(n == ".mha" for n, _ in cases):
            print(f"  {'.mha':<22} SKIP (SimpleITK not installed here; it is in the image)")

        print("\nquality vs ground truth (plain case; same metric as tools/make_report.py):")
        for ch in CHANNELS:
            gt_path = field["gt"].get(ch)
            if not gt_path or not os.path.isfile(gt_path):
                print(f"  {ch:<13} no ground truth in the index")
                continue
            pred = tifffile.imread(os.path.join(
                plain_out, "images", f"{ch}-fluorescence-ome-tiff", "case-plain.tif"))
            p, r, s = score(pred, tifffile.imread(gt_path))
            print(f"  {ch:<13} PSNR {p:6.2f} dB   Pearson {r:.3f}   SSIM {s:.3f}")
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print("\nRESULT:", "PASS" if not problems else f"FAIL ({len(problems)} problems)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
