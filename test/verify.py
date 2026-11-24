#!/usr/bin/env python3
"""Verify the container against the training/eval code it was exported from.

Three things are checked:

1. `normalise()` in gc_entrypoint.py is bit-identical to the one in
   data/lmc_dataset.py for uint8 / uint16 / float32 planes.
2. `load_generator()` produces the same weights, and `tiled_infer()` produces
   the same prediction, as predict.load_generator / predict.infer_image.
3. End to end: a fake /input tree holding real test fields makes the entrypoint
   emit four uint16 OME-TIFFs with the input uuids.

Run it on the training server, where both the repo and the checkpoints live:

    python test/verify.py --repo /mnt/data0/lixt/pix/pytorch-CycleGAN-and-pix2pix-master \
                          --app  /tmp/gc/app --weights /tmp/gc/weights \
                          --index /mnt/data0/lixt/pix/datasets/lmc/index.json
"""
import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import tifffile

CHANNELS = ("nucleus", "mitochondria", "actin", "tubulin")


def planes_from_index(index_path, n_fields=2):
    import json
    with open(index_path) as fh:
        index = json.load(fh)
    src = index["src"]
    out = []
    for f in index["fields"]:
        targets = {k.lower(): v for k, v in f["targets"].items()}
        if f["split"] != "test" or "nucleus" not in targets:
            continue
        z = int(f["input_best_z"])
        if str(z) not in f["input_z"]:
            z = sorted(int(k) for k in f["input_z"])[len(f["input_z"]) // 2]
        out.append({
            "uuid": f"{f['study']}_img{f['image']:05d}",
            "input": os.path.join(src, f["input_z"][str(z)]),
            "gt": {c: os.path.join(src, t["path"]) for c, t in targets.items()},
        })
        if len(out) >= n_fields:
            break
    if not out:
        raise SystemExit("no test fields with a nucleus target in the index")
    return out


# ------------------------------------------------------------------ dump modes
def dump(kind, repo, app, weights, index_path, out_path):
    import torch
    fields = planes_from_index(index_path)
    plane = np.asarray(tifffile.imread(fields[0]["input"]))
    rng = np.random.default_rng(0)
    synth = {
        "u8": rng.integers(0, 256, (300, 400), dtype=np.uint8),
        "u16": rng.integers(0, 65536, (300, 400), dtype=np.uint16),
        "f32": rng.random((300, 400)).astype(np.float32),
    }
    ckpt = os.path.join(weights, "nucleus.pth")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if kind == "repo":
        sys.path.insert(0, repo)
        os.chdir(repo)
        from data.lmc_dataset import normalise as norm
        from predict import load_generator, infer_image
    else:
        sys.path.insert(0, app)
        import gc_entrypoint as gc
        norm = gc.normalise
        load_generator = lambda c, d: gc.load_generator(c, d)  # noqa: E731
        infer_image = None

    ref = {k: norm(v) for k, v in synth.items()}
    ref["real"] = norm(plane)

    if kind == "repo":
        net = load_generator(ckpt, 64, device, norm="batch", sfm_levels=5,
                             sfm_base_size=512, sfm_start_level=2,
                             sfm_msau=1, sfm_fan=1, sfm_fan_out=0)
        pred = infer_image(net, ref["real"], device, "tile", 8, False, True)
    else:
        net = load_generator(ckpt, device)
        pred = gc.tiled_infer(net, ref["real"], device, 8)

    ref["pred"] = pred.astype(np.float32)
    import hashlib
    h = hashlib.sha256()
    for k in sorted(net.state_dict()):
        t = net.state_dict()[k].detach().cpu().contiguous().numpy()
        h.update(k.encode())
        h.update(str(t.dtype).encode())
        h.update(str(t.shape).encode())
        h.update(t.tobytes())
    ref["weights_digest"] = np.array(h.hexdigest())
    ref["weights_count"] = np.array(len(net.state_dict()))
    np.savez_compressed(out_path, **ref)
    print(f"wrote {out_path} ({int(ref['weights_count'])} tensors, pred {pred.shape}, "
          f"digest {h.hexdigest()[:16]}...)")


# --------------------------------------------------------------------- compare
def compare(ref_path, app_path):
    ref, app = np.load(ref_path), np.load(app_path)
    ok = True
    for key in ("u8", "u16", "f32", "real"):
        same = np.array_equal(ref[key], app[key])
        ok &= same
        print(f"normalise[{key:<4}] identical: {same}")
    rw, aw = str(ref["weights_digest"]), str(app["weights_digest"])
    n_ok = int(ref["weights_count"]) == int(app["weights_count"])
    ok &= n_ok and rw == aw
    print(f"weights: {int(ref['weights_count'])} tensors, same count={n_ok}, "
          f"identical={rw == aw}")
    dp = float(np.abs(ref["pred"].astype(np.float64) - app["pred"].astype(np.float64)).max())
    ok &= dp == 0.0
    print(f"prediction: max|diff|={dp:.3e}  (shape {ref['pred'].shape})")
    return ok


# -------------------------------------------------------------------------- e2e
def e2e(app, weights, index_path, workdir, python):
    fields = planes_from_index(index_path, n_fields=2)
    in_dir = os.path.join(workdir, "input")
    out_dir = os.path.join(workdir, "output")
    slug = os.path.join(in_dir, "images", "organelles-transmitted-light-ome-tiff")
    os.makedirs(slug, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    for f in fields:
        dst = os.path.join(slug, f"{f['uuid']}.tif")
        arr = np.asarray(tifffile.imread(f["input"]))
        tifffile.imwrite(dst, arr, ome=True, photometric="minisblack", metadata={"axes": "YX"})

    cmd = [python, os.path.join(app, "gc_entrypoint.py"),
           os.path.join(workdir, "interface.json"), out_dir,
           "--weights", weights, "--input_dir", in_dir, "--output_dir", out_dir]
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, check=True)

    ok = True
    for f in fields:
        for ch in CHANNELS:
            p = os.path.join(out_dir, "images", f"{ch}-fluorescence-ome-tiff",
                             f"{f['uuid']}.tif")
            if not os.path.isfile(p):
                print(f"MISSING {p}")
                ok = False
                continue
            arr = tifffile.imread(p)
            good = (arr.dtype == np.uint16 and arr.ndim == 2
                    and arr.shape == np.asarray(tifffile.imread(f["input"])).shape[:2])
            ok &= good
            print(f"  {ch:<13} {os.path.basename(p)}  {arr.shape} {arr.dtype} "
                  f"range {arr.min()}..{arr.max()}  {'OK' if good else 'BAD'}")
    extra = set(os.listdir(os.path.join(out_dir, "images")))
    expect = {f"{c}-fluorescence-ome-tiff" for c in CHANNELS}
    if extra != expect:
        print(f"unexpected output folders: {sorted(extra - expect)}")
        ok = False
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/mnt/data0/lixt/pix/pytorch-CycleGAN-and-pix2pix-master")
    ap.add_argument("--app", default="/tmp/gc/app")
    ap.add_argument("--weights", default="/tmp/gc/weights")
    ap.add_argument("--index", default="/mnt/data0/lixt/pix/datasets/lmc/index.json")
    ap.add_argument("--mode", default="all", choices=("all", "repo", "app"))
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    if args.mode == "repo":
        dump("repo", args.repo, args.app, args.weights, args.index,
             os.path.join(here, "_ref.npz"))
        return 0
    if args.mode == "app":
        dump("app", args.repo, args.app, args.weights, args.index,
             os.path.join(here, "_app.npz"))
        return 0

    py = sys.executable
    tmp = tempfile.mkdtemp(prefix="gcverify_")
    try:
        for mode in ("repo", "app"):
            subprocess.run([py, os.path.abspath(__file__), "--mode", mode,
                            "--repo", args.repo, "--app", args.app,
                            "--weights", args.weights, "--index", args.index],
                           check=True)
        print("\n=== unit comparison ===")
        ok = compare(os.path.join(here, "_ref.npz"), os.path.join(here, "_app.npz"))
        print("\n=== end to end ===")
        ok &= e2e(args.app, args.weights, args.index, tmp, py)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        for f in ("_ref.npz", "_app.npz"):
            p = os.path.join(here, f)
            if os.path.isfile(p):
                os.remove(p)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
