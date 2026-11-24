#!/usr/bin/env python3
"""Download the four checkpoints from a GitHub release into the image.

The weights are 4 x 208 MB, which is over GitHub's 100 MB per-file limit, so they
live as release assets instead of git objects. They are fetched at build time and
then validated, so a broken tag/asset fails the build instead of the evaluation.

    python fetch_weights.py --dest /opt/weights --tag v1
"""
import argparse
import os
import sys
import urllib.request

# The repository this file ships in; Dockerfile passes --tag only.
REPO = "tian101/lightmycells-gc"

CHANNELS = ("nucleus", "mitochondria", "actin", "tubulin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="/opt/weights")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--repo", default=REPO)
    ap.add_argument("--base-url", default="",
                    help="override the release URL prefix (used by the tests)")
    args = ap.parse_args()

    base = args.base_url or (
        f"https://github.com/{args.repo}/releases/download/{args.tag}")

    os.makedirs(args.dest, exist_ok=True)
    for ch in CHANNELS:
        url = f"{base}/{ch}.pth"
        dst = os.path.join(args.dest, f"{ch}.pth")
        print(f"fetching {url}", flush=True)
        try:
            urllib.request.urlretrieve(url, dst)
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"cannot download {url}: {exc}")
        size = os.path.getsize(dst)
        if size < 100 * 2 ** 20:
            sys.exit(f"{dst} is only {size / 2**20:.1f} MB - wrong asset or an LFS pointer")
        print(f"  {ch}.pth {size / 2**20:.1f} MB", flush=True)

    import torch
    for ch in CHANNELS:
        state = torch.load(os.path.join(args.dest, f"{ch}.pth"), map_location="cpu")
        print(f"  {ch}.pth loads ({len(state)} tensors)")
    print("weights OK")


if __name__ == "__main__":
    main()
