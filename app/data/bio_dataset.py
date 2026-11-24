"""Paired single-channel dataset for the BF -> Nucleus fluorescence translation task.

Layout expected under <dataroot>:
    <dataroot>/<phase>/A/*.png    input  (bright field, 8-bit grayscale)
    <dataroot>/<phase>/B/*.png    target (nucleus / DAPI, 8-bit grayscale, same file name)

On top of the stocks `aligned` dataset this class
  * keeps the two modalities in separate folders (no A|B montage on disk),
  * works on 1-channel images and normalises to [-1, 1],
  * uses a stronger geometric augmentation for training (h/v flip + 90 deg rotations),
  * adds no resizing/cropping at load time (tiles are already the network size).

Usage:
    --dataset_mode bio --input_nc 1 --output_nc 1 --direction AtoB
"""
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils import data

from data.base_dataset import BaseDataset
from data.image_folder import make_dataset


class BioDataset(BaseDataset):
    """Load (A, B) grayscale tile pairs."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(preprocess="none", no_flip=True, load_size=512, crop_size=512)
        parser.add_argument("--target_gamma", type=float, default=1.0,
                            help="train in the non-linear domain B^gamma (gamma>1 emphasises the "
                                 "bright foreground, gamma<1 the dim one). Invert at inference.")
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.dir_A = os.path.join(opt.dataroot, opt.phase, "A")
        self.dir_B = os.path.join(opt.dataroot, opt.phase, "B")
        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))
        assert len(self.A_paths) == len(self.B_paths)
        assert len(self.A_paths) > 0, f"no images in {self.dir_A}"
        for a, b in zip(self.A_paths, self.B_paths):
            assert os.path.basename(a) == os.path.basename(b), (a, b)
        self.augment = is_train = opt.isTrain

    def __len__(self):
        return len(self.A_paths)

    def __getitem__(self, index):
        A = Image.open(self.A_paths[index]).convert("L")
        B = Image.open(self.B_paths[index]).convert("L")

        if self.augment:
            # geometric transforms shared by the pair (keeps A/B registered)
            k = np.random.randint(0, 4)
            if k:
                A, B = A.rotate(90 * k, expand=True), B.rotate(90 * k, expand=True)
            if np.random.rand() < 0.5:
                A, B = A.transpose(Image.FLIP_LEFT_RIGHT), B.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.rand() < 0.5:
                A, B = A.transpose(Image.FLIP_TOP_BOTTOM), B.transpose(Image.FLIP_TOP_BOTTOM)
            if np.random.rand() < 0.5:
                A, B = self._zoom_centre(A), self._zoom_centre(B)
            if np.random.rand() < 0.5:
                dx, dy = np.random.randint(-12, 13, 2)
                A, B = TF.affine(A, 0, [int(dx), int(dy)], 1, 0), TF.affine(B, 0, [int(dx), int(dy)], 1, 0)

        A = TF.to_tensor(A)
        B = TF.to_tensor(B)

        if self.augment:
            # photometric jitter on the input only: imitates focus / exposure drift
            if np.random.rand() < 0.7:
                m = A.mean()
                A = (A - m) * float(np.random.uniform(0.8, 1.25)) + m
            if np.random.rand() < 0.5:
                A = A * float(np.random.uniform(0.85, 1.15)) + float(np.random.uniform(-0.05, 0.05))
            if np.random.rand() < 0.3:
                A = A + torch.randn_like(A) * float(np.random.uniform(0.005, 0.03))
            A = A.clamp(0.0, 1.0)
            # light sensor noise on the target discourages over-smoothing
            if np.random.rand() < 0.3:
                B = (B + torch.randn_like(B) * float(np.random.uniform(0.005, 0.02))).clamp(0.0, 1.0)

        gamma = getattr(self.opt, "target_gamma", 1.0)
        if gamma != 1.0:
            # work in B^gamma: the residual is suppressed where the target is dim,
            # so the sparse bright structures dominate the L1 gradient
            B = B.clamp(0.0, 1.0) ** gamma

        A = A * 2.0 - 1.0
        B = B * 2.0 - 1.0
        return {"A": A, "B": B, "A_paths": self.A_paths[index], "B_paths": self.B_paths[index]}

    @staticmethod
    def _zoom_centre(img, lo=0.9, hi=1.1):
        """Resize by a random factor and centre-crop back to the original size."""
        w, h = img.size
        s = float(np.random.uniform(lo, hi))
        nw, nh = max(int(round(w * s)), w), max(int(round(h * s)), h)
        resized = img.resize((nw, nh), Image.BILINEAR)
        left, top = (nw - w) // 2, (nh - h) // 2
        return resized.crop((left, top, left + w, top + h))
