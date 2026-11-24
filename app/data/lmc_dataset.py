"""LightMyCells dataset with random focal-plane sampling.

Unlike `bio_dataset` (pre-cut PNG tiles), this class reads the original
OME-TIFFs through `index.json` (see build_lmc_index.py) and, for every training
sample, picks
  * a **random z-slice** of the input modality (BF / DIC / PC), and
  * a **random 512 crop**,
while the target is always the *best-focus* slice of the requested channel of
the same field of view. The input therefore is frequently out of focus, which is
what the test data looks like (the challenge hands out arbitrary focus planes).
See BiaPy / LightMyCells, 'Training with random out-of-focus planes'.

Layout expected under <dataroot>:
    <dataroot>/index.json          built by build_lmc_index.py

Usage:
    --dataset_mode lmc --target nucleus --input_z_mode random
"""
import json
import os
from collections import OrderedDict

import numpy as np
import torch
import tifffile

from data.base_dataset import BaseDataset

CHANNELS = ("nucleus", "mitochondria", "actin", "tubulin")
PERC_LO, PERC_HI = 1.0, 99.5
PERC_STRIDE = 2  # percentiles from a strided subsample: ~4x cheaper, same values


def normalise(arr, mode="percentile"):
    """Per-image intensity normalisation to 8-bit, matching the old PNG pipeline.

    The percentile stretch is applied through a lookup table, so it costs a
    single pass over the plane instead of several float32 temporaries.
    """
    if mode == "meanstd":
        a = arr.astype(np.float32)
        scale = max(3.0 * float(a.std()), 1e-6)
        a = (a - float(a.mean())) / scale + 0.5
        return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    if arr.dtype.kind in "iu" and np.iinfo(arr.dtype).max < 2 ** 16:
        sample = arr[::PERC_STRIDE, ::PERC_STRIDE]
        lo, hi = np.percentile(sample, (PERC_LO, PERC_HI))
        top = np.iinfo(arr.dtype).max + 1
        src = np.clip(np.arange(top, dtype=np.float32), lo, hi)
        lut = ((src - lo) * (255.0 / max(float(hi - lo), 1e-6)) + 0.5).astype(np.uint8)
        return np.take(lut, arr)
    if arr.dtype.kind in "iu":
        a = arr.astype(np.float32)
        sample = a[::PERC_STRIDE, ::PERC_STRIDE]
        lo, hi = np.percentile(sample, (PERC_LO, PERC_HI))
        a = (a - lo) * (1.0 / max(float(hi - lo), 1e-6))
        return (np.clip(a, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def _restore_dtype(out, ref):
    """Keep the original dtype/scale.

    suppress_periodic works in float32, but normalise() decides between the
    percentile LUT and the "already 0..1" branch from the dtype. Returning a
    float32 array that still holds uint16 values made normalise() clip the whole
    plane to 1.0, i.e. a pure white input.
    """
    if np.issubdtype(ref.dtype, np.integer):
        info = np.iinfo(ref.dtype)
        return np.clip(np.rint(out), info.min, info.max).astype(ref.dtype)
    return out.astype(np.float32)


def suppress_periodic(arr, ratio=8.0, max_peaks=3, min_freq=8, notch_radius=2.5,
                      max_freq_frac=0.85, min_bins=16):
    """Notch out narrow periodic artifacts (DIC interference fringes, scan lines).

    A periodic artifact shows up as an isolated spike in the magnitude spectrum;
    real structure spreads over a band. So we only remove peaks that are much
    sharper than the median magnitude at the same frequency, and only outside a
    small low-frequency disk (which holds illumination gradients, not artifacts)
    and inside the outermost ring (which is pixel noise, not a fringe).

    Parameters:
        ratio         -- peak must exceed this multiple of the radial median
        max_peaks     -- at most this many frequencies are notched per image
        min_freq      -- ignore frequencies this close to DC (cycles per image)
        notch_radius  -- radius of the removed disk, in frequency bins
        max_freq_frac -- ignore frequencies beyond this fraction of Nyquist
        min_bins      -- a radius must average at least this many bins to be trusted

    Returns (filtered_float32, list_of_(dy, dx, period_px)).
    """
    a = arr.astype(np.float32)
    mean = float(a.mean())
    spec = np.fft.fftshift(np.fft.fft2(a - mean))
    mag = np.abs(spec)
    h, w = a.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot(yy - cy, xx - cx)
    r_int = np.rint(r).astype(np.int32)
    cnt = np.bincount(r_int.ravel(), minlength=int(r_int.max()) + 1)
    prof = np.bincount(r_int.ravel(), weights=mag.ravel(), minlength=cnt.size)
    prof = prof / np.maximum(cnt, 1)

    r_max = max_freq_frac * min(h, w) / 2.0
    valid = (r >= min_freq) & (r <= r_max) & (cnt[r_int] >= min_bins)
    rel = np.zeros_like(mag, dtype=np.float32)
    rel[valid] = (mag[valid] / np.maximum(prof[r_int[valid]], 1e-6)).astype(np.float32)

    hits = []
    work = rel.copy()
    for _ in range(int(max_peaks)):
        i = int(np.argmax(work))
        if work.ravel()[i] < ratio:
            break
        y, x = divmod(i, w)
        dy, dx = y - cy, x - cx
        radius = float(np.hypot(dy, dx))
        hits.append((int(dy), int(dx), float(h / max(radius, 1e-6))))
        for (oy, ox) in ((dy, dx), (-dy, -dx)):
            yy0, yy1 = max(0, cy + oy - 5), min(h, cy + oy + 6)
            xx0, xx1 = max(0, cx + ox - 5), min(w, cx + ox + 6)
            work[yy0:yy1, xx0:xx1] = 0.0

    if not hits:
        return _restore_dtype(a, arr), []

    mask = np.ones((h, w), np.float32)
    for dy, dx, _ in hits:
        for (oy, ox) in ((dy, dx), (-dy, -dx)):
            y, x = cy + oy, cx + ox
            yy0, yy1 = max(0, y - int(notch_radius)), min(h, y + int(notch_radius) + 1)
            xx0, xx1 = max(0, x - int(notch_radius)), min(w, x + int(notch_radius) + 1)
            mask[yy0:yy1, xx0:xx1] = 0.0

    out = np.real(np.fft.ifft2(np.fft.ifftshift(spec * mask))) + mean
    return _restore_dtype(out, arr), hits


def spectral_dropout(arr, n_bands=3, min_freq=8, max_freq_frac=0.85, radius=3, rng=None):
    """Randomly remove narrow frequency bands from the magnitude spectrum.

    Training-time augmentation for input images. Transmitted-light modalities
    carry periodic textures (DIC fringes, scan lines) that a network will happily
    transmit into the prediction even though the fluorescence target has no such
    periodicity. Removing random narrow bands forces the model to rely on
    broadband structure instead of any single frequency.

    Returns the filtered float32 image.
    """
    rng = rng or np.random
    a = arr.astype(np.float32)
    mean = float(a.mean())
    spec = np.fft.fftshift(np.fft.fft2(a - mean))
    h, w = a.shape
    cy, cx = h // 2, w // 2
    r_max = int(max_freq_frac * min(h, w) / 2.0)
    if r_max <= min_freq + 2:
        return a
    mask = np.ones((h, w), np.float32)
    for _ in range(int(n_bands)):
        radius_f = float(rng.uniform(min_freq, r_max))
        angle = float(rng.uniform(0.0, 2.0 * np.pi))
        y = int(round(cy + radius_f * np.sin(angle)))
        x = int(round(cx + radius_f * np.cos(angle)))
        for (oy, ox) in ((y, x), (2 * cy - y, 2 * cx - x)):
            yy0, yy1 = max(0, oy - radius), min(h, oy + radius + 1)
            xx0, xx1 = max(0, ox - radius), min(w, ox + radius + 1)
            mask[yy0:yy1, xx0:xx1] = 0.0
    out = np.real(np.fft.ifft2(np.fft.ifftshift(spec * mask))) + mean
    return out.astype(np.float32)


def preprocess_input(arr, notch=False, **notch_kwargs):
    """Input-side preprocessing shared by training and inference."""
    if notch:
        arr, _ = suppress_periodic(arr, **notch_kwargs)
    return arr


class LmcDataset(BaseDataset):
    """Random (focus, crop) pairs: input slice -> best-focus target slice."""

    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(preprocess="none", no_flip=True, load_size=512, crop_size=512)
        parser.add_argument("--target", default="nucleus", choices=CHANNELS,
                            help="fluorescence channel to predict")
        parser.add_argument("--lmc_index", default="index.json", help="index file inside dataroot")
        parser.add_argument("--lmc_src", default="",
                            help="override the data root stored in index.json (use it when the "
                                 "images live somewhere else than where the index was built)")
        parser.add_argument("--input_z_mode", default="auto", choices=("auto", "random", "best", "all"),
                            help="auto: random while training, best focus otherwise")
        parser.add_argument("--input_z_max_dev", type=int, default=0,
                            help="only sample slices within +-N of the best input focus (0 = no limit)")
        parser.add_argument("--lmc_patch", type=int, default=512, help="crop size in pixels")
        parser.add_argument("--lmc_repeat", type=int, default=8,
                            help="training only: how many random crops are drawn per field per epoch")
        parser.add_argument("--lmc_cache", type=int, default=12, help="decoded planes cached per worker")
        parser.add_argument("--lmc_norm", default="percentile", choices=("percentile", "meanstd"),
                            help="per-image intensity normalisation")
        parser.add_argument("--lmc_random_crop", type=int, default=1,
                            help="random crop while training (0 = fixed grid tiles)")
        parser.add_argument("--lmc_intensity_jitter", type=float, default=0.0,
                            help="random brightness/contrast jitter (0 = off)")
        parser.add_argument("--lmc_notch", type=int, default=0,
                            help="1 = notch out narrow periodic artifacts from the INPUT "
                                 "(DIC interference fringes, scan lines). Never applied to the target.")
        parser.add_argument("--lmc_notch_ratio", type=float, default=8.0,
                            help="a spectral peak must exceed this multiple of the radial median "
                                 "to count as an artifact")
        parser.add_argument("--lmc_notch_peaks", type=int, default=3,
                            help="max number of artifact frequencies removed per image")
        parser.add_argument("--lmc_notch_cache", default="",
                            help="directory for on-disk cache of notch-filtered input planes. "
                                 "The filter costs ~0.4 s per 2048^2 plane, so without a cache "
                                 "every epoch re-does it; with one each plane is filtered once.")
        parser.add_argument("--lmc_spec_aug", type=float, default=0.0,
                            help="probability of applying spectral band dropout to the INPUT "
                                 "during training (0 = off, 0.5 is a reasonable start)")
        parser.add_argument("--lmc_spec_bands", type=int, default=3,
                            help="narrow bands removed per spectral-dropout draw")
        parser.add_argument("--lmc_modality", default="any",
                            help="comma separated input modalities to keep (BF, DIC, PC); "
                                 "'any' uses all of them. Useful to train a modality-specific model.")
        parser.add_argument("--persistent_workers", type=int, default=1,
                            help="keep dataloader workers alive between epochs "
                                 "(avoids a ~15 s respawn per epoch on Windows)")
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        index_path = os.path.join(opt.dataroot, opt.lmc_index)
        with open(index_path) as fh:
            index = json.load(fh)
        self.src = getattr(opt, "lmc_src", "") or index["src"]
        if not os.path.isdir(self.src):
            raise RuntimeError(f"image root not found: {self.src} (fix index.json or pass --lmc_src)")
        self.channel = opt.target.lower().capitalize()
        self.patch = int(opt.lmc_patch)
        self.norm = opt.lmc_norm
        self.jitter = float(opt.lmc_intensity_jitter)
        self.notch = bool(opt.lmc_notch)
        self.notch_kwargs = {"ratio": float(opt.lmc_notch_ratio),
                             "max_peaks": int(opt.lmc_notch_peaks)}
        self.notch_cache = getattr(opt, "lmc_notch_cache", "") or ""
        if self.notch and self.notch_cache:
            os.makedirs(self.notch_cache, exist_ok=True)
        self.spec_aug = float(getattr(opt, "lmc_spec_aug", 0.0))
        self.spec_bands = int(getattr(opt, "lmc_spec_bands", 3))
        self.train = bool(opt.isTrain)

        phase = getattr(opt, "phase", "train")
        wanted_mods = [m.strip().upper() for m in str(opt.lmc_modality).split(",") if m.strip()]
        use_all_mods = not wanted_mods or wanted_mods == ["ANY"]
        self.fields = []
        for f in index["fields"]:
            if self.channel not in f["targets"]:
                continue
            if not use_all_mods and f["modality"] not in wanted_mods:
                continue
            split = f.get("split")
            if split is None:  # v1 index: split is only recorded per study
                split = ("val" if f["study"] in index["splits"].get("val", [])
                         else "test" if f["study"] in index["splits"].get("test", [])
                         else "train")
            if split == phase:
                self.fields.append(f)
        self.modalities = sorted({f["modality"] for f in self.fields})
        studies = {f["study"] for f in self.fields}
        if not self.fields:
            raise RuntimeError(f"no {self.channel} fields in split {phase!r} "
                               f"(modalities={opt.lmc_modality}, index: {index_path})")

        mode = opt.input_z_mode
        self.z_mode = ("random" if self.train else "best") if mode == "auto" else mode
        self.max_dev = int(opt.input_z_max_dev)

        self._cache = OrderedDict()
        self._cache_size = max(1, int(opt.lmc_cache))
        self.samples = self._build_samples(opt)
        print(f"lmc: channel={self.channel} phase={phase} studies={sorted(studies)} "
              f"modalities={self.modalities} fields={len(self.fields)} "
              f"samples={len(self.samples)} z_mode={self.z_mode}")

    def _crops(self, shape):
        def axis(n):
            if n <= self.patch:
                return [0]
            offs = list(range(0, n - self.patch + 1, self.patch))
            if offs[-1] != n - self.patch:
                offs.append(n - self.patch)
            return offs

        return [(y, x) for y in axis(shape[0]) for x in axis(shape[1])]

    def _build_samples(self, opt):
        """(field_idx, z or None, y, x); y=x=-1 means 'crop randomly at runtime'."""
        random_crop = self.train and bool(opt.lmc_random_crop)
        repeat = max(1, int(opt.lmc_repeat)) if self.train else 1
        samples = []
        for fi, f in enumerate(self.fields):
            zs = self._z_choices(f)
            if self.z_mode == "all":
                for z in zs:
                    if random_crop:
                        samples.extend([(fi, z, -1, -1)] * repeat)
                    else:
                        samples.extend((fi, z, y, x) for y, x in self._crops(f["shape"]))
            elif random_crop:
                samples.extend([(fi, None, -1, -1)] * repeat)
            else:
                samples.extend((fi, None, y, x) for y, x in self._crops(f["shape"]))
        return samples

    def _z_choices(self, field):
        zs = sorted(int(z) for z in field["input_z"])
        if self.max_dev > 0:
            best = int(field["input_best_z"])
            near = [z for z in zs if abs(z - best) <= self.max_dev]
            zs = near or zs
        return zs

    def __len__(self):
        return len(self.samples)

    # ---- plane loading -------------------------------------------------------
    def _notched(self, rel_path, plane):
        """Notch-filter with an optional on-disk cache (the FFT is ~0.4 s per plane)."""
        if not self.notch_cache:
            return preprocess_input(plane, notch=True, **self.notch_kwargs)
        key = "%s_r%g_p%d.npy" % (rel_path.replace("/", "__").replace(".ome.tiff", ""),
                                  self.notch_kwargs["ratio"], self.notch_kwargs["max_peaks"])
        path = os.path.join(self.notch_cache, key)
        if os.path.exists(path):
            try:
                got = np.load(path)
                if got.shape == plane.shape:
                    return got
            except (OSError, ValueError):
                pass
        out = preprocess_input(plane, notch=True, **self.notch_kwargs)
        # np.save(str) appends ".npy", so a tmp name ending in ".tmp" actually wrote
        # ".tmp.npy" and os.replace(tmp) then failed - the cache never got a hit.
        tmp = "%s.%d.tmp" % (path, os.getpid())
        try:
            with open(tmp, "wb") as fh:
                np.save(fh, out)
            os.replace(tmp, path)
        except OSError:
            if os.path.exists(tmp):
                os.remove(tmp)
        return out

    def _load(self, rel_path, is_input=True):
        cached = self._cache.get(rel_path)
        if cached is not None:
            self._cache.move_to_end(rel_path)
            return cached
        plane = tifffile.imread(os.path.join(self.src, rel_path))
        if is_input and self.notch:
            plane = self._notched(rel_path, plane)
        plane = self._normalise(plane)
        self._cache[rel_path] = plane
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return plane

    def _normalise(self, arr):
        return normalise(arr, self.norm)

    def _crop(self, plane, y, x):
        h, w = plane.shape
        if h < self.patch or w < self.patch:
            pad = ((0, max(0, self.patch - h)), (0, max(0, self.patch - w)))
            plane = np.pad(plane, pad, mode="reflect")
            h, w = plane.shape
        y = int(np.clip(y, 0, h - self.patch))
        x = int(np.clip(x, 0, w - self.patch))
        return plane[y:y + self.patch, x:x + self.patch]

    def __getitem__(self, index):
        fi, z, y, x = self.samples[index]
        field = self.fields[fi]
        zs = self._z_choices(field)

        z_in = z
        if z_in is None:
            if self.z_mode == "best":
                z_in = int(field["input_best_z"])
                if z_in not in field["input_z"]:
                    z_in = zs[len(zs) // 2]
            else:
                z_in = zs[int(np.random.randint(len(zs)))]

        if y < 0:
            h, w = field["shape"]
            y = int(np.random.randint(0, max(h - self.patch, 0) + 1))
            x = int(np.random.randint(0, max(w - self.patch, 0) + 1))

        target = field["targets"][self.channel]
        A = self._crop(self._load(field["input_z"][str(z_in)], is_input=True), y, x)
        B = self._crop(self._load(target["path"], is_input=False), y, x)

        if self.train and self.spec_aug > 0 and np.random.rand() < self.spec_aug:
            A = np.clip(spectral_dropout(A, n_bands=self.spec_bands, rng=np.random),
                        0, 255).round().astype(np.uint8)

        if self.train:
            k = int(np.random.randint(4))
            if k:
                A, B = np.rot90(A, k), np.rot90(B, k)
            if np.random.rand() < 0.5:
                A, B = A[:, ::-1], B[:, ::-1]
            if np.random.rand() < 0.5:
                A, B = A[::-1], B[::-1]

        A = torch.from_numpy(np.ascontiguousarray(A))[None].float().div_(127.5).sub_(1.0)
        B = torch.from_numpy(np.ascontiguousarray(B))[None].float().div_(127.5).sub_(1.0)
        if self.train and self.jitter > 0:
            gain = 1.0 + self.jitter * (2.0 * float(np.random.rand()) - 1.0)
            bias = self.jitter * (2.0 * float(np.random.rand()) - 1.0)
            A = (A * gain + bias).clamp_(-1.0, 1.0)
        return {
            "A": A,
            "B": B,
            "A_paths": f"{field['study']}/img{field['image']}_zin{z_in}_y{y}_x{x}",
            "B_paths": f"{field['study']}/img{field['image']}_{self.channel}_z{target['best_z']}_y{y}_x{x}",
        }
