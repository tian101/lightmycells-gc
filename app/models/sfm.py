"""Adaptive Resampling (ARS) from "Spatial Frequency Modulation for Semantic
Segmentation" (Linwei Chen et al., TPAMI 2025, https://github.com/Linwei-Chen/SFM).

Why this exists here
--------------------
A strided convolution simply drops every other pixel, so detail finer than the
new Nyquist limit folds back as aliasing. Plain anti-aliasing (BlurPool) *removes*
that band - which we verified does nothing for us: the prediction stays smooth
because the detail is thrown away instead of being carried down.

ARS takes the opposite approach. It predicts a saliency map, turns it into a
non-uniform sampling grid (Gaussian-weighted centroid, Eq. 2/3 of the paper) and
warps the feature map so that high-frequency regions are sampled densely. Those
regions are spatially compressed, their content moves to a lower frequency band,
and therefore survives the following stride. The decoder then has to undo the
warp - in the original work that is MSAU; in a U-Net the skip connections plus
the transposed-conv decoder play that role implicitly.

This is a faithful port of ``sfm/adaptive_sampling.py`` (``SaliencySampler``,
``sampler_mode='avg_lhpf_psp'``) with three deliberate changes for this repo:

* ``SyncBatchNorm`` -> ``BatchNorm2d`` (we train on a single GPU),
* no ``sampler_mode`` switch: only the released ``avg_lhpf_psp`` branch is kept,
* grid/PSP sizes are scaled per U-Net level instead of hard-coded, because a
  U-Net sees everything from 512x512 down to 2x2.

The inverse (qhull triangulation) upsampling path is not ported: it triangulates
h*w points per sample and is far too slow to run at every U-Net level.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["makeGaussian", "LHPFConv3", "PSP", "PSPModule", "SaliencyResampler",
           "FAN", "AdaptiveUpsampler"]


def makeGaussian(size, fwhm=3, center=None):
    """2-D Gaussian of edge length ``size``; ``fwhm`` sets how localised it is."""
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0, y0 = center
    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)


class LHPFConv3(nn.Module):
    """Learnable high-pass filter (3x3 depthwise).

    The kernel is soft-max normalised into a low-pass kernel and subtracted from
    the identity, which yields a high-pass kernel. High-frequency energy is what
    SFM modulates, so the saliency predictor sees it explicitly first.
    """

    def __init__(self, channels=3, stride=1, padding=1, residual=False):
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.padding = padding
        base = torch.tensor([[1., 1., 1.],
                             [1., 0., 1.],
                             [1., 1., 1.]], dtype=torch.float32)
        base = base.unsqueeze(0).unsqueeze(0).repeat(self.channels, 1, 1, 1)
        self.weight = nn.Parameter(base, requires_grad=True)
        identity = torch.tensor([[0., 0., 0.],
                                 [0., 1., 0.],
                                 [0., 0., 0.]], dtype=torch.float32)
        if residual:
            identity = identity * 2
        self.register_buffer("identity_kernel", identity.unsqueeze(0).unsqueeze(0))

    def forward(self, x):
        weight_softmax = self.weight.reshape(self.channels, 1, -1).softmax(dim=-1)
        low_pass = weight_softmax.reshape(self.channels, 1, 3, 3)
        high_pass = self.identity_kernel - low_pass
        return F.conv2d(x, high_pass, padding=self.padding, groups=self.channels, stride=self.stride)


class PSPModule(nn.Module):
    """AdaptiveAvgPool(bin) -> 1x1 conv -> BN -> ReLU -> resize back."""

    def __init__(self, in_channels, out_channels, bin_size):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(bin_size)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.size()
        out = self.global_avg_pool(x)
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return F.interpolate(out, size[2:], mode="bilinear", align_corners=True)


class PSP(nn.Module):
    """Pyramid pooling context, concatenated with the input."""

    def __init__(self, in_channels, out_channels, bins=(1, 2, 3, 6)):
        super().__init__()
        self.psp1 = PSPModule(in_channels, out_channels, bins[0])
        self.psp2 = PSPModule(in_channels, out_channels, bins[1])
        self.psp3 = PSPModule(in_channels, out_channels, bins[2])
        self.psp4 = PSPModule(in_channels, out_channels, bins[3])

    def forward(self, x):
        return torch.cat([x, self.psp1(x), self.psp2(x), self.psp3(x), self.psp4(x)], dim=1)


class FAN(nn.Module):
    """Frequency Adaptive Normalization (SFM, Linwei Chen et al.).

    A single set of normalization statistics is dominated by the low-frequency
    response (most of the energy in a natural or microscopy feature map), so the
    high-frequency branch gets washed out. FAN splits the feature into a
    low-frequency part (local average) and a high-frequency residual, normalizes
    the two branches with independent statistics, and fuses them with a learnable
    per-channel weight. That keeps the fine response alive instead of letting the
    dominant smooth term set the scale.
    """

    def __init__(self, channels, norm_layer=None, kernel=3):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.low_norm = norm_layer(channels)
        self.high_norm = norm_layer(channels)
        pad = kernel // 2
        self.blur = nn.AvgPool2d(kernel, stride=1, padding=pad, count_include_pad=False)
        # softmax over the branch axis, initialised 50/50
        self.fuse = nn.Parameter(torch.zeros(2, channels, 1, 1))

    def forward(self, x):
        lo = self.blur(x)
        hi = x - lo
        w = self.fuse.softmax(dim=0)
        return w[0] * self.low_norm(lo) + w[1] * self.high_norm(hi)


class AdaptiveUpsampler(nn.Module):
    """MSAU: undo the ARS warp with a splatting-based inverse grid.

    The released implementation inverts the non-uniform sampling grid with a
    qhull triangulation over h*w points, which is far too slow to run at every
    U-Net level. We build the inverse by scattering instead: the ARS grid is a
    gather map (output position -> source position), so every source pixel knows
    where it landed. Accumulating the destination positions into their source
    bins and averaging yields the reverse map at O(pixels) cost. Bins that
    receive no contribution fall back to their uniform coordinate, which keeps
    the map a smooth, monotone deformation instead of collapsing.

    Without this, the decoder reconstructs in the warped coordinate frame while
    the skip connection is still uniform, and the mismatch shows up as blur.
    """

    def __init__(self, align_corners=False):
        super().__init__()
        self.align_corners = align_corners

    def inverse_grid(self, grid):
        """grid: [B,2,h,w] normalized gather map -> reverse map of the same shape."""
        b, _, h, w = grid.shape
        dev, dt = grid.device, grid.dtype
        ones = torch.ones(b, h * w, device=dev, dtype=dt)

        if self.align_corners:
            dst_x = (grid[:, 0] + 1) * 0.5 * (w - 1)
            dst_y = (grid[:, 1] + 1) * 0.5 * (h - 1)
        else:
            dst_x = ((grid[:, 0] + 1) * w - 1) * 0.5
            dst_y = ((grid[:, 1] + 1) * h - 1) * 0.5

        idx_x = dst_x.round().clamp(0, w - 1).long().reshape(b, -1)
        idx_y = dst_y.round().clamp(0, h - 1).long().reshape(b, -1)
        bin_idx = (idx_y * w + idx_x).clamp(0, h * w - 1)

        # destination positions (the coordinates we want to recover)
        if self.align_corners:
            ex = torch.arange(w, device=dev, dtype=dt) / max(w - 1, 1) * 2 - 1
            ey = torch.arange(h, device=dev, dtype=dt) / max(h - 1, 1) * 2 - 1
        else:
            ex = (torch.arange(w, device=dev, dtype=dt) + 0.5) / w * 2 - 1
            ey = (torch.arange(h, device=dev, dtype=dt) + 0.5) / h * 2 - 1
        gy, gx = torch.meshgrid(ey, ex, indexing="ij")
        src_x = gx.reshape(1, -1).expand(b, -1).contiguous()
        src_y = gy.reshape(1, -1).expand(b, -1).contiguous()

        acc_x = torch.zeros(b, h * w, device=dev, dtype=dt)
        acc_y = torch.zeros(b, h * w, device=dev, dtype=dt)
        cnt = torch.zeros(b, h * w, device=dev, dtype=dt)
        acc_x.scatter_add_(1, bin_idx, src_x)
        acc_y.scatter_add_(1, bin_idx, src_y)
        cnt.scatter_add_(1, bin_idx, ones)

        hit = cnt > 0.5
        inv_x = torch.where(hit, acc_x / cnt.clamp(min=1.0), src_x)
        inv_y = torch.where(hit, acc_y / cnt.clamp(min=1.0), src_y)
        return torch.cat([inv_x.view(b, 1, h, w), inv_y.view(b, 1, h, w)], 1)

    def forward(self, feature, grid):
        inv = self.inverse_grid(grid.to(feature.dtype))
        return F.grid_sample(feature, inv.permute(0, 2, 3, 1),
                             mode="bilinear", align_corners=self.align_corners,
                             padding_mode="border")


class SaliencyResampler(nn.Module):
    """ARS: warp features by a saliency-driven non-uniform grid before a stride.

    Args:
        feat_dim: input channels.
        padding_size: search radius (px) of the Gaussian-weighted centroid.
        fwhm: width of that Gaussian; smaller stays closer to the peak.
        psp_ratio: channel reduction inside PSP.
        align_corners: must match the ``grid_sample`` convention.
        dilation: dilation of the final 3x3 saliency conv.
    """

    def __init__(self, feat_dim, padding_size=8, fwhm=8, psp_ratio=16,
                 align_corners=False, dilation=1):
        super().__init__()
        self.feat_dim = feat_dim
        self.padding_size = int(padding_size)
        self.fwhm = float(fwhm)
        self.align_corners = align_corners
        self.eps = 1e-16
        reduced = max(1, feat_dim // psp_ratio)
        self.psp_bins = [1, 2, 3, 7] if align_corners else [1, 2, 3, 6]

        self.saliency_conv = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            LHPFConv3(channels=feat_dim, stride=1, padding=1, residual=True),
            PSP(feat_dim, reduced, bins=self.psp_bins),
            nn.Conv2d(feat_dim + reduced * 4, 1, kernel_size=3,
                      padding=dilation, dilation=dilation, stride=1),
        )

        self._build_filter()

    def _build_filter(self):
        """(Re)build the fixed Gaussian kernel used for the weighted centroid."""
        kernel = torch.FloatTensor(makeGaussian(2 * self.padding_size + 1, fwhm=self.fwhm))
        if "gaussian_weights" in self._buffers:
            del self._buffers["gaussian_weights"]
        self.register_buffer("gaussian_weights", kernel.view(1, 1, *kernel.shape))
        self.filter = nn.Conv2d(1, 1, kernel_size=(2 * self.padding_size + 1,) * 2, bias=False)
        self.filter.weight.data[:, :, :, :] = self.gaussian_weights
        self.filter.weight.requires_grad = False
        self._coord_cache = {}

    def set_window(self, padding_size=None, fwhm=None):
        """Resize the centroid search window (released SFM uses pad=5, fwhm=3)."""
        if padding_size is not None:
            self.padding_size = int(padding_size)
        if fwhm is not None:
            self.fwhm = float(fwhm)
        self._build_filter()

    # ---------------------------------------------------------------- coords
    def gen_coord_grid(self, h, w, device=None):
        """Uniform coordinates over the padded extent, shape [2, h+2p, w+2p]."""
        key = f"{h}x{w}"
        cached = self._coord_cache.get(key)
        if cached is None:
            p = self.padding_size
            span_x = torch.arange(-p, w + p, dtype=torch.float32)
            span_y = torch.arange(-p, h + p, dtype=torch.float32)
            if self.align_corners:
                x = span_x / (w - 1.0) if w > 1 else span_x * 0
                y = span_y / (h - 1.0) if h > 1 else span_y * 0
            else:
                x = span_x / w + 0.5 / w
                y = span_y / h + 0.5 / h
            cached = torch.stack(torch.meshgrid(y, x, indexing="ij"), dim=0)
            self._coord_cache[key] = cached
        return cached.to(device).detach()

    def gen_coord_grid_without_pad(self, h, w, device=None):
        p = self.padding_size
        return self.gen_coord_grid(h, w, device)[:, p:-p, p:-p]

    # ------------------------------------------------------------------ grid
    def create_grid(self, density, h, w):
        """Gaussian-weighted centroid of the saliency density -> sampling grid."""
        p = self.padding_size
        x = F.pad(density, pad=[p] * 4, mode="reflect")
        padded_h, padded_w = h + 2 * p, w + 2 * p
        P = self.gen_coord_grid(h, w, device=x.device)[None]
        P = P.expand(x.size(0), 2, padded_h, padded_w).to(x.device)

        x_cat = torch.cat((x, x), 1)
        p_filter = self.filter(x).float()
        x_mul = torch.mul(P, x_cat).reshape(-1, 1, padded_h, padded_w)
        all_filter = self.filter(x_mul).reshape(-1, 2, h, w)

        # P[0] is the y coordinate (varies along rows) and P[1] is x (varies along
        # columns), because gen_coord_grid stacks meshgrid(y, x, indexing="ij").
        # grid_sample expects channel 0 = x and channel 1 = y, so map them back.
        #
        # p_filter is the Gaussian-weighted saliency mass inside the search window.
        # When the saliency collapses onto a few pixels it underflows to 0 away from
        # them, and the centroid becomes 0/eps = 0 -> grid = -1 for the whole map:
        # every output pixel then samples the same corner and the model degenerates
        # to a constant image. Wherever the window holds no mass we therefore fall
        # back to the uniform coordinate, i.e. that region is simply not warped.
        floor = 1e-6
        denom = p_filter.clamp(min=floor)
        has_mass = p_filter > floor
        y_centroid = torch.where(has_mass, all_filter[:, 0, :, :].reshape(-1, 1, h, w) / denom,
                                 P[:, 0:1, p:-p, p:-p])
        x_centroid = torch.where(has_mass, all_filter[:, 1, :, :].reshape(-1, 1, h, w) / denom,
                                 P[:, 1:2, p:-p, p:-p])
        xgrids = x_centroid * 2 - 1
        ygrids = y_centroid * 2 - 1
        clamp_x = 1 if self.align_corners else (1 - 0.5 / w * 2)
        clamp_y = 1 if self.align_corners else (1 - 0.5 / h * 2)
        xgrids = torch.clamp(xgrids, min=-clamp_x, max=clamp_x)
        ygrids = torch.clamp(ygrids, min=-clamp_y, max=clamp_y)

        grid = torch.cat((xgrids.view(-1, 1, h, w), ygrids.view(-1, 1, h, w)), 1)
        # pin the border rows/cols back to their uniform position
        grid[:, 1, 0, :] = P[:, 0, p, p:-p] * 2 - 1.0
        grid[:, 1, -1, :] = P[:, 0, -p - 1, p:-p] * 2 - 1.0
        grid[:, 0, :, 0] = P[:, 1, p:-p, p] * 2 - 1.0
        grid[:, 0, :, -1] = P[:, 1, p:-p, -p - 1] * 2 - 1.0
        return grid

    def reset_to_identity(self):
        """Zero the saliency head so the warp starts as the identity mapping.

        With random weights the predicted saliency is noise, the centroid grid
        comes out non-monotonic and grid_sample scrambles the feature map before
        the network has learned anything. Starting from an all-zero logit makes
        the density uniform, hence an identity grid, and the warp is then learned
        from a sane point instead of degrading gracefully from a broken one.
        """
        last = self.saliency_conv[-1]
        nn.init.zeros_(last.weight)
        if last.bias is not None:
            nn.init.zeros_(last.bias)

    def warp_with(self, feature, grid):
        """Apply an already-computed sampling grid (same shape as the input)."""
        return F.grid_sample(feature, grid.permute(0, 2, 3, 1),
                             mode="bilinear", align_corners=self.align_corners,
                             padding_mode="border")

    def forward(self, feature):
        """Return the warped feature (same shape as the input).

        The sampling grid is kept on ``last_grid`` so the caller can apply the
        *same* warp to the skip-connection tensor; warping only one of the two
        branches leaves them in different coordinate frames and the U-Net would
        concatenate misaligned features.
        """
        _, _, h, w = feature.shape
        saliency = self.saliency_conv(feature)
        # per-pixel softmax -> density with total mass equal to #pixels
        saliency = saliency.reshape(-1, h * w).softmax(dim=-1).reshape(-1, 1, h, w) * h * w
        grid = self.create_grid(saliency, h, w)
        self.last_grid = grid
        return self.warp_with(feature, grid)
