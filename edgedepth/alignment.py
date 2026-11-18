r"""Automatic RGB–depth misalignment estimation and correction.

RGB-guided methods assume the colour image and the depth map are pixel-aligned.
A calibration or synchronisation error shifts every colour edge, and the
reconstruction then puts depth boundaries in the wrong place (failure case F5).

Estimation (no ground truth needed)
-----------------------------------
1. Build an RGB-free depth estimate D0 from the observation alone (bicubic of
   the hole-filled low-res grid, or push-pull fill of sparse samples). Its edges
   are blurry but sit at the *true* positions, because no RGB was used.
2. Depth-edge map  E_d = |grad log D0|   (relative depth change).
   Colour-edge map E_c = Sobel colour gradient, blurred by sigma ~ s/2 so its
   sharpness matches the blur of D0.
3. For every integer shift (dy, dx) in [-R, R]^2, compute the zero-mean
   normalised cross-correlation  NCC(E_d, shift(E_c, dy, dx))  over the central
   region. Real depth edges coincide with colour edges at the correct alignment,
   so the correlation peaks there; texture edges add noise but no consistent peak.
4. The shift is accepted only if it is larger than 1 px and beats "no shift"
   by a margin (gain = NCC(best) - NCC(0) > 0.05). Otherwise the input is
   treated as aligned and left untouched; on validation no correctly aligned
   input was changed with this rule.

Correction: the RGB image is shifted by the estimated (dy, dx) with replicated
borders; the depth grid defines the output geometry and is never moved.
Parameters were tuned on the VALIDATION split (scripts/tune_alignment.py).
"""
from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from .classical.edges import depth_gradient_rel
from .classical.holes import push_pull_fill
from .classical.interpolation import interpolate
from .degradation import Observation

MAX_SHIFT = 10          # search range in HR pixels
BLUR_FACTOR = 0.5       # colour-edge blur sigma = max(1, BLUR_FACTOR * spacing)
GAIN_THRESHOLD = 0.05   # minimum NCC improvement over "no shift" to correct (set on VAL: no aligned case changed)
DEAD_ZONE = 1           # shifts of <= 1 px are ignored (sub-pixel registration noise; NYU itself has ~1 px)
COMPRESS = 0.5          # edge magnitudes are raised to this power (reduces dominance of strong texture)


def shift_image(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate an image by (dy, dx) pixels with replicated borders."""
    if dy == 0 and dx == 0:
        return img.copy()
    H, W = img.shape[:2]
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)


def _edge_maps(rgb: np.ndarray, obs: Observation, blur_factor: float, compress: float):
    d0 = push_pull_fill(obs.values, obs.mask) if obs.mode == "sparse" else interpolate(obs, "bicubic")
    if obs.mode == "sparse":  # smooth the push-pull fill so its edges are comparable
        d0 = cv2.GaussianBlur(d0, (0, 0), max(1.0, 0.5 * obs.effective_scale))
    ed = depth_gradient_rel(np.maximum(d0, 1e-3))
    g = rgb.astype(np.float32) / 255.0
    sig = max(1.0, blur_factor * obs.effective_scale)
    g = cv2.GaussianBlur(g, (0, 0), sig)
    gx = np.stack([cv2.Sobel(g[..., c], cv2.CV_32F, 1, 0, ksize=3) for c in range(3)])
    gy = np.stack([cv2.Sobel(g[..., c], cv2.CV_32F, 0, 1, ksize=3) for c in range(3)])
    ec = np.sqrt((gx ** 2 + gy ** 2).max(0))
    return np.power(ed, compress).astype(np.float32), np.power(ec, compress).astype(np.float32)


def estimate_shift(rgb: np.ndarray, obs: Observation, max_shift: int = MAX_SHIFT, blur_factor: float = BLUR_FACTOR,
                   compress: float = COMPRESS) -> Dict:
    """Return the shift (dy, dx) to apply to the RGB image, the NCC surface and the gain over no shift."""
    ed, ec = _edge_maps(rgb, obs, blur_factor, compress)
    H, W = ed.shape
    R = max_shift
    a = ed[R:H - R, R:W - R]
    a = (a - a.mean()) / (a.std() + 1e-9)
    ecp = np.pad(ec, R, mode="edge")
    ncc = np.full((2 * R + 1, 2 * R + 1), -1.0, np.float32)
    for i, dy in enumerate(range(-R, R + 1)):
        for j, dx in enumerate(range(-R, R + 1)):
            # content of the RGB shifted by (dy, dx): out[y, x] = ec[y - dy, x - dx]
            b = ecp[2 * R - dy:H - dy, 2 * R - dx:W - dx][: H - 2 * R, : W - 2 * R]
            b = (b - b.mean()) / (b.std() + 1e-9)
            ncc[i, j] = float((a * b).mean())
    i, j = np.unravel_index(int(np.argmax(ncc)), ncc.shape)
    dy, dx = int(i - R), int(j - R)
    gain = float(ncc[i, j] - ncc[R, R])
    return {"dy": dy, "dx": dx, "ncc": ncc, "ncc_best": float(ncc[i, j]), "ncc_zero": float(ncc[R, R]), "gain": gain}


def auto_align(rgb: np.ndarray, obs: Observation, threshold: float = GAIN_THRESHOLD, **kw) -> Tuple[np.ndarray, Dict]:
    """Estimate the misalignment and, if it is significant, return the corrected RGB image."""
    est = estimate_shift(rgb, obs, **kw)
    apply = max(abs(est["dy"]), abs(est["dx"])) > DEAD_ZONE and est["gain"] > threshold
    est["applied"] = bool(apply)
    if not apply:
        est["dy_applied"], est["dx_applied"] = 0, 0
        return rgb, est
    est["dy_applied"], est["dx_applied"] = est["dy"], est["dx"]
    return shift_image(rgb, est["dy"], est["dx"]), est
