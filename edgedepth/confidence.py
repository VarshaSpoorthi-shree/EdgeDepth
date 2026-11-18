r"""Per-pixel confidence: a predicted-error map for the hybrid reconstruction.

Every pixel of the output gets an estimate of how wrong it is likely to be
(in metres) and a confidence score in [0, 1]. Nothing here sees the ground
truth at inference time; the map is built only from signals the hybrid
already computes:

    f1  disagreement between the four experts      log(std_k D_k / D)
    f2  size of the back-projection correction     log(|D_out - D_fused| / D)
    f3  network vs. interpolation disagreement     log(|D_dl - D_bicubic| / D)
    f4  network boundary probability               B_dl
    f5  discontinuity probability of the output    1 - exp(-(|grad log D|/0.03)^2)
    f6  distance to the nearest real measurement   (pixels / measurement spacing)
    f7  estimated relative sensor noise            log(sigma_hat / median depth)   (per image)
    f8  measurement spacing                        log(effective scale)            (per image)
    f9  missing-measurement indicator              H

A linear model in these features predicts log relative error,

    log(|D - D_gt| / D + 1e-4)  ~  w0 + sum_i w_i f_i,

fitted by least squares on pixels of the VALIDATION split
(scripts/fit_confidence.py). The predicted error is then
    e_hat(p) = D(p) * (k exp(w . f(p)) - 1e-4)     [metres]
with k = exp(s^2/2) the log-normal mean correction (s^2 = residual variance).
and the confidence is c(p) = exp(-e_hat(p) / (0.05 D(p))), i.e. 1 for a pixel
expected to be exact, 0.37 for a pixel expected to be 5 % off.
Quality is measured on the TEST split with sparsification curves (AUSE) and
rank correlation against the true error (scripts/eval_new_features.py).
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import cv2
import numpy as np

from .classical.edges import discontinuity_probability
from .classical.holes import hole_distance
from .classical.interpolation import upsample_mask
from .degradation import Observation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAMS = os.path.join(ROOT, "checkpoints", "confidence_params.json")
FEATURES = ["expert_disagreement", "bp_correction", "dl_vs_bicubic", "network_boundary_prob", "output_discontinuity",
            "distance_to_measurement", "sensor_noise", "measurement_spacing", "missing_indicator"]
EPS = 1e-4
REL_TOL = 0.05  # error (relative to depth) at which confidence = 1/e


def features(obs: Observation, out: np.ndarray, details: Dict) -> np.ndarray:
    """Return the (F, H, W) feature stack for a hybrid output and its details."""
    cands = details["candidates"]
    D = np.maximum(out, 1e-3)
    stack = np.stack([cands[k] for k in cands], 0)
    f1 = np.log(stack.std(0) / D + EPS)
    f2 = np.log(np.abs(details["correction"]) / D + EPS)
    dl = cands.get("deep_learning", details["fused_before_refine"])
    f3 = np.log(np.abs(dl - cands["bicubic"]) / D + EPS)
    f4 = details["cues"]["B_dl"]
    f5 = discontinuity_probability(D)
    spacing = max(obs.effective_scale, 1.0)
    f6 = np.clip(hole_distance(upsample_mask(obs)) / spacing, 0, 20)
    pre = details.get("preprocess", {})
    f7 = np.full(D.shape, np.log(pre.get("rel_noise", 0.0) + EPS), np.float32)
    f8 = np.full(D.shape, np.log(spacing), np.float32)
    f9 = details["cues"]["H"]
    return np.stack([f1, f2, f3, f4, f5, f6, f7, f8, f9], 0).astype(np.float32)


def load_params(path: str = PARAMS) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def predict_error(obs: Observation, out: np.ndarray, details: Dict, params: Optional[Dict] = None) -> Dict[str, np.ndarray]:
    """Predicted absolute error (m) and confidence in [0, 1] for every pixel."""
    params = params or load_params()
    F = features(obs, out, details)
    D = np.maximum(out, 1e-3)
    if params is None:  # untrained fallback: expert disagreement only
        rel = np.exp(F[0]) - EPS
    else:
        w = np.asarray(params["weights"], np.float32)
        mu, sd = np.asarray(params["mean"], np.float32), np.asarray(params["std"], np.float32)
        Z = (F - mu[:, None, None]) / sd[:, None, None]
        rel = np.exp(w[0] + np.einsum("f,fhw->hw", w[1:], Z)) * params.get("mean_correction", 1.0) - EPS
    rel = np.clip(cv2.GaussianBlur(rel.astype(np.float32), (3, 3), 0), 0, 5)
    err = rel * D
    conf = np.exp(-rel / REL_TOL)
    return {"error_m": err.astype(np.float32), "confidence": conf.astype(np.float32), "fitted": params is not None}


# ---------- evaluation helpers ----------
def sparsification(err_true: np.ndarray, score: np.ndarray, fractions=np.linspace(0, 0.9, 19)):
    """RMSE of the pixels that remain after removing the `f` most uncertain ones (by `score`)."""
    e = err_true.ravel()
    order = np.argsort(-score.ravel())  # most uncertain first
    e2 = e[order] ** 2
    n = len(e)
    out = []
    for f in fractions:
        k = int(round(f * n))
        out.append(float(np.sqrt(e2[k:].mean())))
    return np.asarray(out)


def ause(err_true: np.ndarray, score: np.ndarray, fractions=np.linspace(0, 0.9, 19)):
    """Area between the sparsification curve and the oracle (lower = better), normalised by the full RMSE."""
    s = sparsification(err_true, score, fractions)
    o = sparsification(err_true, np.abs(err_true), fractions)
    full = s[0] if s[0] > 0 else 1.0
    d = s - o
    area = float(np.sum((d[1:] + d[:-1]) / 2 * np.diff(fractions)))  # trapezoid rule (works on NumPy 1.x and 2.x)
    return area / full, s / full, o / full
