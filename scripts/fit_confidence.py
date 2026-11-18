"""Fit the confidence-map model on the VALIDATION split.

    python scripts/fit_confidence.py

For each validation frame and the same degradations used for fusion fitting
(2x/4x/8x x {clean, noisy+missing, heavy noise+holes} + 5% sparse), the hybrid
is run, the 9 confidence features are computed and 6000 pixels are sampled
(boundary pixels over-sampled 3x). A least-squares fit predicts
log(|error| / depth + 1e-4); the residual variance s^2 gives the log-normal mean
correction exp(s^2/2), so predicted errors are unbiased on average. Result: checkpoints/confidence_params.json.
The test split is never used.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgedepth.confidence import EPS, FEATURES, PARAMS, features  # noqa: E402
from edgedepth.degradation import DegradationConfig, degrade  # noqa: E402
from edgedepth.hybrid.fusion import hybrid_reconstruct  # noqa: E402
from edgedepth.metrics import gt_edge_band  # noqa: E402
from edgedepth.model.dataset import load_split  # noqa: E402

CONDS = [dict(), dict(noise_level=1.0, missing_rate=0.05, edge_dropout=0.5), dict(noise_level=2.0, hole_blobs=3, missing_rate=0.1)]


def main():
    rng = np.random.default_rng(0)
    val = load_split("data/nyu", "val")
    X, Y = [], []
    for i, (_, rgb, gt) in enumerate(val):
        cfgs = [DegradationConfig(scale=s, seed=5000 + i * 10 + j, **c) for s in (2, 4, 8) for j, c in enumerate(CONDS)]
        cfgs.append(DegradationConfig(mode="sparse", sparse_density=0.05, seed=7000 + i))
        for cfg in cfgs:
            obs = degrade(gt, cfg)
            out, det = hybrid_reconstruct(rgb, obs, return_details=True)
            F = features(obs, out, det)
            valid = gt > 0
            p = valid.astype(np.float64) * (1 + 2 * (gt_edge_band(gt) & valid))
            p /= p.sum()
            idx = rng.choice(gt.size, 6000, replace=False, p=p.ravel())
            X.append(F.reshape(F.shape[0], -1)[:, idx].T)
            Y.append(np.log(np.abs(out.ravel()[idx] - gt.ravel()[idx]) / np.maximum(out.ravel()[idx], 1e-3) + EPS))
        print(f"frame {i + 1}/{len(val)}", flush=True)
    X, Y = np.concatenate(X), np.concatenate(Y)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w, *_ = np.linalg.lstsq(Z, Y, rcond=None)
    pred = Z @ w
    resid_var = float(np.var(Y - pred))
    # log-normal mean correction: E[exp(y)] = exp(mu + s^2/2), so predicted errors are unbiased on average
    calib = float(np.exp(resid_var / 2))
    r2 = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
    rho = float(np.corrcoef(np.argsort(np.argsort(pred)), np.argsort(np.argsort(Y)))[0, 1])
    out = {"features": FEATURES, "weights": w.tolist(), "mean": mu.tolist(), "std": sd.tolist(),
           "fitted_on": f"NYU val split, {len(Y)} pixels", "residual_var_log": resid_var, "mean_correction": calib, "val_r2_log_error": float(r2), "val_spearman": rho}
    with open(PARAMS, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("mean", "std")}, indent=1))


if __name__ == "__main__":
    main()
