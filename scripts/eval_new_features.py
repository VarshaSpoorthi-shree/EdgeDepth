"""Evaluate the confidence map and the flying-pixel metric on the unseen TEST split.

    python scripts/eval_new_features.py

Inputs: 24 test frames x {2x, 4x, 8x} x {clean, noisy+missing} + 5% sparse (168
observations, same seeds as scripts/evaluate.py).

Confidence map (hybrid output):
  * AUSE (area under sparsification error, normalised; lower = better) of
      - the fitted predicted-error map,
      - a naive baseline: disagreement between the experts only,
      - random ordering (expected AUSE of a useless map);
  * Spearman rank correlation between predicted and true per-pixel error;
  * mean sparsification curves (RMSE of the kept pixels vs fraction removed);
  * calibration: mean true error inside deciles of the predicted error;
  * flying pixels left after hiding the 10 % / 20 % least-confident pixels.
Flying pixels: % of discontinuity-region pixels that are flying, every method.
Outputs results/new_features.json and results/figures/new/*.png.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgedepth import METHOD_LABELS, METHODS  # noqa: E402
from edgedepth.confidence import ause, predict_error  # noqa: E402
from edgedepth.degradation import DegradationConfig, degrade  # noqa: E402
from edgedepth.hybrid.fusion import hybrid_reconstruct  # noqa: E402
from edgedepth.metrics import flying_pixel_mask  # noqa: E402
from edgedepth.model.dataset import load_split  # noqa: E402
from edgedepth.pipeline import run_method  # noqa: E402

FR = np.linspace(0, 0.9, 19)
NOISY = dict(noise_level=1.0, missing_rate=0.05, edge_dropout=0.5)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    test = load_split("data/nyu", "test")
    rng = np.random.default_rng(0)
    conf_rows, fly_rows = [], []
    curves = {"learned": [], "expert_std": [], "oracle": []}
    calib_pred, calib_true = [], []
    t0 = time.time()
    for i, (name, rgb, gt) in enumerate(test):
        cases = [(f"{s}x", c, DegradationConfig(scale=s, seed=20000 + i, **kw)) for s in (2, 4, 8)
                 for c, kw in (("clean", {}), ("noisy_missing", NOISY))]
        cases.append(("sparse", "sparse_5pct", DegradationConfig(mode="sparse", sparse_density=0.05, seed=30000 + i)))
        for sc, cond, cfg in cases:
            obs = degrade(gt, cfg)
            out, det = hybrid_reconstruct(rgb, obs, return_details=True)
            pe = predict_error(obs, out, det)
            v = gt > 0
            err = np.abs(out - gt)[v]
            pred = pe["error_m"][v]
            std = np.stack(list(det["candidates"].values()), 0).std(0)[v]
            a_l, c_l, c_o = ause(err, pred, FR)
            a_s, c_s, _ = ause(err, std, FR)
            a_r, _, _ = ause(err, rng.random(err.shape), FR)
            curves["learned"].append(c_l)
            curves["expert_std"].append(c_s)
            curves["oracle"].append(c_o)
            # flying pixels of the hybrid, all pixels vs after hiding the least-confident ones
            fly, region = flying_pixel_mask(out, gt)
            conf = pe["confidence"]
            hide = {}
            for frac in (0.1, 0.2):
                thr = np.quantile(conf[v], frac)
                keep = conf > thr
                hide[f"flying_pct_after_hiding_{int(frac * 100)}pct"] = float(100 * (fly & keep).sum() / max((region & keep).sum(), 1))
            conf_rows.append({"frame": name, "scale": sc, "condition": cond, "ause_learned": a_l, "ause_expert_std": a_s,
                              "ause_random": a_r, "spearman_learned": spearman(pred, err), "spearman_expert_std": spearman(std, err),
                              "flying_pct_all": float(100 * fly.sum() / max(region.sum(), 1)), **hide})
            idx = rng.choice(len(err), min(4000, len(err)), replace=False)
            calib_pred.append(pred[idx])
            calib_true.append(err[idx])
            for m in METHODS:
                d = out if m == "hybrid" else run_method(m, rgb, obs)
                f, rg = flying_pixel_mask(d, gt)
                fly_rows.append({"frame": name, "scale": sc, "condition": cond, "method": m,
                                 "flying_pct": float(100 * f.sum() / max(rg.sum(), 1))})
        print(f"{i + 1}/{len(test)} {name} {time.time() - t0:.0f}s", flush=True)

    def mean(rows, key, **flt):
        sel = [r[key] for r in rows if all(r[k] == v for k, v in flt.items())]
        return float(np.mean(sel)) if sel else None

    cp, ct = np.concatenate(calib_pred), np.concatenate(calib_true)
    edges = np.quantile(cp, np.linspace(0, 1, 11))
    calib = []
    for b in range(10):
        s = (cp >= edges[b]) & (cp <= edges[b + 1])
        calib.append({"decile": b + 1, "mean_predicted_m": float(cp[s].mean()), "mean_true_m": float(ct[s].mean())})
    res = {
        "protocol": "NYU test split, 24 frames x {2x,4x,8x} x {clean, noisy_missing} + 5% sparse = 168 observations, seeds as evaluate.py",
        "confidence": {
            "overall": {k: mean(conf_rows, k) for k in ("ause_learned", "ause_expert_std", "ause_random", "spearman_learned",
                                                         "spearman_expert_std", "flying_pct_all", "flying_pct_after_hiding_10pct",
                                                         "flying_pct_after_hiding_20pct")},
            "by_scale": {sc: {k: mean(conf_rows, k, scale=sc) for k in ("ause_learned", "ause_expert_std", "ause_random", "spearman_learned")}
                         for sc in ("2x", "4x", "8x", "sparse")},
            "sparsification_fractions": FR.tolist(),
            "mean_curves": {k: np.mean(v, 0).tolist() for k, v in curves.items()},
            "calibration_deciles": calib,
        },
        "flying_pixels": {
            "overall": {m: mean(fly_rows, "flying_pct", method=m) for m in METHODS},
            "by_scale": {sc: {m: mean(fly_rows, "flying_pct", method=m, scale=sc) for m in METHODS} for sc in ("2x", "4x", "8x", "sparse")},
        },
    }
    with open("results/new_features.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"confidence": res["confidence"]["overall"], "conf_by_scale": res["confidence"]["by_scale"],
                      "flying": res["flying_pixels"]}, indent=1))
    make_figures(res)


def make_figures(res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("results/figures/new", exist_ok=True)
    c = res["confidence"]
    fr = np.array(c["sparsification_fractions"]) * 100
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(fr, c["mean_curves"]["oracle"], "k--", label="Oracle (true error)")
    ax[0].plot(fr, c["mean_curves"]["learned"], color="tab:green", lw=2.5, label=f"EdgeDepth confidence (AUSE {c['overall']['ause_learned']:.3f})")
    ax[0].plot(fr, c["mean_curves"]["expert_std"], color="tab:orange", label=f"Expert disagreement only (AUSE {c['overall']['ause_expert_std']:.3f})")
    ax[0].axhline(1.0, color="gray", lw=1, ls=":", label=f"Random (AUSE ≈ {c['overall']['ause_random']:.3f})")
    ax[0].set_xlabel("% of least-confident pixels removed")
    ax[0].set_ylabel("RMSE of remaining pixels (relative to all)")
    ax[0].set_title("Sparsification: does low confidence mark real errors?")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)
    cal = c["calibration_deciles"]
    p = [x["mean_predicted_m"] * 100 for x in cal]
    t = [x["mean_true_m"] * 100 for x in cal]
    ax[1].plot(p, t, "o-", color="tab:green", label="EdgeDepth confidence")
    m = max(max(p), max(t))
    ax[1].plot([0, m], [0, m], "k--", lw=1, label="perfect calibration")
    ax[1].set_xlabel("predicted error (cm), decile mean")
    ax[1].set_ylabel("true error (cm), decile mean")
    ax[1].set_title("Calibration on unseen test frames")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/figures/new/confidence_evaluation.png", dpi=120, facecolor="white")
    plt.close(fig)

    fp = res["flying_pixels"]["by_scale"]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ms = list(res["flying_pixels"]["overall"])
    xs = np.arange(len(ms))
    for j, sc in enumerate(("2x", "4x", "8x", "sparse")):
        ax.bar(xs + (j - 1.5) * 0.2, [fp[sc][m] for m in ms], 0.2, label=sc)
    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABELS[m].split(" (")[0] for m in ms], fontsize=8)
    ax.set_ylabel("% flying pixels at discontinuities ↓")
    ax.set_title("Flying pixels per method (unseen test split, clean + noisy)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig("results/figures/new/flying_pixels.png", dpi=120, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
