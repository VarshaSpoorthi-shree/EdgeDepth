"""Extremely sparse inputs: when does reconstruction become unreliable, and does the
confidence map say so?

    python scripts/eval_sparse_confidence.py      # took ~25 min on 2 CPU cores (sharing them with another run)

Densities 10, 5, 2, 1, 0.5, 0.2 % of the 320x240 pixels (random samples, seeds as
scripts/failure_cases.py F1 for the test split), on ALL frames of
  * the validation split (29 frames)  - used only to choose the app's warning threshold
  * the test split (24 frames)        - every reported number
Per observation: hybrid / network-only / bicubic RMSE (the latter two are read from the
hybrid's own candidates, so they are the same experts), and for the confidence map:
  * AUSE (vs random ordering) and Spearman correlation with the true error,
  * calibration ratio = mean predicted error / mean true error (1 = calibrated),
  * flagging of "unreliable" pixels (true error > 10 cm) by predicted error > 10 cm
    (precision, recall, and flagged % vs truly unreliable %),
  * true and predicted error binned by distance to the nearest measurement.
Image-level warning rule (for the app): warn when the image's mean predicted error
exceeds T. "Unusable" image = hybrid RMSE > 0.15 m. T is chosen on VALIDATION to
maximise balanced accuracy; its precision / recall are then reported on TEST.
The confidence model itself is NOT refitted (it was fitted on validation at 2x-8x and
5 % sparse only, so densities below 5 % are an extrapolation test).

Outputs: results/sparse_confidence.json, results/figures/new/sparse_confidence.png
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgedepth.classical.holes import hole_distance  # noqa: E402
from edgedepth.classical.interpolation import upsample_mask  # noqa: E402
from edgedepth.confidence import ause, predict_error  # noqa: E402
from edgedepth.degradation import DegradationConfig, degrade  # noqa: E402
from edgedepth.hybrid.fusion import hybrid_reconstruct  # noqa: E402
from edgedepth.model.dataset import load_split  # noqa: E402

DENS = (0.10, 0.05, 0.02, 0.01, 0.005, 0.002)
SEED = {"test": 40000, "val": 50000}
BAD_PX = 0.10       # m: an "unreliable" pixel
BAD_IMG = 0.15      # m: an "unusable" image (hybrid RMSE)
DIST_BINS = [0, 2, 4, 8, 16, 32, 1e9]
FR = np.linspace(0, 0.9, 19)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def rmse(a, b, v):
    return float(np.sqrt(np.mean((a[v] - b[v]) ** 2)))


def run_split(split):
    rows, dist = [], {d: {"true": [[] for _ in DIST_BINS[:-1]], "pred": [[] for _ in DIST_BINS[:-1]]} for d in DENS}
    t0 = time.time()
    data = load_split("data/nyu", split)
    for i, (name, rgb, gt) in enumerate(data):
        for d in DENS:
            obs = degrade(gt, DegradationConfig(mode="sparse", sparse_density=d, seed=SEED[split] + i))
            out, det = hybrid_reconstruct(rgb, obs, use_dl=True, return_details=True)
            pe = predict_error(obs, out, det)
            v = gt > 0
            err_map = np.abs(out - gt)
            err, pred = err_map[v], pe["error_m"][v]
            a_l, _, _ = ause(err, pred, FR)
            a_r, _, _ = ause(err, np.random.default_rng(i).random(err.shape), FR)
            bad, flag = err > BAD_PX, pred > BAD_PX
            c = det["candidates"]
            rows.append({"frame": name, "density": d, "n_samples": int(obs.mask.sum()),
                         "rmse_hybrid": rmse(out, gt, v), "rmse_dl": rmse(c["deep_learning"], gt, v),
                         "rmse_bicubic": rmse(c["bicubic"], gt, v),
                         "mean_true_err": float(err.mean()), "mean_pred_err": float(pred.mean()),
                         "calibration_ratio": float(pred.mean() / max(err.mean(), 1e-9)),
                         "ause_learned": a_l, "ause_random": a_r, "spearman": spearman(pred, err),
                         "bad_px_pct": float(bad.mean() * 100), "flagged_px_pct": float(flag.mean() * 100),
                         "flag_precision": float((bad & flag).sum() / max(flag.sum(), 1)),
                         "flag_recall": float((bad & flag).sum() / max(bad.sum(), 1)),
                         "mean_confidence": float(pe["confidence"][v].mean())})
            if split == "test":
                hd = hole_distance(upsample_mask(obs))[v]
                for b in range(len(DIST_BINS) - 1):
                    s = (hd >= DIST_BINS[b]) & (hd < DIST_BINS[b + 1])
                    if s.any():
                        dist[d]["true"][b].append((float(err[s].sum()), int(s.sum())))
                        dist[d]["pred"][b].append((float(pred[s].sum()), int(s.sum())))
        print(f"{split} {i + 1}/{len(data)} {name} {time.time() - t0:.0f}s", flush=True)
    return rows, dist


def summarise(rows, keys):
    out = {}
    for d in DENS:
        rs = [r for r in rows if r["density"] == d]
        out[f"{d * 100:g}%"] = {k: float(np.mean([r[k] for r in rs])) for k in keys}
        out[f"{d * 100:g}%"]["unusable_images_pct"] = float(np.mean([r["rmse_hybrid"] > BAD_IMG for r in rs]) * 100)
        out[f"{d * 100:g}%"]["n_images"] = len(rs)
    return out


def choose_threshold(rows):
    y = np.array([r["rmse_hybrid"] > BAD_IMG for r in rows])
    s = np.array([r["mean_pred_err"] for r in rows])
    best = None
    for t in np.unique(np.round(s, 4)):
        p = s > t
        tpr = (p & y).sum() / max(y.sum(), 1)
        tnr = (~p & ~y).sum() / max((~y).sum(), 1)
        ba = (tpr + tnr) / 2
        if best is None or ba > best[1]:
            best = (float(t), float(ba))
    return best


def rule_stats(rows, t):
    y = np.array([r["rmse_hybrid"] > BAD_IMG for r in rows])
    p = np.array([r["mean_pred_err"] > t for r in rows])
    return {"threshold_m": t, "n_images": int(len(y)), "n_unusable": int(y.sum()), "n_warned": int(p.sum()),
            "precision": float((p & y).sum() / max(p.sum(), 1)), "recall": float((p & y).sum() / max(y.sum(), 1)),
            "false_alarm_rate": float((p & ~y).sum() / max((~y).sum(), 1)),
            "warned_by_density": {f"{d * 100:g}%": float(np.mean([r["mean_pred_err"] > t for r in rows if r["density"] == d]) * 100)
                                  for d in DENS}}


def main():
    keys = ["n_samples", "rmse_hybrid", "rmse_dl", "rmse_bicubic", "mean_true_err", "mean_pred_err", "calibration_ratio",
            "ause_learned", "ause_random", "spearman", "bad_px_pct", "flagged_px_pct", "flag_precision", "flag_recall",
            "mean_confidence"]
    val_rows, _ = run_split("val")
    test_rows, dist = run_split("test")
    t, ba_val = choose_threshold(val_rows)
    res = {
        "protocol": __doc__.strip().split("\n\n", 2)[1:],
        "definitions": {"unreliable_pixel_m": BAD_PX, "unusable_image_rmse_m": BAD_IMG},
        "test": summarise(test_rows, keys),
        "val": summarise(val_rows, keys),
        "warning_rule": {"chosen_on": "validation", "balanced_accuracy_val": ba_val,
                         "val": rule_stats(val_rows, t), "test": rule_stats(test_rows, t)},
        "error_vs_distance_test": {
            "bins_px": [f"{DIST_BINS[b]:g}-{DIST_BINS[b + 1]:g}" if DIST_BINS[b + 1] < 1e8 else f">={DIST_BINS[b]:g}"
                        for b in range(len(DIST_BINS) - 1)],
            **{f"{d * 100:g}%": {k: [sum(x for x, _ in L) / max(sum(n for _, n in L), 1) if L else None for L in dist[d][k]]
                                 for k in ("true", "pred")} for d in DENS},
            "pixel_share": {f"{d * 100:g}%": [sum(n for _, n in L) for L in dist[d]["true"]] for d in DENS}},
    }
    with open("results/sparse_confidence.json", "w") as f:
        json.dump(res, f, indent=1)
    for k, v in res["test"].items():
        print(k, {a: round(b, 4) for a, b in v.items()})
    print(json.dumps(res["warning_rule"], indent=1))
    make_figure(res)


def make_figure(res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = res["test"]
    labs = list(T)
    x = np.arange(len(labs))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot(x, [T[k]["rmse_bicubic"] for k in labs], "o-", color="#9aa0a6", label="Bicubic")
    ax[0].plot(x, [T[k]["rmse_dl"] for k in labs], "o-", color="#5b8def", label="Network only")
    ax[0].plot(x, [T[k]["rmse_hybrid"] for k in labs], "o-", color="#1f9d55", lw=2.5, label="Hybrid")
    ax[0].axhline(BAD_IMG, color="#d93025", ls=":", lw=1)
    ax[0].text(0, BAD_IMG + 0.005, "unusable (0.15 m)", color="#d93025", fontsize=8)
    ax[0].set_title("Test RMSE vs density (24 frames)")
    ax[0].set_ylabel("RMSE (m)")
    ax[0].legend(frameon=False, fontsize=9)
    ax[1].plot(x, [T[k]["mean_true_err"] for k in labs], "o-", color="#202124", label="True mean error")
    ax[1].plot(x, [T[k]["mean_pred_err"] for k in labs], "o--", color="#1f9d55", label="Predicted by confidence map")
    ax[1].set_title("Does the confidence map track it?")
    ax[1].set_ylabel("Mean absolute error (m)")
    ax[1].legend(frameon=False, fontsize=9)
    E = res["error_vs_distance_test"]
    bx = np.arange(len(E["bins_px"]))
    for d, col in (("5%", "#5b8def"), ("1%", "#f2a33a"), ("0.2%", "#d93025")):
        tr = [np.nan if v is None else v for v in E[d]["true"]]
        pr = [np.nan if v is None else v for v in E[d]["pred"]]
        ax[2].plot(bx, tr, "o-", color=col, label=f"{d} true")
        ax[2].plot(bx, pr, "o--", color=col, alpha=0.7, label=f"{d} predicted")
    ax[2].set_xticks(bx, E["bins_px"])
    ax[2].set_xlabel("Distance to nearest measurement (px)")
    ax[2].set_title("Error grows with distance from data")
    ax[2].legend(frameon=False, fontsize=8, ncol=2)
    for a in ax[:2]:
        a.set_xticks(x, labs)
        a.set_xlabel("Measurement density")
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    os.makedirs("results/figures/new", exist_ok=True)
    fig.savefig("results/figures/new/sparse_confidence.png", dpi=130)


if __name__ == "__main__":
    main()
