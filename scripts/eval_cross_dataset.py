"""Cross-dataset generalisation benchmark (no training, no tuning on these frames).

    python scripts/fetch_cross_dataset.py   # once: downloads 133 frames to data/cross
    python scripts/eval_cross_dataset.py    # took 31 min on 2 CPU cores (sharing them with another run)

Two groups, every method, identical inputs per method:

A. Synthetic with exact ground truth - Augmented ICL-NUIM living room 1 / office 1
   (25 frames each). Same protocol and seeds as the NYU test (scripts/evaluate.py):
   2x/4x/8x x {clean, noisy_missing} + 5 % sparse, so numbers are directly comparable
   with results/summary.json. Plus "sim_sensor": the *simulated sensor* depth (real
   noise model and holes) box-downsampled 4x / 8x, scored against the clean GT.

B. Real sensors, raw reference - Stanford Lounge (25), Redwood Bedroom (25),
   Stanford Fountain (33). The reference is the raw sensor depth: holes are excluded
   from every metric, and its own noise is counted as error, so absolute numbers are
   pessimistic; the comparison between methods is what matters.
   Conditions: 2x/4x/8x (masked box-downsampling of the raw depth) + 5 % sparse.

For the hybrid also: confidence-map AUSE (vs random ordering), Spearman correlation
and calibration ratio (mean predicted error / mean true error; 1 = calibrated).
Per source: paired per-frame comparison of the hybrid with bicubic and with the
network alone (win rate + bootstrap 95 % CI of the relative RMSE change).

Outputs: results/cross_dataset_eval.json, results/cross_dataset_rows.csv,
         results/figures/new/cross_dataset.png
"""
import csv
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgedepth import METHODS  # noqa: E402
from edgedepth.confidence import ause, predict_error  # noqa: E402
from edgedepth.degradation import DegradationConfig, degrade, observation_from_sensor  # noqa: E402
from edgedepth.hybrid.fusion import hybrid_reconstruct  # noqa: E402
from edgedepth.io_utils import load_depth, load_rgb  # noqa: E402
from edgedepth.metrics import METRIC_KEYS, all_metrics  # noqa: E402
from edgedepth.pipeline import run_method  # noqa: E402

ROOT = "data/cross"
SYNTH = ("icl_living", "icl_office")
REAL = ("lounge", "bedroom", "fountain")
NOISY = dict(noise_level=1.0, missing_rate=0.05, edge_dropout=0.5)
FR = np.linspace(0, 0.9, 19)


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def frames(src):
    out = []
    for p in sorted(glob.glob(os.path.join(ROOT, src, "*_rgb.png"))):
        fid = os.path.basename(p)[:-8]
        sim = p.replace("_rgb.png", "_simdepth.png")
        out.append((fid, load_rgb(p), load_depth(p.replace("_rgb.png", "_depth.png")),
                    load_depth(sim) if os.path.exists(sim) else None))
    return out


def cases(src, i, gt, sim):
    if src in SYNTH:  # identical to scripts/evaluate.py
        for s in (2, 4, 8):
            yield f"{s}x", "clean", degrade(gt, DegradationConfig(scale=s, seed=20000 + i))
            yield f"{s}x", "noisy_missing", degrade(gt, DegradationConfig(scale=s, seed=20000 + i, **NOISY))
        yield "sparse", "sparse_5pct", degrade(gt, DegradationConfig(mode="sparse", sparse_density=0.05, seed=30000 + i))
        for s in (4, 8):
            yield f"{s}x", "sim_sensor", observation_from_sensor(sim, s)
    else:
        for s in (2, 4, 8):
            yield f"{s}x", "raw_sensor", observation_from_sensor(gt, s)
        yield "sparse", "sparse_5pct", degrade(gt, DegradationConfig(mode="sparse", sparse_density=0.05, seed=30000 + i))


def boot_ci(x, n=2000, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, (n, len(x)), replace=True).mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def mean_of(rows, key, **flt):
    v = [r[key] for r in rows if all(r[k] == val for k, val in flt.items()) and r.get(key) is not None and np.isfinite(r[key])]
    return float(np.mean(v)) if v else None


def paired(rows, src, a, b, key="rmse", **flt):
    """Per frame (averaged over conditions): relative change of `a` vs `b` (negative = a better)."""
    per = {}
    for r in rows:
        if r["source"] == src and r["method"] in (a, b) and all(r[k] == v for k, v in flt.items()):
            per.setdefault(r["frame"], {}).setdefault(r["method"], []).append(r[key] if r[key] is not None else np.nan)
    rel = [np.nanmean(d[a]) / np.nanmean(d[b]) - 1 for d in per.values()
           if a in d and b in d and np.isfinite(d[a]).any() and np.isfinite(d[b]).any()]
    if not rel:
        return None
    return {"n_frames": len(rel), "mean_rel_change_pct": float(np.mean(rel) * 100),
            "ci95_pct": [c * 100 for c in boot_ci(rel)], "win_rate_pct": float(np.mean(np.array(rel) < 0) * 100)}


def main():
    rows, conf_rows = [], []
    curves = {}
    t0 = time.time()
    for src in SYNTH + REAL:
        fr = frames(src)
        for i, (fid, rgb, gt, sim) in enumerate(fr):
            for sc, cond, obs in cases(src, i, gt, sim):
                out, det = hybrid_reconstruct(rgb, obs, use_dl=True, return_details=True)
                for m in METHODS:
                    d = out if m == "hybrid" else run_method(m, rgb, obs)
                    rows.append({"source": src, "group": "synthetic_gt" if src in SYNTH else "real_sensor",
                                 "frame": fid, "scale": sc, "condition": cond, "method": m, **all_metrics(d, gt)})
                v = gt > 0
                err = np.abs(out - gt)[v]
                pe = predict_error(obs, out, det)
                pred = pe["error_m"][v]
                a_l, c_l, c_o = ause(err, pred, FR)
                a_r, _, _ = ause(err, np.random.default_rng(i).random(err.shape), FR)
                curves.setdefault(src, []).append(c_l)
                conf_rows.append({"source": src, "frame": fid, "scale": sc, "condition": cond, "ause_learned": a_l,
                                  "ause_random": a_r, "spearman": spearman(pred, err),
                                  "calibration_ratio": float(pred.mean() / max(err.mean(), 1e-9))})
            print(f"{src} {i + 1}/{len(fr)} {time.time() - t0:.0f}s", flush=True)

    with open("results/cross_dataset_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    res = {"protocol": __doc__.strip().split("\n\n", 2)[1:],
           "frames": {s: len(frames(s)) for s in SYNTH + REAL},
           "by_source": {}, "synthetic_by_scale_condition": {}, "confidence": {}, "paired": {}}
    for src in SYNTH + REAL:
        res["by_source"][src] = {m: {k: mean_of(rows, k, source=src, method=m) for k in METRIC_KEYS} for m in METHODS}
        res["by_source"][src]["_by_scale"] = {
            sc: {m: {k: mean_of(rows, k, source=src, method=m, scale=sc) for k in ("rmse", "b_rmse", "edge_f1")} for m in METHODS}
            for sc in ("2x", "4x", "8x", "sparse")}
        res["confidence"][src] = {k: mean_of(conf_rows, k, source=src) for k in ("ause_learned", "ause_random", "spearman", "calibration_ratio")}
        res["confidence"][src]["mean_curve"] = np.mean(curves[src], 0).tolist()
        res["paired"][src] = {"hybrid_vs_bicubic": paired(rows, src, "hybrid", "bicubic"),
                              "hybrid_vs_deep_learning": paired(rows, src, "hybrid", "deep_learning"),
                              "hybrid_vs_deep_learning_b_rmse": paired(rows, src, "hybrid", "deep_learning", key="b_rmse")}
    syn = [r for r in rows if r["group"] == "synthetic_gt"]
    for sc, cond in sorted({(r["scale"], r["condition"]) for r in syn}):
        res["synthetic_by_scale_condition"][f"{sc}/{cond}"] = {
            m: {k: mean_of(syn, k, scale=sc, condition=cond, method=m) for k in ("rmse", "b_rmse", "edge_f1")} for m in METHODS}
    for grp in ("synthetic_gt", "real_sensor"):
        res[f"overall_{grp}"] = {m: {k: mean_of(rows, k, group=grp, method=m) for k in METRIC_KEYS} for m in METHODS}
    res["confidence_by_density_condition"] = {
        f"{g}": {k: mean_of(conf_rows, k, condition=g) for k in ("ause_learned", "ause_random", "spearman", "calibration_ratio")}
        for g in sorted({r["condition"] for r in conf_rows})}
    res["fraction_axis"] = FR.tolist()
    res["elapsed_s"] = time.time() - t0
    with open("results/cross_dataset_eval.json", "w") as f:
        json.dump(res, f, indent=1)
    for src in SYNTH + REAL:
        b = res["by_source"][src]
        print(src, {m: round(b[m]["rmse"], 4) for m in METHODS}, res["paired"][src]["hybrid_vs_bicubic"],
              res["paired"][src]["hybrid_vs_deep_learning"], res["confidence"][src]["ause_learned"])
    make_figure(res)


def make_figure(res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = {"icl_living": "ICL living\n(synthetic GT)", "icl_office": "ICL office\n(synthetic GT)",
             "lounge": "Lounge\n(real)", "bedroom": "Bedroom\n(real)", "fountain": "Fountain\n(real)"}
    show = [("bicubic", "Bicubic", "#9aa0a6"), ("guided_filter", "Guided filter", "#f2a33a"),
            ("deep_learning", "Network only", "#5b8def"), ("hybrid", "Hybrid", "#1f9d55")]
    srcs = list(names)
    x = np.arange(len(srcs))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
    for k, (metric, title) in enumerate((("rmse", "RMSE (m), mean over conditions"), ("b_rmse", "Boundary RMSE (m)"))):
        for j, (m, lab, col) in enumerate(show):
            ax[k].bar(x + (j - 1.5) * 0.2, [res["by_source"][s][m][metric] for s in srcs], 0.2, label=lab, color=col)
        ax[k].set_xticks(x, [names[s] for s in srcs])
        ax[k].set_title(title)
        ax[k].spines[["top", "right"]].set_visible(False)
    ax[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    os.makedirs("results/figures/new", exist_ok=True)
    fig.savefig("results/figures/new/cross_dataset.png", dpi=130)


if __name__ == "__main__":
    main()
