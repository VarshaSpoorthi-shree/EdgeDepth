"""Tune the misalignment-correction acceptance threshold on the VALIDATION split.

    python scripts/tune_alignment.py

Cases: every 2nd validation frame x {2x, 4x, 8x low-res, 5% sparse} x RGB shifts
{(0,0), (0,2), (0,4), (0,-4), (0,8), (4,0), (-8,0), (4,4), (-6,6)} (dy, dx).
For each case the shift is estimated once; the hybrid is run on the original
RGB and on the corrected RGB. The threshold on the NCC gain decides which
result is used. We choose the threshold with the best mean boundary RMSE
subject to aligned (0,0) inputs never being changed.
Writes results/alignment_tuning.json.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgedepth.alignment import estimate_shift, shift_image  # noqa: E402
from edgedepth.degradation import DegradationConfig, degrade  # noqa: E402
from edgedepth.metrics import boundary_rmse, edge_f1, rmse  # noqa: E402
from edgedepth.model.dataset import load_split  # noqa: E402
from edgedepth.pipeline import run_method  # noqa: E402

SHIFTS = [(0, 0), (0, 2), (0, 4), (0, -4), (0, 8), (4, 0), (-8, 0), (4, 4), (-6, 6)]
DEAD_ZONE = 1


def main():
    val = load_split("data/nyu", "val")[::2]
    rows = []
    for i, (name, rgb, gt) in enumerate(val):
        cfgs = [(f"{s}x", DegradationConfig(scale=s, seed=1000 + i)) for s in (2, 4, 8)]
        cfgs.append(("sparse", DegradationConfig(mode="sparse", sparse_density=0.05, seed=3000 + i)))
        for sc, cfg in cfgs:
            obs = degrade(gt, cfg)
            base = run_method("hybrid", rgb, obs)  # aligned reference (best achievable)
            for sh in SHIFTS:
                r = shift_image(rgb, *sh)
                est = estimate_shift(r, obs)
                d_no = run_method("hybrid", r, obs) if sh != (0, 0) else base
                movable = max(abs(est["dy"]), abs(est["dx"])) > DEAD_ZONE
                d_yes = run_method("hybrid", shift_image(r, est["dy"], est["dx"]), obs) if movable else d_no
                rows.append({"frame": name, "scale": sc, "shift": list(sh), "est": [est["dy"], est["dx"]], "gain": est["gain"],
                             "movable": movable,
                             "residual_px": float(np.hypot(sh[0] + est["dy"], sh[1] + est["dx"])),
                             "no": [rmse(d_no, gt), boundary_rmse(d_no, gt), edge_f1(d_no, gt)],
                             "yes": [rmse(d_yes, gt), boundary_rmse(d_yes, gt), edge_f1(d_yes, gt)],
                             "aligned": [rmse(base, gt), boundary_rmse(base, gt), edge_f1(base, gt)]})
        print(f"{i + 1}/{len(val)}", flush=True)
    out = {"dead_zone_px": DEAD_ZONE, "thresholds": []}
    for t in (0.0, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08):
        res = []
        for r in rows:
            use = r["movable"] and r["gain"] > t
            res.append(r["yes"] if use else r["no"])
        res = np.array(res)
        aligned_changed = sum(1 for r in rows if r["shift"] == [0, 0] and r["movable"] and r["gain"] > t)
        shifted = [k for k, r in enumerate(rows) if r["shift"] != [0, 0]]
        aligned = [k for k, r in enumerate(rows) if r["shift"] == [0, 0]]
        out["thresholds"].append({"threshold": t, "mean_b_rmse_shifted": float(res[shifted, 1].mean()),
                                  "mean_b_rmse_aligned": float(res[aligned, 1].mean()),
                                  "aligned_cases_changed": aligned_changed, "aligned_cases": len(aligned),
                                  "corrected_shifted_cases": int(sum(1 for k in shifted if rows[k]["movable"] and rows[k]["gain"] > t))})
        print(out["thresholds"][-1])
    no = np.array([r["no"] for r in rows])
    al = np.array([r["aligned"] for r in rows])
    sh = [k for k, r in enumerate(rows) if r["shift"] != [0, 0]]
    out["reference"] = {"b_rmse_shifted_uncorrected": float(no[sh, 1].mean()), "b_rmse_if_perfectly_aligned": float(al[sh, 1].mean())}
    out["estimate_accuracy_within_1px_shifted"] = float(np.mean([rows[k]["residual_px"] <= 1.5 for k in sh]))
    out["by_scale_accuracy"] = {sc: float(np.mean([rows[k]["residual_px"] <= 1.5 for k in sh if rows[k]["scale"] == sc])) for sc in ("2x", "4x", "8x", "sparse")}
    with open("results/alignment_tuning.json", "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "thresholds"}, indent=1))


if __name__ == "__main__":
    main()
