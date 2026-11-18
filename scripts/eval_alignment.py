"""Before/after evaluation of automatic misalignment correction on the unseen TEST split.

    python scripts/eval_alignment.py

24 test frames x {2x, 4x, 8x low-res, 5% sparse} x RGB shifts (dy, dx) in
{(0,0), (0,2), (0,4), (0,8), (4,0), (-6,6)}. For each case the hybrid runs on
(a) the shifted RGB without correction, (b) after auto_align, and (c) the
correctly aligned RGB (reference = best achievable).
Writes results/alignment_eval.json and results/figures/new/alignment_before_after.png.
"""
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgedepth.alignment import auto_align, shift_image  # noqa: E402
from edgedepth.degradation import DegradationConfig, degrade  # noqa: E402
from edgedepth.metrics import boundary_rmse, edge_f1, rmse  # noqa: E402
from edgedepth.model.dataset import load_split  # noqa: E402
from edgedepth.pipeline import run_method  # noqa: E402

SHIFTS = [(0, 0), (0, 2), (0, 4), (0, 8), (4, 0), (-6, 6)]


def met(d, gt):
    return [rmse(d, gt), boundary_rmse(d, gt), edge_f1(d, gt)]


def main():
    test = load_split("data/nyu", "test")
    rows = []
    t0 = time.time()
    for i, (name, rgb, gt) in enumerate(test):
        cfgs = [(f"{s}x", DegradationConfig(scale=s, seed=20000 + i)) for s in (2, 4, 8)]
        cfgs.append(("sparse", DegradationConfig(mode="sparse", sparse_density=0.05, seed=30000 + i)))
        for sc, cfg in cfgs:
            obs = degrade(gt, cfg)
            ref = met(run_method("hybrid", rgb, obs), gt)
            for sh in SHIFTS:
                r = shift_image(rgb, *sh)
                before = ref if sh == (0, 0) else met(run_method("hybrid", r, obs), gt)
                fixed, est = auto_align(r, obs)
                after = before if not est["applied"] else met(run_method("hybrid", fixed, obs), gt)
                rows.append({"frame": name, "scale": sc, "shift": list(sh), "estimate": [est["dy"], est["dx"]],
                             "applied": est["applied"], "gain": est["gain"],
                             "residual_px": float(np.hypot(sh[0] + est["dy_applied"], sh[1] + est["dx_applied"])),
                             "before": before, "after": after, "aligned": ref})
        print(f"{i + 1}/{len(test)} {time.time() - t0:.0f}s", flush=True)

    def summ(sel):
        b = np.array([r["before"] for r in sel])
        a = np.array([r["after"] for r in sel])
        al = np.array([r["aligned"] for r in sel])
        lost = b[:, 1].mean() - al[:, 1].mean()
        return {"n": len(sel), "rmse_before": b[:, 0].mean(), "rmse_after": a[:, 0].mean(), "rmse_aligned": al[:, 0].mean(),
                "b_rmse_before": b[:, 1].mean(), "b_rmse_after": a[:, 1].mean(), "b_rmse_aligned": al[:, 1].mean(),
                "f1_before": b[:, 2].mean(), "f1_after": a[:, 2].mean(), "f1_aligned": al[:, 2].mean(),
                "pct_of_boundary_loss_recovered": float(100 * (b[:, 1].mean() - a[:, 1].mean()) / lost) if lost > 1e-9 else None,
                "corrected_pct": 100 * float(np.mean([r["applied"] for r in sel])),
                "residual_within_1_5px_pct": 100 * float(np.mean([r["residual_px"] <= 1.5 for r in sel])),
                "residual_within_2_5px_pct": 100 * float(np.mean([r["residual_px"] <= 2.5 for r in sel]))}

    shifted = [r for r in rows if r["shift"] != [0, 0]]
    aligned = [r for r in rows if r["shift"] == [0, 0]]
    res = {"protocol": "NYU test split, 24 frames x {2x,4x,8x,5% sparse} x RGB shifts " + str(SHIFTS),
           "shifted_all": summ(shifted),
           "aligned_inputs": {"n": len(aligned), "changed": int(sum(r["applied"] for r in aligned)),
                              "b_rmse_before": float(np.mean([r["before"][1] for r in aligned])),
                              "b_rmse_after": float(np.mean([r["after"][1] for r in aligned]))},
           "by_scale": {sc: summ([r for r in shifted if r["scale"] == sc]) for sc in ("2x", "4x", "8x", "sparse")},
           "by_shift": {str(tuple(sh)): summ([r for r in shifted if r["shift"] == list(sh)]) for sh in SHIFTS[1:]}}
    res = json.loads(json.dumps(res, default=float))
    with open("results/alignment_eval.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: res[k] for k in ("shifted_all", "aligned_inputs")}, indent=1))
    for sc, v in res["by_scale"].items():
        print(sc, {k: round(x, 4) if isinstance(x, float) else x for k, x in v.items()})
    for sh, v in res["by_shift"].items():
        print(sh, {k: round(x, 4) if isinstance(x, float) else x for k, x in v.items()})
    plot(res)


def plot(res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs("results/figures/new", exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
    for k, (key, title) in enumerate((("b_rmse", "Boundary RMSE (m) ↓"), ("f1", "Edge F1 ↑"))):
        labels = list(res["by_scale"])
        x = np.arange(len(labels))
        for j, (state, col, lab) in enumerate((("before", "tab:red", "misaligned, no correction"), ("after", "tab:green", "with auto-alignment"),
                                                 ("aligned", "tab:gray", "perfectly aligned (reference)"))):
            ax[k].bar(x + (j - 1) * 0.27, [res["by_scale"][s][f"{key}_{state}"] for s in labels], 0.27, color=col, label=lab)
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(labels)
        ax[k].set_title(title)
        ax[k].grid(axis="y", alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle("Automatic RGB–depth misalignment correction (hybrid, unseen test split, shifted RGB inputs)", fontsize=10)
    fig.tight_layout()
    fig.savefig("results/figures/new/alignment_before_after.png", dpi=120, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
