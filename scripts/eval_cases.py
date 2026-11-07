"""Per-input comparison figures for the evaluation file: baseline vs proposed.

    python scripts/eval_cases.py
Writes results/figures/eval/case*.png and results/eval_cases.json.
Baseline = bicubic interpolation; also shown: deep learning only; proposed = hybrid.
"""
import json
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgedepth.degradation import DegradationConfig, degrade, observation_from_sensor  # noqa: E402
from edgedepth.io_utils import colorize, colorize_error, depth_range, load_depth, load_rgb  # noqa: E402
from edgedepth.metrics import all_metrics  # noqa: E402
from edgedepth.pipeline import run_method  # noqa: E402

OUT = "results/figures/eval"
NOISY = dict(noise_level=1.0, missing_rate=0.05, edge_dropout=0.5)
CASES = [
    ("case1_nyu_8x_noisy", "data/nyu/test/study_9_0003", "NYU test study_9_0003 - 8x low-res, Kinect noise x1, 5% dropout, 50% edge dropout (seed 20001)",
     lambda gt: degrade(gt, DegradationConfig(scale=8, seed=20001, **NOISY))),
    ("case2_nyu_4x_clean", "data/nyu/test/basement_3_0004", "NYU test basement_3_0004 - 4x low-res, clean (seed 7)",
     lambda gt: degrade(gt, DegradationConfig(scale=4, seed=7))),
    ("case3_nyu_sparse5", "data/nyu/test/basement_3_0006", "NYU test basement_3_0006 - 5% random sparse samples (seed 31)",
     lambda gt: degrade(gt, DegradationConfig(mode="sparse", sparse_density=0.05, seed=31))),
    ("case4_tum_4x_sensor", "data/samples/tum_fr1", "TUM RGB-D fr1 (unseen sensor, 640x480, 19% raw holes) - 4x low-res from raw sensor depth",
     lambda gt: observation_from_sensor(gt, 4)),
]
METHODS = [("bicubic", "Baseline: bicubic"), ("deep_learning", "Deep learning only"), ("hybrid", "Proposed: hybrid")]


def main():
    os.makedirs(OUT, exist_ok=True)
    report = {}
    for cid, stem, desc, make in CASES:
        rgb, gt = load_rgb(stem + "_rgb.png"), load_depth(stem + "_depth.png")
        obs = make(gt)
        lo, hi = depth_range(gt)
        outs = {m: run_method(m, rgb, obs) for m, _ in METHODS}
        mets = {m: all_metrics(d, gt) for m, d in outs.items()}
        report[cid] = {"input": desc, "valid_input_pct": float(obs.mask.mean() * 100), "metrics": mets}
        H, W = gt.shape
        inp = colorize(obs.values, lo, hi)
        inp = cv2.resize(inp, (W, H), interpolation=cv2.INTER_NEAREST) if obs.values.shape != gt.shape else cv2.dilate(inp, np.ones((2, 2), np.uint8))
        fig, ax = plt.subplots(2, 4, figsize=(13, 5.6))
        ax[0, 0].imshow(rgb)
        ax[0, 0].set_title("RGB guide")
        ax[1, 0].imshow(inp)
        ax[1, 0].set_title("Input depth (nearest view)")
        for j, (m, lab) in enumerate(METHODS):
            k = mets[m]
            ax[0, j + 1].imshow(colorize(outs[m], lo, hi))
            ax[0, j + 1].set_title(lab, fontweight="bold" if m == "hybrid" else None)
            ax[1, j + 1].imshow(colorize_error(np.abs(outs[m] - gt) * (gt > 0), 0.3))
            ax[1, j + 1].set_title(f"|error|  RMSE {k['rmse']:.3f}  B-RMSE {k['b_rmse']:.3f}  F1 {k['edge_f1']:.2f}", fontsize=8)
        for a in ax.ravel():
            a.set_xticks([])
            a.set_yticks([])
        fig.suptitle(desc, fontsize=10)
        fig.tight_layout(h_pad=2.2)
        fig.savefig(f"{OUT}/{cid}.png", dpi=110, facecolor="white")
        plt.close(fig)
        print(cid, {m: (round(v["rmse"], 4), round(v["b_rmse"], 4), round(v["edge_f1"], 3), round(v["ssim"], 4)) for m, v in mets.items()})
    with open("results/eval_cases.json", "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
