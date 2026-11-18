"""Controlled latency / memory benchmark: repeated runs under identical conditions.

    python scripts/benchmark_latency.py

For each method and input size: 2 warm-up runs, then N timed runs on the same
observation (latency without memory tracking), then one separate run for
peak memory (tracemalloc for NumPy allocations + 2 ms RSS sampling for PyTorch).
Writes results/latency_memory.json.
"""
import json
import os
import platform
import sys
import time

import numpy as np
import psutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edgedepth.degradation import DegradationConfig, degrade, observation_from_sensor  # noqa: E402
from edgedepth.io_utils import load_depth, load_rgb  # noqa: E402
from edgedepth.metrics import measure  # noqa: E402
from edgedepth.model.inference import load_model  # noqa: E402
from edgedepth.pipeline import run_method  # noqa: E402

METHODS = ["nearest", "bilinear", "bicubic", "joint_bilateral", "guided_filter", "deep_learning", "hybrid"]
N = 10


def bench(rgb, obs, label):
    out = {}
    for m in METHODS:
        for _ in range(2):
            run_method(m, rgb, obs)
        ts = []
        for _ in range(N):
            t = time.perf_counter()
            run_method(m, rgb, obs)
            ts.append((time.perf_counter() - t) * 1000)
        _, _, mb = measure(run_method, m, rgb, obs, track_memory=True)
        out[m] = {"mean_ms": float(np.mean(ts)), "std_ms": float(np.std(ts)), "min_ms": float(np.min(ts)), "runs": N,
                  "peak_extra_ram_mb": float(mb)}
        print(f"{label:28s} {m:16s} {np.mean(ts):8.1f} ± {np.std(ts):6.1f} ms   {mb:7.1f} MB", flush=True)
    return out


def main():
    proc = psutil.Process()
    rss0 = proc.memory_info().rss / 2 ** 20
    load_model()
    rss_model = proc.memory_info().rss / 2 ** 20
    res = {"machine": {"cpu_cores": os.cpu_count(), "ram_gb": round(psutil.virtual_memory().total / 2 ** 30, 1),
                       "platform": platform.platform(), "python": platform.python_version(), "gpu": "none"},
           "process_rss_mb_before_model": rss0, "process_rss_mb_after_model": rss_model}
    rgb, gt = load_rgb("data/nyu/test/study_9_0003_rgb.png"), load_depth("data/nyu/test/study_9_0003_depth.png")
    res["nyu_320x240_4x_noisy"] = bench(rgb, degrade(gt, DegradationConfig(scale=4, noise_level=1, missing_rate=0.05, seed=1)), "NYU 320x240 4x noisy")
    res["nyu_320x240_sparse5"] = bench(rgb, degrade(gt, DegradationConfig(mode="sparse", sparse_density=0.05, seed=1)), "NYU 320x240 sparse 5%")
    rgb2, d2 = load_rgb("data/samples/tum_fr1_rgb.png"), load_depth("data/samples/tum_fr1_depth.png")
    res["tum_640x480_4x"] = bench(rgb2, observation_from_sensor(d2, 4), "TUM 640x480 4x")
    res["process_peak_rss_mb"] = proc.memory_info().rss / 2 ** 20
    with open("results/latency_memory.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if not isinstance(v, dict) or k == "machine"}, indent=1))


if __name__ == "__main__":
    main()
