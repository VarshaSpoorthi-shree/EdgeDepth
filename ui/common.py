"""Shared state, data loading and rendering helpers for the Streamlit app."""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from edgedepth import METHOD_LABELS, METHODS
from edgedepth import config as C
from edgedepth.degradation import DegradationConfig, Observation, degrade, observation_from_lowres
from edgedepth.io_utils import InputError, colorize, depth_range, fit_to_multiple, load_depth, load_rgb
from edgedepth.metrics import METRIC_LABELS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "data", "samples")
RESULTS = os.path.join(ROOT, "results")

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.ed-hero {background: linear-gradient(120deg, rgba(79,209,197,0.16), rgba(99,102,241,0.12));
  border: 1px solid rgba(79,209,197,0.25); border-radius: 14px; padding: 1.2rem 1.4rem; margin-bottom: 1rem;}
.ed-hero h1 {margin: 0 0 .3rem 0; font-size: 2.0rem;}
.ed-hero p {margin: 0; color: #A9B4C2;}
.ed-card {background: #151B26; border: 1px solid #243042; border-radius: 12px; padding: .9rem 1rem; height: 100%;}
.ed-card h4 {margin: 0 0 .35rem 0; color: #4FD1C5; font-size: 1rem;}
.ed-card p, .ed-card li {color: #C4CDD8; font-size: .9rem; margin: 0;}
.ed-pill {display:inline-block; padding: .1rem .55rem; border-radius: 999px; background:#1F2A3A; color:#9FB3C8;
  font-size:.75rem; margin-right:.3rem; border:1px solid #2C3A4F}
.ed-muted {color:#8B98A8; font-size:.85rem}
div[data-testid="stMetricValue"] {font-size: 1.35rem;}
@media (max-width: 640px) { .ed-hero h1 {font-size: 1.45rem;} .block-container {padding-left: .7rem; padding-right: .7rem;} }
</style>
"""


@dataclass
class InputBundle:
    rgb: np.ndarray
    obs: Observation
    gt: Optional[np.ndarray]
    label: str
    source: str  # "simulated" | "real_lowres"
    rgb_shift: tuple = (0, 0)  # simulated RGB-depth misalignment (dy, dx)


def hero(title: str, subtitle: str):
    st.markdown(f'<div class="ed-hero"><h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def card(title: str, body: str):
    st.markdown(f'<div class="ed-card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def list_samples() -> Dict[str, dict]:
    meta_path = os.path.join(SAMPLES, "samples.json")
    meta = {m["id"]: m for m in json.load(open(meta_path))} if os.path.exists(meta_path) else {}
    out = {}
    for p in sorted(glob.glob(os.path.join(SAMPLES, "*_rgb.png"))):
        sid = os.path.basename(p)[:-8]
        m = meta.get(sid, {"dataset": "sample"})
        out[sid] = {"rgb": p, "depth": p.replace("_rgb.png", "_depth.png"), **m}
    return out


@st.cache_data(show_spinner=False)
def load_sample(sid: str):
    s = list_samples()[sid]
    rgb = load_rgb(s["rgb"])
    d = load_depth(s["depth"], 1000.0)
    return fit_to_multiple(rgb, d, 8, C.MAX_SIDE)


@st.cache_data(show_spinner=False)
def load_json(name: str):
    p = os.path.join(RESULTS, name)
    return json.load(open(p)) if os.path.exists(p) else None


def sample_label(sid: str) -> str:
    s = list_samples()[sid]
    return f"{sid}  ·  {s.get('dataset', '')}"


def _with_shift(bundle: InputBundle, sh) -> InputBundle:
    if sh == (0, 0):
        return bundle
    from edgedepth.alignment import shift_image

    bundle.rgb = shift_image(bundle.rgb, *sh)
    bundle.rgb_shift = sh
    bundle.label += f" (RGB shifted {sh[1]:+d},{sh[0]:+d})"
    return bundle


def build_simulated(rgb, gt, cfg: DegradationConfig, label) -> InputBundle:
    return InputBundle(rgb, degrade(gt, cfg), gt, label, "simulated")


def build_real_lowres(rgb, lr_depth, label) -> InputBundle:
    rgb, _ = fit_to_multiple(rgb, None, 8, C.MAX_SIDE)
    obs = observation_from_lowres(lr_depth, rgb.shape[:2])
    if obs.scale > 16:
        raise InputError("Depth map is more than 16x smaller than the RGB image; not supported.")
    return InputBundle(rgb, obs, None, label, "real_lowres")


def misalignment_controls(key: str):
    """Optional simulated calibration error: shift the RGB image against the depth."""
    with st.expander("Simulate RGB–depth misalignment (camera calibration error)", expanded=False):
        c1, c2 = st.columns(2)
        dx = c1.slider("Horizontal shift of the RGB image (px)", -10, 10, 0, key=f"{key}_shx")
        dy = c2.slider("Vertical shift of the RGB image (px)", -10, 10, 0, key=f"{key}_shy")
    return int(dy), int(dx)


def align_toggle(key: str) -> bool:
    return st.toggle("Auto-correct RGB–depth misalignment", value=True, key=f"{key}_align",
                     help="Estimates a shift between the colour image and the depth by matching colour edges to depth edges, "
                          "and corrects it before reconstruction. Shifts of 1 px or less, or without a clear correlation peak, "
                          "are left untouched, so aligned inputs are not changed.")


def apply_alignment(bundle: "InputBundle", enabled: bool):
    """Return (rgb to use, estimate dict or None)."""
    if not enabled:
        return bundle.rgb, None
    from edgedepth.alignment import auto_align

    return auto_align(bundle.rgb, bundle.obs)


def alignment_report(est, rgb_shift=(0, 0)):
    """Short text + NCC surface figure for an alignment estimate."""
    import plotly.graph_objects as go

    if est is None:
        return
    R = (est["ncc"].shape[0] - 1) // 2
    if est["applied"]:
        msg = (f"🔧 Misalignment detected and corrected: RGB shifted by **{est['dx']:+d} px horizontally, {est['dy']:+d} px vertically** "
               f"(correlation gain {est['gain']:.3f}).")
    else:
        msg = (f"✅ No significant misalignment detected (best shift {est['dx']:+d}, {est['dy']:+d} px, gain {est['gain']:.3f} "
               f"≤ threshold or within ±1 px) — RGB used unchanged.")
    if rgb_shift != (0, 0):
        msg += f"  *Simulated error: {rgb_shift[1]:+d} px horizontal, {rgb_shift[0]:+d} px vertical → the ideal correction is {-rgb_shift[1]:+d}, {-rgb_shift[0]:+d}.*"
    st.caption(msg)
    with st.expander("Alignment diagnostics — edge correlation for every candidate shift", expanded=False):
        fig = go.Figure(go.Heatmap(z=est["ncc"], x=list(range(-R, R + 1)), y=list(range(-R, R + 1)), colorscale="Viridis",
                                   colorbar=dict(title="NCC")))
        fig.add_trace(go.Scatter(x=[est["dx"]], y=[est["dy"]], mode="markers", marker=dict(symbol="x", size=14, color="red"), name="best"))
        fig.update_layout(height=360, xaxis_title="horizontal shift applied to RGB (px)", yaxis_title="vertical shift (px)",
                          margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
        st.caption("Normalised cross-correlation between depth edges (from an RGB-free interpolation) and colour edges. "
                   "A single sharp peak away from the centre indicates a real misalignment.")


def degradation_controls(key: str, default_scale: int = 4) -> DegradationConfig:
    """Sidebar-free controls for the simulation settings (returns a config)."""
    c1, c2 = st.columns(2)
    with c1:
        mode = st.radio("Input type", ["Low resolution", "Sparse samples"], horizontal=True, key=f"{key}_mode")
        if mode == "Low resolution":
            scale = st.select_slider("Upsampling factor", options=[2, 4, 8], value=default_scale, key=f"{key}_scale")
            density = 0.05
        else:
            scale = 1
            density = st.slider("Sparse density (% of pixels kept)", 0.2, 20.0, 5.0, 0.1, key=f"{key}_dens") / 100
    with c2:
        noise = st.slider("Noise level (× Kinect noise model)", 0.0, 8.0, 0.0, 0.5, key=f"{key}_noise",
                          help="σ(z) = k·(1.2 mm + 1.9 mm·(z−0.4)²) — Nguyen et al. 2012")
        missing = st.slider("Random missing measurements (%)", 0.0, 50.0, 0.0, 1.0, key=f"{key}_miss") / 100
        cc1, cc2 = st.columns(2)
        blobs = cc1.number_input("Large holes", 0, 8, 0, key=f"{key}_blobs")
        edge = cc2.slider("Edge dropout", 0.0, 1.0, 0.0, 0.1, key=f"{key}_edge", help="Probability of losing measurements on depth edges (sensor shadows)")
    seed = st.number_input("Random seed", 0, 99999, 7, key=f"{key}_seed")
    return DegradationConfig(scale=int(scale) if scale > 1 else 4, mode="lowres" if mode == "Low resolution" else "sparse",
                             sparse_density=float(density), noise_level=float(noise), missing_rate=float(missing),
                             hole_blobs=int(blobs), edge_dropout=float(edge), seed=int(seed))


def input_selector(key: str) -> Optional[InputBundle]:
    """Full input UI: bundled sample or upload, simulation or real low-res. Returns an InputBundle."""
    src = st.radio("Input source", ["Bundled sample (unseen test data)", "Upload my own"], horizontal=True, key=f"{key}_src")
    if src.startswith("Bundled"):
        samples = list_samples()
        if not samples:
            st.error("No bundled samples found in data/samples. Run scripts/prepare_samples.py.")
            return None
        sid = st.selectbox("Sample", list(samples), format_func=sample_label, key=f"{key}_sid")
        rgb, gt = load_sample(sid)
        if "raw sensor" in samples[sid].get("gt", ""):
            st.caption("This sensor frame has holes: they are excluded from all reference metrics.")
        cfg = degradation_controls(key)
        sh = misalignment_controls(key)
        return _with_shift(build_simulated(rgb, gt, cfg, sid), sh)

    kind = st.radio("What does the depth file contain?",
                    ["High-res ground-truth depth → simulate degradation (metrics available)",
                     "Real low-resolution / sparse depth (no ground truth → no reference metrics)"], key=f"{key}_kind")
    c1, c2, c3 = st.columns([2, 2, 1])
    up_rgb = c1.file_uploader("RGB guidance image", type=["png", "jpg", "jpeg", "bmp"], key=f"{key}_urgb")
    up_d = c2.file_uploader("Depth map (16-bit PNG, 8-bit PNG or .npy in metres)", type=["png", "npy", "tif", "tiff"], key=f"{key}_ud")
    units = c3.number_input("Depth units per metre", 1.0, 100000.0, 1000.0, help="1000 for millimetres (NYU, Redwood), 5000 for TUM", key=f"{key}_units")
    if up_rgb is None or up_d is None:
        st.info("Upload both an RGB image and a depth map to continue.")
        return None
    try:
        rgb = load_rgb(up_rgb.getvalue())
        depth = load_depth(up_d.getvalue(), units, up_d.name)
    except InputError as e:
        st.error(str(e))
        return None
    if kind.startswith("High-res"):
        if depth.shape != rgb.shape[:2]:
            st.warning(f"RGB {rgb.shape[1]}×{rgb.shape[0]} and depth {depth.shape[1]}×{depth.shape[0]} differ; RGB is resized to the depth size.")
            rgb = cv2.resize(rgb, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_AREA)
        rgb, depth = fit_to_multiple(rgb, depth, 8, C.MAX_SIDE)
        cfg = degradation_controls(key)
        sh = misalignment_controls(key)
        return _with_shift(build_simulated(rgb, depth, cfg, up_rgb.name), sh)
    if depth.shape[0] > rgb.shape[0] or depth.shape[1] > rgb.shape[1]:
        st.error("For real low-resolution input the depth map must not be larger than the RGB image.")
        return None
    try:
        b = build_real_lowres(rgb, depth, up_rgb.name)
    except InputError as e:
        st.error(str(e))
        return None
    st.caption(f"Detected upsampling factor ≈ {b.obs.scale}× (depth {depth.shape[1]}×{depth.shape[0]} → {b.rgb.shape[1]}×{b.rgb.shape[0]}).")
    return b


def show_images(items, cols: int = 4, caption_size: str = "small"):
    """items: list of (title, image). All rendered at the same width."""
    for i in range(0, len(items), cols):
        row = st.columns(cols)
        for c, (t, img) in zip(row, items[i:i + cols]):
            c.image(img, caption=t, width="stretch")


def depth_vis(d, lo, hi):
    return colorize(d, lo, hi)


def lr_vis(obs: Observation, lo, hi):
    H, W = obs.target_hw
    v = colorize(obs.values, lo, hi)
    return cv2.resize(v, (W, H), interpolation=cv2.INTER_NEAREST)


def metrics_frame(results: Dict[str, Dict], with_gt: bool) -> pd.DataFrame:
    rows = []
    for m, r in results.items():
        row = {"Method": METHOD_LABELS.get(m, m)}
        if r["error"]:
            row["Note"] = "unavailable: " + r["error"][:60]
            rows.append(row)
            continue
        if with_gt and r["metrics"]:
            for k, v in r["metrics"].items():
                row[METRIC_LABELS[k]] = v
        row[METRIC_LABELS["latency_ms"]] = r["latency_ms"]
        if r.get("memory_mb") is not None:
            row[METRIC_LABELS["memory_mb"]] = r["memory_mb"]
        rows.append(row)
    return pd.DataFrame(rows)


def range_for(bundle: InputBundle, extra=None):
    ds = [bundle.gt] if bundle.gt is not None else []
    ds.append(bundle.obs.values)
    if extra is not None:
        ds.append(extra)
    return depth_range(*ds)


def method_options(model_ok: bool):
    return [m for m in METHODS if m != "deep_learning" or model_ok]
