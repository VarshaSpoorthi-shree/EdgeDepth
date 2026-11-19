import time

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from ui.common import (align_toggle, alignment_report, apply_alignment, depth_vis, hero, input_selector, lr_vis, method_options,
                       range_for, show_images)
from edgedepth import METHOD_LABELS
from edgedepth.classical.edges import canny_edges, depth_discontinuities
from edgedepth.confidence import reliability_warning, ause, predict_error
from edgedepth.hybrid.fusion import hybrid_reconstruct
from edgedepth.io_utils import colorize_error, save_depth_png
from edgedepth.metrics import METRIC_LABELS, all_metrics, measure, no_reference_stats
from edgedepth.model.inference import predict
from edgedepth.pipeline import ModelUnavailable, model_available, run_method


def _heat(x, vmax=1.0):
    import cv2

    v = (np.clip(x / max(vmax, 1e-6), 0, 1) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(v, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)


def point_cloud(depth, rgb, step=4):
    d = depth[::step, ::step]
    c = rgb[::step, ::step].reshape(-1, 3)
    H, W = d.shape
    f = 0.9 * max(H, W)
    yy, xx = np.mgrid[0:H, 0:W]
    X = (xx - W / 2) * d / f
    Y = -(yy - H / 2) * d / f
    fig = go.Figure(go.Scatter3d(x=X.ravel(), y=-d.ravel(), z=Y.ravel(), mode="markers",
                                 marker=dict(size=1.6, color=[f"rgb({r},{g},{b})" for r, g, b in c])))
    fig.update_layout(height=420, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False))
    return fig


def confidence_panel(b, depth, details):
    """Predicted-error / confidence map for the hybrid output (no ground truth used to compute it)."""
    import cv2

    pe = predict_error(b.obs, depth, details)
    err, conf = pe["error_m"], pe["confidence"]
    st.markdown("#### 🎯 Confidence map — where to trust this depth")
    warn = reliability_warning(err)
    if warn:
        st.warning(warn, icon="⚠️")
    emax = float(np.percentile(err, 99)) + 1e-6
    c = st.columns(4)
    c[0].metric("Pixels with high confidence (> 0.8)", f"{(conf > 0.8).mean() * 100:.0f} %")
    c[1].metric("Pixels with low confidence (< 0.3)", f"{(conf < 0.3).mean() * 100:.0f} %")
    c[2].metric("Median predicted error", f"{np.median(err) * 100:.1f} cm")
    items = [("Confidence (yellow = trusted, dark = uncertain)", _heat(conf, 1.0)),
             (f"Predicted error (0–{emax * 100:.0f} cm) — computed without ground truth", _heat(err, emax))]
    if b.gt is not None:
        v = b.gt > 0
        true = np.abs(depth - b.gt) * v
        items.append((f"True error (0–{emax * 100:.0f} cm) — for checking", _heat(true, emax)))
        ra = np.argsort(np.argsort(err[v]))
        rb = np.argsort(np.argsort(true[v]))
        rho = float(np.corrcoef(ra, rb)[0, 1])
        c[3].metric("Rank correlation with true error", f"{rho:.2f}", help="Spearman correlation between predicted and true per-pixel error (1 = perfect ranking).")
    show_images(items, cols=3)
    if b.gt is not None:
        fr = np.linspace(0, 0.9, 19)
        a, curve, oracle = ause(true[v], err[v], fr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fr * 100, y=oracle, name="Oracle (true error)", line=dict(dash="dash", color="white")))
        fig.add_trace(go.Scatter(x=fr * 100, y=curve, name=f"Confidence map (AUSE {a:.3f})", line=dict(width=3, color="#4FD1C5")))
        fig.add_hline(y=1.0, line=dict(dash="dot", color="gray"), annotation_text="random")
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="% least-confident pixels removed",
                          yaxis_title="remaining RMSE (relative)", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          title="Sparsification on this image: removing low-confidence pixels should cut the error fast",
                          legend=dict(orientation="h", y=-0.35))
        st.plotly_chart(fig, width="stretch")
    st.caption("The map is predicted from the hybrid's own internal signals (method disagreement, correction size, distance to "
               "the nearest measurement, edge probability, noise level), with a model fitted on the validation split. "
               "On the unseen test split, hiding the 15 % least-confident pixels cuts the remaining RMSE by about two thirds.")
    ok, buf = cv2.imencode(".png", (np.clip(conf, 0, 1) * 255).astype(np.uint8))
    st.download_button("⬇ Download confidence map (8-bit PNG, 255 = fully trusted)", buf.tobytes(), file_name="edgedepth_confidence.png", mime="image/png")


def render():
    hero("Depth Reconstruction", "Upload or pick an RGB-D pair, degrade it (or feed real low-res depth) and reconstruct.")
    model_ok = model_available()
    bundle = input_selector("rec")
    if bundle is None:
        return
    opts = method_options(model_ok)
    method = st.selectbox("Reconstruction method", opts, index=opts.index("hybrid"), format_func=lambda m: METHOD_LABELS[m])
    snap = False
    if method == "hybrid":
        snap = st.toggle("Extra boundary snapping (weighted median) — sharper edges, usually slightly higher RMSE", value=False)
    auto = align_toggle("rec")
    run = st.button("▶ Run reconstruction", type="primary", width="stretch")
    if run:
        try:
            with st.spinner(f"Running {METHOD_LABELS[method]} …"):
                rgb_used, est = apply_alignment(bundle, auto)
                details = None
                if method == "hybrid":
                    (depth, details), ms, mb = measure(hybrid_reconstruct, rgb_used, bundle.obs, boundary_snapping=snap, return_details=True)
                else:
                    depth, ms, mb = measure(run_method, method, rgb_used, bundle.obs)
                dl = predict(rgb_used, bundle.obs) if method == "deep_learning" else None
            st.session_state["rec_result"] = {"bundle": bundle, "method": method, "depth": depth, "ms": ms, "mb": mb,
                                              "details": details, "dl": dl, "t": time.time(), "align": est}
            st.session_state["bundle"] = bundle
        except ModelUnavailable as e:
            st.error(str(e))
            return
        except Exception as e:  # surface any failure clearly instead of a stack trace
            st.error(f"Reconstruction failed: {e}")
            return
    res = st.session_state.get("rec_result")
    if not res:
        st.info("Configure the input and press **Run reconstruction**.")
        return
    if res["bundle"].label != bundle.label or res["bundle"].obs.values.shape != bundle.obs.values.shape:
        st.caption("Showing the previous result — press Run to update it for the current settings.")
    b, depth = res["bundle"], res["depth"]
    lo, hi = range_for(b)
    H, W = b.obs.target_hw
    alignment_report(res.get("align"), getattr(b, "rgb_shift", (0, 0)))
    st.markdown(f"#### Result — {METHOD_LABELS[res['method']]}")
    k = st.columns(4)
    k[0].metric("Processing time", f"{res['ms']:.0f} ms")
    k[1].metric("Peak memory (approx.)", f"{res['mb']:.1f} MB")
    k[2].metric("Output resolution", f"{W}×{H}")
    k[3].metric("Input", f"{b.obs.values.shape[1]}×{b.obs.values.shape[0]} · {b.obs.mask.mean() * 100:.1f}% valid"
                if b.obs.mode == "lowres" else f"sparse {b.obs.mask.mean() * 100:.2f}%")
    items = [("RGB guide", b.rgb), (f"Input depth ({'×%d low-res' % b.obs.scale if b.obs.mode == 'lowres' else 'sparse'}, nearest view)", lr_vis(b.obs, lo, hi)),
             ("Reconstruction", depth_vis(depth, lo, hi))]
    if b.gt is not None:
        items.append(("Ground truth", depth_vis(b.gt, lo, hi)))
    show_images(items, cols=4)

    if b.gt is not None:
        m = all_metrics(depth, b.gt)
        cols = st.columns(7)
        for c, (key, v) in zip(cols, m.items()):
            c.metric(METRIC_LABELS[key], f"{v:.4f}" if key not in ("psnr",) else f"{v:.2f}")
        show_images([("Absolute error (0 – 0.3 m)", colorize_error(np.abs(depth - b.gt), 0.3)),
                     ("GT depth discontinuities", _heat(depth_discontinuities(b.gt).astype(np.float32))),
                     ("Predicted depth discontinuities", _heat(depth_discontinuities(np.maximum(depth, 1e-3)).astype(np.float32))),
                     ("RGB Canny edges", _heat(canny_edges(b.rgb).astype(np.float32)))], cols=4)
    else:
        st.info("No ground truth for this input → reference metrics (RMSE, MAE, PSNR, SSIM, edge accuracy) are **not available**. "
                "Showing no-reference statistics instead.")
        st.json({k: round(v, 3) for k, v in no_reference_stats(depth, b.rgb).items()})

    if res["details"] is not None:
        confidence_panel(b, depth, res["details"])
        d = res["details"]
        with st.expander("Hybrid internals — fusion weights, cues and edge-aware correction", expanded=True):
            pre = d["preprocess"]
            st.caption(f"Pre-processing: estimated noise σ = {pre['noise_sigma_est_m'] * 1000:.1f} mm "
                       f"({pre.get('rel_noise', 0) * 100:.2f}% of depth, threshold {pre.get('threshold', 0) * 100:.2f}%) → "
                       f"{'denoised before the classical experts' if pre['denoised'] else 'no denoising'}. "
                       f"Network used: {d['used_dl']}. Back-projection gain α = {d.get('bp_gain', 1):.2f}."
                       + (f" Snapping band covers {d['band'].mean() * 100:.1f}% of pixels." if d["band"].any() else ""))
            show_images([(f"weight · {METHOD_LABELS[n]}", _heat(w)) for n, w in d["weights"].items()], cols=4)
            cue_names = {"B_dl": "Boundary prob. (network)", "B_cl": "Discontinuity prob. (classical)", "V": "Local depth variation",
                         "E": "RGB edge strength", "H": "Missing-measurement indicator"}
            corr = d["correction"]
            cmax = float(np.percentile(np.abs(corr), 99)) + 1e-6
            show_images([(cue_names[k], _heat(v, 1.0 if k != "V" else 3.0)) for k, v in d["cues"].items()] +
                        [(f"|back-projection correction| (0–{cmax * 100:.1f} cm)", _heat(np.abs(corr), cmax))], cols=3)
            if d["band"].any():
                show_images([("Snapping band", _heat(d["band"].astype(np.float32)))], cols=3)
    if res["dl"] is not None:
        with st.expander("Network internals", expanded=False):
            show_images([("Backbone output (before CSPN)", depth_vis(res["dl"]["coarse"], lo, hi)),
                         ("Predicted boundary probability", _heat(res["dl"]["edge_prob"])),
                         ("Mean RGB gate (level 1)", _heat(res["dl"]["rgb_gate"]))], cols=3)

    c1, c2 = st.columns([1, 3])
    c1.download_button("⬇ Download depth (16-bit PNG, mm)", save_depth_png(depth), file_name=f"edgedepth_{res['method']}.png", mime="image/png")
    if c2.toggle("Show 3-D point cloud", value=False):
        st.plotly_chart(point_cloud(depth, b.rgb, step=max(2, W // 160)), width="stretch")
