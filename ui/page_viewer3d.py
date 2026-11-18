"""3-D flying-pixel viewer: side-by-side point clouds with flying pixels highlighted."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from edgedepth import METHOD_LABELS
from edgedepth.confidence import predict_error
from edgedepth.hybrid.fusion import hybrid_reconstruct
from edgedepth.metrics import flying_pixel_mask
from edgedepth.pipeline import ModelUnavailable, model_available, run_method
from ui.common import hero, input_selector, load_json, method_options

COLOR_MODES = ["Highlight flying pixels (red)", "True colour (RGB)", "Confidence (green = trusted, red = uncertain)"]


def _cloud(depth, rgb, mask_red=None, conf=None, mode=0, max_pts=22000):
    H, W = depth.shape
    step = max(1, int(np.ceil(np.sqrt(H * W / max_pts))))
    yy, xx = np.mgrid[0:H:step, 0:W:step]
    d = depth[::step, ::step]
    f = 0.81 * W  # NYU-like focal length (518 px at 640 px width)
    X = (xx - W / 2) * d / f
    Y = -(yy - H / 2) * d / f
    keep = d > 0
    X, Y, Z = X[keep], Y[keep], d[keep]
    c = rgb[::step, ::step][keep]
    if mode == 2 and conf is not None:
        cf = conf[::step, ::step][keep]
        cols = [f"rgb({int(255 * (1 - v))},{int(200 * v)},60)" for v in cf]
    else:
        cols = [f"rgb({r},{g},{b})" for r, g, b in c]
        if mode == 0:
            cols = [f"rgb({int(r * .45 + 70)},{int(g * .45 + 70)},{int(b * .45 + 70)})" for r, g, b in c]
    traces = [go.Scatter3d(x=X, y=Z, z=Y, mode="markers", marker=dict(size=1.5, color=cols), hoverinfo="skip", showlegend=False)]
    if mode == 0 and mask_red is not None:
        m = mask_red[::step, ::step][keep]
        traces.append(go.Scatter3d(x=X[m], y=Z[m], z=Y[m], mode="markers", marker=dict(size=2.6, color="#ff2d2d"),
                                   name="flying pixels", hoverinfo="skip", showlegend=False))
    return traces


def render():
    hero("3-D Flying-Pixel Viewer", "See boundary quality in 3-D: depths stranded between foreground and background show up as red streaks.")
    st.caption("A *flying pixel* sits in the gap between a near and a far surface — at least 25 % of the gap away from both — "
               "so it belongs to neither object. Interpolation creates them along every occlusion edge; they appear as "
               "streaks when the depth map is turned into a 3-D point cloud.")
    bundle = input_selector("v3d")
    if bundle is None:
        return
    opts = method_options(model_available())
    c1, c2, c3 = st.columns(3)
    ma = c1.selectbox("Left", opts, index=opts.index("bicubic"), format_func=lambda m: METHOD_LABELS[m], key="v3d_a")
    mb = c2.selectbox("Right", opts, index=opts.index("hybrid"), format_func=lambda m: METHOD_LABELS[m], key="v3d_b")
    mode = COLOR_MODES.index(c3.selectbox("Colouring", COLOR_MODES, key="v3d_colour"))
    show_gt = bundle.gt is not None and st.toggle("Also show the ground truth cloud", value=False, key="v3d_gt")
    if st.button("▶ Build 3-D comparison", type="primary", width="stretch"):
        try:
            with st.spinner("Reconstructing and building point clouds …"):
                outs, confs = {}, {}
                for m in dict.fromkeys([ma, mb]):
                    if m == "hybrid":
                        d, det = hybrid_reconstruct(bundle.rgb, bundle.obs, return_details=True)
                        confs[m] = predict_error(bundle.obs, d, det)["confidence"]
                        outs[m] = d
                    else:
                        outs[m] = run_method(m, bundle.rgb, bundle.obs)
            st.session_state["v3d"] = {"bundle": bundle, "outs": outs, "confs": confs, "pair": (ma, mb)}
        except ModelUnavailable as e:
            st.error(str(e))
            return
    data = st.session_state.get("v3d")
    if not data:
        st.info("Choose two methods and press **Build 3-D comparison**. Tip: 8× input makes the difference most visible.")
        return
    b, outs, confs = data["bundle"], data["outs"], data["confs"]
    ma, mb = data["pair"]
    ref = b.gt
    panels = [(METHOD_LABELS[ma], outs[ma], ma), (METHOD_LABELS[mb], outs[mb], mb)]
    if show_gt:
        panels.append(("Ground truth", b.gt, None))
    cols = st.columns(len(panels))
    masks = {}
    for col, (title, d, m) in zip(cols, panels):
        fm, region = flying_pixel_mask(d, ref) if ref is not None else flying_pixel_mask(d)
        masks[title] = fm
        pct = 100 * fm.sum() / max(region.sum(), 1)
        col.metric(f"Flying pixels · {title}", f"{pct:.1f} %",
                   help="Share of pixels in discontinuity regions that are flying. "
                        + ("Measured against the ground truth." if ref is not None else "No ground truth: measured against the output itself (approximate)."))
    if mode == 2 and not confs:
        st.info("Confidence colouring is available for the hybrid method only — select Hybrid on one side.")
    fig = make_subplots(rows=1, cols=len(panels), specs=[[{"type": "scene"}] * len(panels)], subplot_titles=[p[0] for p in panels],
                        horizontal_spacing=0.01)
    for j, (title, d, m) in enumerate(panels):
        for tr in _cloud(d, b.rgb, masks[title], confs.get(m), mode):
            fig.add_trace(tr, row=1, col=j + 1)
    cam = dict(eye=dict(x=1.25, y=-1.45, z=0.8), up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=-0.1))
    scene = dict(aspectmode="manual", aspectratio=dict(x=1.0, y=1.25, z=0.75), xaxis_visible=False, yaxis_visible=False,
                 zaxis_visible=False, camera=cam)
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=30, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      **{f"scene{'' if j == 0 else j + 1}": scene for j in range(len(panels))})
    st.plotly_chart(fig, width="stretch")
    st.caption("Drag to rotate, scroll to zoom. The default view looks along the object edges from the side, where flying "
               "pixels form streaks between surfaces. Points are subsampled for speed; the percentages use every pixel.")

    r = load_json("new_features.json")
    if r:
        fp = r["flying_pixels"]["overall"]
        st.markdown("##### Flying pixels on the full unseen test split (168 inputs, `scripts/eval_new_features.py`)")
        st.dataframe({"Method": [METHOD_LABELS[m] for m in fp], "Flying pixels (%) ↓": [round(v, 1) for v in fp.values()],
                      "2×": [round(r["flying_pixels"]["by_scale"]["2x"][m], 1) for m in fp],
                      "4×": [round(r["flying_pixels"]["by_scale"]["4x"][m], 1) for m in fp],
                      "8×": [round(r["flying_pixels"]["by_scale"]["8x"][m], 1) for m in fp],
                      "Sparse 5 %": [round(r["flying_pixels"]["by_scale"]["sparse"][m], 1) for m in fp]},
                     hide_index=True, width="stretch")
