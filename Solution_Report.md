# Hackathon Technical Solution Report

| Field | Details |
| :--- | :--- |
| **Problem ID** | `P4` |
| **Problem Title** | Edge-Preserving Depth Map Upsampling and Refinement |
| **Team Name** | `<Team Name>` |
| **Team Members** | `<Full Name 1> (<USN 1>)` · `<Full Name 2> (<USN 2>)` · `<Full Name 3> (<USN 3>)` |
| **GitHub Repository** | https://github.com/VarshaSpoorthi-shree/EdgeDepth — commit evaluated: `<commit hash>` |
| **Live Demo / Deployment** | https://edgedepth-p4.streamlit.app/ (Streamlit Community Cloud, CPU) |

---

## 1. Problem Understanding & Solution Strategy

* **Core Technical Challenge:** Commodity depth sensors (Kinect, ToF, LiDAR) produce depth maps that are **low resolution, noisy and full of holes**, while the RGB camera next to them is sharp. Upsampling such depth by 2×, 4× or 8× fails in three typical ways: (1) interpolation **blurs depth discontinuities**, creating "flying pixels" halfway between foreground and background; (2) **missing values and sensor noise** (depth-dependent, ≈ 2.6 cm at 4 m for a Kinect) propagate into the output; (3) naive RGB guidance causes **texture copying** — colour edges that are not depth edges are printed into flat surfaces. Target constraints: CPU-only inference, < 10 s per request on a free 2-vCPU / 1 GB host, one model for 2×/4×/8× and for sparse input, and evaluation only on scenes never used for training or tuning.
* **Proposed Methodology:** EdgeDepth treats every method as an *expert* that maps the same observation to a full-resolution depth map: classical interpolators and RGB-guided filters, plus **EdgeDepthNet**, a 0.72 M-parameter CNN with depth-gated RGB fusion and learned edge-aware propagation. A **cue-driven softmax mixture of experts** fuses them per pixel (cues: boundary probability, RGB edges, depth variation, missing data, texture-copy score, data residual), and a **noise-adaptive, edge-aware back-projection** then forces the result to agree with the real measurements without leaking corrections across depth edges. We chose this over a static baseline because no single method is best everywhere: interpolation is unbiased but blurry, RGB filters are sharp but copy texture, and the network is accurate but can drift from the measurements — the hybrid keeps each one's strength and is checked against the data.

---

## 2. Technical Architecture & Implementation

![System architecture](results/figures/report/fig0_architecture.png)

*Figure 0 — system architecture (Graphviz source `docs/report_architecture.dot`; detailed Mermaid diagrams in [`docs/architecture.md`](docs/architecture.md)). Code map: `edgedepth/classical/` (classical CV), `edgedepth/model/` (network), `edgedepth/hybrid/fusion.py` (fusion + back-projection), `edgedepth/pipeline.py` (single entry point for all methods), `edgedepth/metrics.py`, `app.py` + `ui/` (Streamlit).*


### 2.1 Classical Algorithmic Baseline
* **Algorithms Utilized** (all implemented as separate modules in `edgedepth/classical/`, each producing a real depth output):
  * **Interpolation** — nearest, bilinear, bicubic on the low-res grid; for sparse input their scattered-data equivalents (nearest sample, Delaunay-linear, Clough–Tocher cubic).
  * **Joint bilateral upsampling** (Kopf et al. 2007) — each HR pixel is a weighted mean of LR measurements with spatial × RGB-similarity × validity-mask weights, so missing values get zero weight.
  * **Colour guided filter** (He et al. 2013) — local linear model q = aᵀI + b of the RGB guide applied to the bilinear upsampling.
  * **Edge detection** — Sobel colour gradient and a per-channel colour Canny (so edges between colours of equal brightness are not lost).
  * **Depth-discontinuity detection** — threshold on ‖∇ log D‖ (relative depth change per pixel, so a 5 cm step at 1 m and a 50 cm step at 10 m are treated alike) with non-maximum thinning.
  * **Noise removal** — Immerkær noise estimate, then 3×3 median + bilateral filter.
  * **Missing-depth detection and hole filling** — zero / non-finite / out-of-range detection; push-pull normalised-convolution pyramid (Telea inpainting as an alternative).
  * **Boundary snapping** (optional) — joint weighted median in a thin band around discontinuities.
* **Adaptive Selection Logic** — parameters are derived from the input, not fixed:
  * *Noise level:* σ̂ is estimated per input with Immerkær's 3×3 plane-cancelling kernel on the smoothest 50 % of pixels; the bilateral range is set to **σ_r = 3σ̂**, so only variations comparable to the noise are smoothed.
  * *Denoise decision:* the observation is denoised only if σ̂ / median depth exceeds a **scale-dependent threshold** (0.22 % / 0.35 % / 0.85 % at 2× / 4× / 8×), because on coarse grids real scene structure inflates the estimate.
  * *Window sizes:* the guided-filter radius scales with the factor, r = max(1, round(s/4)); for sparse input the JBU and back-projection radius follows the measured density, r = ⌈1.6/√density⌉.
  * *Edge thresholds:* Canny thresholds are set from the channel median ((1 ± 0.33)·median); the discontinuity test is relative, so it adapts to scene depth.
  * *Measurement trust:* the back-projection gain α (Section 2.3) is computed from σ̂.
  * Fixed hyper-parameters (JBU σ_s = 0.5, σ_r = 0.2; guided-filter ε = 1e-5) were selected by grid search on the **validation** split only (`scripts/tune_classical.py`, `results/classical_tuning.json`).

### 2.2 Deep Learning Pipeline
* **Model Architecture:** **EdgeDepthNet**, a custom lightweight CNN (`edgedepth/model/network.py`, **723,882 parameters**, ≈ 0.15 s per 320×240 frame on CPU).
  * *Inputs:* RGB; initial depth D0 (bicubic of the hole-filled LR map, or push-pull fill of sparse samples) divided by its mean μ for scale invariance across sensors; the measurement mask; a scale code log₂(s)/3, so **one model serves 2×, 4×, 8× and sparse input**.
  * *Encoders:* two encoders (RGB and depth) at 1, 1/2 and 1/4 resolution, and a 1/8 bottleneck with residual blocks.
  * *Depth-gated fusion* at every level: f = f_d + Conv([f_d, σ(W[f_d, f_r]) ⊙ f_r]). The gate lets RGB features through only where the depth branch supports an edge — an architectural defence against texture copying.
  * *Decoder and heads:* a U-Net decoder with three heads — depth residual on D0, boundary probability, and 8 neighbour affinities.
  * *Learned edge-aware refinement:* 6 steps of convolutional spatial propagation (CSPN) using the predicted affinities. It diffuses depth along surfaces but not across predicted edges, and starts as the identity.
  * *Rationale:* residual learning on D0 makes the task "learn only the correction", which is feasible with a small dataset; the model is small enough to train on CPU and to serve on a free host. Larger transformer depth models were ruled out by the CPU and 1 GB deployment budget.
* **Loss Functions & Training:**
  * *Loss:* L = L1_w(D_ref) + RMSE_w(D_ref) + 0.5·L1_w(D_coarse) + L_grad + 0.1·BCE(edge). The `_w` terms weight pixels within 2 px of a ground-truth discontinuity ×3. L1 keeps edges sharp, the RMSE term penalises misplaced edges, the gradient loss penalises blurred or shifted steps, and the boundary loss trains the edge head. The coarse-output term supervises the pre-CSPN output too.
  * *Dataset:* NYU Depth V2, 179 RGB-D frames from 10 scenes, resized to 320×240. Splits are **scene-disjoint**: 126 train / 29 validation / 24 test frames.
  * *Degradations on the fly:* box-downsampling ×2/4/8 or 1–10 % sparse sampling; Kinect depth-dependent noise ×0.3–2.5; random, blob and edge-shadow holes; colour and depth-scale jitter.
  * *Training run:* Adam with learning rate 1e-3 and cosine decay, batch 8, 128×160 crops, 5000 iterations, **51.6 min on 2 CPU cores**. Best validation checkpoint: mean RMSE 0.0427 m. Training log: `results/training_log.json`.

### 2.3 Hybrid Integration & Adaptive Mechanism
* **Fusion Strategy:** The four experts (bicubic, guided filter, JBU, EdgeDepthNet) are fused per pixel by a softmax mixture of experts:

  z_k(p) = β_k0 + Σ_f β_kf · φ_kf(p),  w_k(p) = softmax_k z_k(p),  D(p) = Σ_k w_k(p) D_k(p)

  The cue vector φ_k carries the classical priors into the decision:
  * *Edge evidence:* network boundary probability **B_dl** and classical discontinuity probability **B_cl**.
  * *Local structure:* depth variation **V** and RGB edge strength **E**.
  * *Missing data:* missing-measurement indicator **H**.
  * *Per-expert reliability:* texture-copy score **T_k** (an RGB edge with no depth edge, where the expert departs from bicubic), LR data-consistency residual **R_k**, and deviation from the expert consensus **A_k**.

  The 36 coefficients β are fitted on 1.74 M validation pixels by minimising L1. The output is convex, so it never overshoots the experts, and each β is interpretable.

  The fused map is then refined by an **edge-aware back-projection** for 3 iterations:

  r = M ⊙ (D_lr − ↓D),  D ← D + JTU(r)

  JTU spreads each low-resolution residual with **space × RGB × log-depth** weights. Corrections therefore stay on the surface they were measured on, follow RGB boundaries, and are never invented inside holes (M = 0). This is the classical prior guiding the network's output back to the data.
* **Decision Logic** — the system balances *trusting the measurements* against *suppressing their noise*, and *using RGB edges* against *copying RGB texture*:
  * **Measurement trust vs noise:** the back-projection correction is scaled by **α = clip(1 − (q − 1)/2.5, 0, 1)**, where q is the estimated noise relative to the scale threshold. Clean data gets the full correction (α = 1); very noisy data gets none (α = 0), because there the network's estimate is better than the raw samples. The curve was fitted on a validation sweep. It fixed a measured failure: without it, the hybrid's RMSE at ×8 noise was 0.135 m, worse than the network's 0.107 m.
  * **Edges vs texture:** RGB enters through the network's depth-controlled gate, the texture-copy cue, and the log-depth term of the back-projection, which prevents corrections from crossing depth edges.
  * **Missing regions:** holes get zero weight in JBU and in the back-projection; there the network's learned prior fills in.
  * **Honest observation:** with the trained network available, the fitted gate assigns it ≈ 99 % of the weight everywhere (it is the best expert in every cue configuration observed). The adaptive weighting is therefore decisive mainly in the **classical fallback**, used when the checkpoint is unavailable. There it drops the guided filter, distrusts JBU where it disagrees with the measurements (β_R = −0.91) and trusts it more in holes (Section 3.3, Figure 4).

---

## 3. Experimental Benchmarks & Evaluation

**Protocol** (`scripts/evaluate.py`):
* *Test data:* 24 unseen NYU Depth V2 test frames (scenes `basement_3` and `study_9`, never used for training, tuning or fusion fitting), 320×240.
* *Conditions:* {2×, 4×, 8×} × {clean; noisy + missing (Kinect noise ×1, 5 % random dropout, 50 % dropout on depth edges)}, plus 5 % sparse input — 168 observations in total.
* *Fairness:* identical inputs and fixed seeds for every method, and identical output size.
* *Hardware:* 2 CPU cores, no GPU.

### 3.1 Quantitative Benchmarks

Mean over all 168 test observations:

| Approach / Model Variant | PSNR (dB) ↑ | SSIM ↑ | Perceptual Metric → *Boundary RMSE (m)* ↓ | Latency (ms / frame) |
| :--- | :---: | :---: | :---: | :---: |
| **Classical Baseline** (bicubic, best classical method on RMSE) | 43.87 | 0.9259 | 0.2157 | 6 ms |
| **Deep Learning Only** (EdgeDepthNet) | 45.92 | 0.9520 | 0.1659 | 137 ms |
| **Proposed Hybrid System** | **46.61** | **0.9548** | **0.1567** | 549 ms |

*Notes:*
* *Why boundary RMSE instead of LPIPS/NIQE:* those are perceptual metrics for natural RGB images and are not meaningful for metric depth maps. P4 asks for **edge preservation and boundary accuracy**, so the perceptual column reports **boundary RMSE** (RMSE within 2 px of ground-truth depth discontinuities).
* *PSNR and SSIM:* PSNR uses a 10 m peak (NYU sensor range); SSIM uses the per-frame ground-truth depth range.
* *Latency:* the means above include the slower sparse case. For low-resolution input at 4× the latency is 0.4 ms (bicubic), 153 ms (network) and 383 ms (hybrid); a 640×480 frame takes ≈ 1.2–1.4 s with the hybrid.
* *Memory:* approximate peak memory at 4× is 0.6 MB (bicubic), 2.3 MB (network, incremental) and 20.7 MB (hybrid).

**Extended benchmark (all seven methods, same 168 observations):**

| Method | RMSE (m) ↓ | MAE (m) ↓ | PSNR ↑ | SSIM ↑ | Boundary RMSE (m) ↓ | Edge F1 ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Nearest neighbour | 0.1125 | 0.0476 | 41.91 | 0.8872 | 0.2643 | 0.782 |
| Bilinear | 0.0964 | 0.0433 | 43.38 | 0.9269 | 0.2264 | 0.597 |
| Bicubic | 0.0921 | 0.0413 | 43.87 | 0.9259 | 0.2157 | 0.658 |
| Joint bilateral (JBU) | 0.0986 | 0.0413 | 43.07 | 0.9235 | 0.2383 | 0.768 |
| Guided filter | 0.0976 | 0.0446 | 43.18 | 0.9264 | 0.2290 | 0.746 |
| Deep learning (EdgeDepthNet) | 0.0690 | 0.0289 | 45.92 | 0.9520 | 0.1659 | 0.870 |
| **Hybrid (proposed)** | **0.0650** | **0.0264** | **46.61** | **0.9548** | **0.1567** | **0.880** |

**Performance across upsampling factors (RMSE in m, mean of clean and noisy):** at 2×, bicubic 0.0514 → hybrid 0.0360; at 4×, 0.0857 → 0.0577; at 8×, 0.1381 → 0.0964. That is about 30 % lower RMSE at every factor. The hybrid is also best on RMSE in every single condition, including 5 % sparse input (0.0745 vs 0.0814 m for the network alone). On three frames never seen in training (real TUM RGB-D and SUN RGB-D frames, and one synthetic Redwood frame with simulated sensor depth) at 8×, it reaches 0.0804 m against 0.0859 m (network alone) and 0.1097 m (bicubic).

![Metrics versus upsampling factor](results/figures/report/fig6_metrics_vs_scale.png)

* **Analysis:** The hybrid gives the best structural fidelity: PSNR +2.7 dB and SSIM +0.029 over bicubic, RMSE −29 %, and boundary RMSE −27 %. The cost is inference speed. It runs four experts, cue maps and three back-projection passes, so it is ≈ 4× slower than the network alone and ≈ 90× slower than bicubic, yet still well within the 10 s interactive budget on CPU. Classical RGB-guided filters improve edge F1 (sharper edges) but not RMSE on NYU, because many colour edges are texture rather than depth edges.

### 3.2 Visual Outputs & Intermediate Representations

All figures are generated from the shipped checkpoint on unseen test frames by `scripts/make_report_figures.py`. Depth colour scales are shared within each figure, and the error maps use a 0–0.3 m scale.

**1. Extreme degradation — raw input vs. reconstructed output** (8× upsampling, Kinect noise, 35 % of low-res measurements missing):

![Extreme degradation: input vs outputs](results/figures/report/fig1_extreme_8x_noisy.png)

**Sparse depth completion** — only 5 % and 1 % of pixels measured:

![Sparse depth completion](results/figures/report/fig2_sparse_completion.png)

**2. Depth discontinuities — preserving object boundaries** (the depth counterpart of "preserving bright regions without overexposure"). The zoom and the depth profile along one image row show that bilinear and bicubic ramp across each depth step, while the hybrid follows the ground-truth step:

![Boundary preservation and depth profile](results/figures/report/fig3_boundary_profile.png)

**3. Intermediate states** — the network's RGB gate and boundary probability, missing-measurement cue, back-projection correction, and fusion weights:
* *Top row:* RGB Canny edges vs. the depth discontinuities actually reconstructed — far fewer, i.e. texture edges are not copied.
* *Middle row:* the network output before and after CSPN, and the hole cue. The back-projection correction concentrates on surfaces and stops at depth edges.
* *Bottom row:* the weight on the network (≈ 99 %, uniform), and the genuinely adaptive per-pixel weights of the classical fallback.

![Intermediate representations](results/figures/report/fig4_intermediate_maps.png)

A full comparison of all seven methods on one frame is in [`results/figures/qualitative_8x.png`](results/figures/qualitative_8x.png). Every view can be reproduced interactively in the live app (Method Comparison, and the Hybrid internals on the Depth Reconstruction page).

### 3.3 Ablation Study

The same 24 test frames × {2×, 4×, 8×} × {clean, noisy} — 144 cases, identical inputs (`scripts/ablation.py`, `results/ablation.json`):

| Variant | RMSE (m) ↓ | MAE (m) ↓ | Boundary RMSE (m) ↓ | Edge F1 ↑ | SSIM ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| A1 Interpolation only (bicubic) | 0.0917 | 0.0420 | 0.2136 | 0.635 | 0.9263 |
| A2 Interpolation + guided filtering | 0.0986 | 0.0461 | 0.2293 | 0.736 | 0.9255 |
| A3 Deep learning only (backbone, before CSPN) | 0.0712 | 0.0319 | 0.1693 | 0.846 | 0.9418 |
| A4a Deep learning + learned edge-aware refinement (CSPN) | 0.0669 | 0.0283 | 0.1601 | 0.870 | 0.9532 |
| A4b A4a + classical edge-aware back-projection | 0.0634 | 0.0265 | 0.1518 | 0.877 | 0.9550 |
| **A5 Complete hybrid** | **0.0634** | **0.0265** | **0.1517** | 0.877 | **0.9551** |
| X1 Hybrid without back-projection | 0.0668 | 0.0283 | 0.1601 | 0.869 | 0.9532 |
| X2 Hybrid with constant weights (no cues) | 0.0634 | 0.0265 | 0.1517 | 0.877 | 0.9551 |
| X3 Plain average of the four experts | 0.0889 | 0.0423 | 0.2059 | 0.757 | 0.9370 |
| X4 Hybrid without the network (classical fallback) | 0.0885 | 0.0387 | 0.2099 | 0.799 | 0.9333 |
| X5 Hybrid + weighted-median boundary snapping | 0.0720 | 0.0279 | 0.1759 | **0.887** | 0.9502 |

![Ablation](results/figures/report/fig5_ablation.png)

* **Learned RGB–depth fusion (A1 → A3): the largest gain**, −22 % RMSE.
* **CSPN refinement (A3 → A4a):** −6 % RMSE and +0.011 SSIM.
* **Classical edge-aware back-projection (A4a → A4b, X1 → A5):** −5 % RMSE and −5 % boundary RMSE. With and without classical guidance, this is the hybrid's contribution over the network alone.
* **Fixed vs adaptive weights (X2 vs A5):** identical when the network is present, because the gate selects the network almost everywhere. Without the network (X4), the adaptive fusion with back-projection beats the best classical method (0.0885 vs 0.0917 m).
* **Unjustified averaging (X3)** is 40 % worse than the hybrid.
* **Guided filtering on interpolation (A2)** and **weighted-median snapping (X5)** both raise edge F1 but worsen RMSE. Snapping is therefore an optional switch in the UI, not the default.

---

## 4. Failure Modes & Limitations

* **Observed Failure Case — excessive sensor noise (×8 the Kinect noise model, 4× upsampling).** Error grows sharply for every method. Mean RMSE over 8 unseen test frames:

  | Noise level | ×1 | ×2 | ×4 | ×8 |
  | :--- | :---: | :---: | :---: | :---: |
  | Bicubic | 0.0619 | 0.0656 | 0.0783 | 0.1151 |
  | Guided filter | 0.0712 | 0.0729 | 0.0792 | **0.0998** |
  | Deep learning only | 0.0470 | **0.0500** | 0.0628 | 0.1072 |
  | Hybrid | **0.0451** | 0.0501 | **0.0627** | 0.1068 |

  At ×8 the hybrid and the network are **worse than the classical guided filter**. In the first version of the hybrid, the back-projection made this much worse (0.135 m at ×8).

  ![Failure: excessive noise](results/figures/failure_F3_noise_x8.png)

* **Root Cause & Mitigation:**
  * *Root cause, training range:* the network was trained with noise ×0.3–×2.5. At ×8 the per-sample error (several cm at 3–5 m) exceeds many real depth steps, so the network — trained to trust its input — passes noise and false edges through.
  * *Root cause, back-projection:* it treats measurements as constraints. With very noisy measurements the residual r = D_lr − ↓D is mostly noise, so adding it re-injects noise.
  * *Root cause, estimator at 8×:* the Immerkær estimator cannot fully separate scene structure from noise on coarse grids.
  * *Mitigation already implemented:* the noise-adaptive gain α = clip(1 − (q − 1)/2.5, 0, 1), fitted on validation, which brought the hybrid back to the network's level (0.107 m).
  * *Proposed fixes:* train with a wider noise range (up to ×8) and feed σ̂ to the network as an extra input channel; switch the network's initial depth D0 from bicubic to the guided-filter output when σ̂ is high; use a robust (Huber) residual in the back-projection.

* **Further measured failures** (`docs/failure_analysis.md`, reproducible live on the app's Failure Analysis page):
  * **Extreme sparsity:** at ≤ 0.5 % density, hybrid RMSE exceeds 0.16 m, because whole objects contain no sample.
  * **Large holes:** inside a hole covering 20 % of the map, error is ≈ 10× higher than outside (0.44 m vs 0.04 m), because no measurement fixes the absolute depth there.
  * **Texture copying:** on a flat plane with textured RGB, the error on texture edges is 1.66× the error elsewhere for JBU and 1.32× for the hybrid. The RGB term of the back-projection lets some texture through.
  * **RGB–depth misalignment of 8 px:** the hybrid's boundary RMSE rises from 0.188 to 0.221 m, because every RGB-guided method places edges where the colour edge is.

  ![Failure: texture copying](results/figures/failure_F4_texture_copy.png)

---

## 5. Deployment & Quickstart Guide

### Quickstart Commands
```bash
# 1. Clone the repository and install dependencies (Python 3.10+; git-lfs needed for the model and images)
git lfs install
git clone https://github.com/VarshaSpoorthi-shree/EdgeDepth.git
cd EdgeDepth
pip install -r requirements.txt

# 2. Run inference on a sample directory (<name>_rgb.png + <name>_depth.png pairs)
#    --simulate degrades the given depth (8x, noise) and reports metrics against it
python scripts/infer.py --input_dir ./data/samples/ --output_dir ./results/inference/ --simulate --scale 8 --noise 1

#    Real low-resolution depth (no ground truth -> metrics reported as unavailable)
python scripts/infer.py --rgb rgb.png --depth lowres_depth.png --out hr_depth.png

# 3. Launch the web app locally
streamlit run app.py

# 4. Reproduce the benchmark, ablation, failure study, report figures and tests
python scripts/evaluate.py && python scripts/ablation.py && python scripts/failure_cases.py
python scripts/make_report.py && python scripts/make_report_figures.py
python -m pytest -q tests
```
Outputs are written as 16-bit PNGs in millimetres, together with colour previews and a `report.json` of latency, memory and metrics.

### Deployment Format
* **Streamlit dashboard**, deployed on **Streamlit Community Cloud**: free tier, CPU only, Python 3.11, dependencies from `requirements.txt` (CPU PyTorch wheels). The checkpoint and images are stored with Git LFS. Every push to `main` redeploys automatically, and no secrets or API keys are used.
* **Six sections:** Home, Depth Reconstruction (upload or sample; factor, method, noise/missing simulation; processing time; downloadable result), Method Comparison (all seven methods, shared colour scale, zoom, depth profile), Evaluation Dashboard (live metrics, benchmark, ablation, mini-benchmark), Technical Architecture, and Failure Analysis.
* **Robustness:** reference metrics are shown only when ground truth exists. Invalid uploads produce readable errors. A classical fallback runs automatically if the model checkpoint is missing.
* **Also provided:** a CLI (`scripts/infer.py`, single file or batch) and a `Dockerfile` for any Docker host.

---

### Student Submission Checklist
- [x] Working GitHub repository link included with an updated `README.md`.
- [x] No raw code pasted into this report.
- [x] All table metrics match test runs reproducible via the repository (`results/summary.json`, `results/ablation.json`, `results/failure_cases.json`).
- [ ] All team member details accurately recorded. *(fill in Team Name, member names/USNs and the commit hash above)*
