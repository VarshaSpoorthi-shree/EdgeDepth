---
app_port: 7860
colorFrom: indigo
colorTo: green
emoji: 🧊
license: mit
pinned: false
sdk: docker
short_description: RGB-guided, edge-preserving depth upsampling (P4)
title: EdgeDepth
---

# EdgeDepth --- RGB-Guided, Edge-Preserving Depth Map Upsampling and Refinement

**Computer Vision Hackathon 2026 · Problem P4 · Team `Golden Trio`**

EdgeDepth transforms low-resolution, sparse, noisy, or incomplete depth
maps into dense, high-resolution maps while preserving sharp object
boundaries and depth discontinuities. It uses the paired RGB image as
guidance without simply transferring its texture into the depth map.

The main approach is a **hybrid reconstruction pipeline** that combines
classical interpolation, edge-aware filtering, and a lightweight CNN
through a per-pixel, cue-driven mixture of experts.

-   **Live app:** <https://edgedepth-p4.streamlit.app/>
-   **Evaluation file:** [`eval_P4_TeamID.md`](eval_P4_TeamID.md) ·
    **Design document:**
    [`docs/design_document.md`](docs/design_document.md)
-   **Architecture:** [`docs/architecture.md`](docs/architecture.md)
    ([Mermaid source](docs/architecture.mmd)) · **Ablation study:**
    [`docs/ablation_study.md`](docs/ablation_study.md) · **Failure
    analysis:** [`docs/failure_analysis.md`](docs/failure_analysis.md)
-   **Demo script:** [`docs/demo_script.md`](docs/demo_script.md) ·
    **Viva Q&A:** [`docs/viva_qa.md`](docs/viva_qa.md)

## Problem

Depth sensors often produce low-resolution measurements with noise,
missing values (for example, on absorbing or specular surfaces and
around sensor-shadowed occlusions), and blurred object boundaries. At
the same time, the paired RGB image is usually much sharper.

Simple upsampling can blur boundaries and create *flying pixels*, while
naive RGB guidance may copy colour and texture details into otherwise
flat depth regions. Problem P4 focuses on 2×, 4×, and 8× upsampling,
including hole filling, denoising, and boundary preservation. It also
requires comparison across interpolation, classical refinement, deep
learning, and hybrid approaches.

## Features

-   **Seven methods in one interface**
    (`edgedepth.pipeline.run_method`): nearest-neighbour, bilinear,
    bicubic, joint bilateral upsampling, guided filter, EdgeDepthNet
    (CNN), and hybrid. All methods use the same input and output
    dimensions.
-   **Classical computer vision modules:** grid and scattered
    interpolation, JBU, colour-guided filtering, Sobel and colour Canny
    edges, scale-invariant depth-discontinuity detection, Immerkær noise
    estimation, edge-preserving denoising, missing-depth detection,
    push-pull hole filling, and weighted-median boundary snapping.
-   **EdgeDepthNet (0.72 M parameters):** dual RGB/depth encoders,
    depth-gated RGB fusion to reduce texture copying, a U-Net decoder,
    residual, boundary and affinity heads, and CSPN refinement. One
    model supports 2×, 4×, and 8× upsampling, as well as sparse inputs.
    It was trained on CPU using scene-disjoint NYU Depth V2 frames.
-   **Hybrid fusion:** a softmax mixture of four experts driven by eight
    per-pixel cues, including boundary probability, RGB edges, depth
    variation, missing data, texture-copy score, data-consistency
    residual, and consensus deviation. The fusion is fitted on the
    validation split and followed by edge-aware joint-trilateral
    back-projection to maintain consistency with raw measurements. A
    classical fallback is available when the checkpoint is missing.
-   **Realistic degradations:** box downsampling, Kinect depth-dependent
    noise, random, blob and edge-shadow holes, and sparse sampling.
-   **Streamlit application:** a dark, responsive interface with Home,
    Depth Reconstruction, Method Comparison (shared colour scale, zoom
    and boundary profile), Evaluation Dashboard (live results,
    benchmarks, ablation and mini-benchmark), Technical Architecture,
    and Failure Analysis sections. Users can upload their own RGB-D pair
    or low-resolution depth map.
-   **Evaluation with clear ground-truth handling:** reference metrics
    are reported only when ground truth is available. The documented
    values are generated from `results/*.json` using
    `scripts/make_report.py`.

## Results (unseen NYU Depth V2 test split)

Results are averaged over 24 test frames, three upsampling factors (2×,
4×, and 8×), clean and noisy-plus-missing conditions, and an additional
5% sparse-input condition.

```{=html}
<!-- AUTO:overall -->
```
  ------------------------------------------------------------------------------------------------------------------
  Method             RMSE (m) ↓    MAE (m) ↓ PSNR (dB) ↑       SSIM ↑     Boundary   Edge F1 ↑        Edge   Latency
                                                                        RMSE (m) ↓               sharpness    (ms) ↓
                                                                                                      (→1) 
  ---------------- ------------ ------------ ----------- ------------ ------------ ----------- ----------- ---------
  Nearest                0.1125       0.0476       41.91       0.8872       0.2643       0.782       0.509         1
  neighbour                                                                                                

  Bilinear               0.0964       0.0433       43.38       0.9269       0.2264       0.597       0.408         5

  Bicubic                0.0921       0.0413       43.87       0.9259       0.2157       0.658       0.476         6

  Joint bilateral        0.0986       0.0413       43.07       0.9235       0.2383       0.768       0.463        86
  (JBU)                                                                                                    

  Guided filter          0.0976       0.0446       43.18       0.9264       0.2290       0.746       0.395        26

  Deep learning          0.0690       0.0289       45.92       0.9520       0.1659       0.870       0.698       137
  (EdgeDepthNet)                                                                                           

  Hybrid (fusion +   **0.0650**   **0.0264**   **46.61**   **0.9548**   **0.1567**   **0.880**       0.713       549
  edge-aware                                                                                               
  refinement)                                                                                              
  ------------------------------------------------------------------------------------------------------------------

```{=html}
<!-- /AUTO:overall -->
```
### Results by upsampling factor

Values are averaged across the clean and noisy-plus-missing conditions.

```{=html}
<!-- AUTO:by_scale -->
```
  ----------------------------------------------------------------------------------------------
  Method            RMSE 2×  RMSE 4×  RMSE 8×   B-RMSE   B-RMSE   B-RMSE Edge F1 Edge F1 Edge F1
                                                    2×       4×       8×      2×      4×      8×
  ---------------- -------- -------- -------- -------- -------- -------- ------- ------- -------
  Nearest            0.0627   0.1045   0.1628   0.1589   0.2530   0.3534   0.933   0.845   0.532
  neighbour                                                                              

  Bilinear           0.0548   0.0910   0.1455   0.1403   0.2216   0.3158   0.878   0.636   0.204

  Bicubic            0.0514   0.0857   0.1381   0.1308   0.2074   0.3027   0.917   0.695   0.294

  Joint bilateral    0.0552   0.0911   0.1431   0.1427   0.2291   0.3261   0.918   0.792   0.608
  (JBU)                                                                                  

  Guided filter      0.0577   0.0918   0.1463   0.1481   0.2234   0.3163   0.889   0.761   0.558

  Deep learning      0.0389   0.0610   0.1008   0.0975   0.1495   0.2334   0.948   0.893   0.769
  (EdgeDepthNet)                                                                         

  Hybrid (fusion +   0.0360   0.0577   0.0964   0.0902   0.1417   0.2233   0.954   0.895   0.782
  edge-aware                                                                             
  refinement)                                                                            
  ----------------------------------------------------------------------------------------------

```{=html}
<!-- /AUTO:by_scale -->
```
### Latency and memory

Measurements were taken on a CPU at 320×240 output resolution.

```{=html}
<!-- AUTO:efficiency -->
```
  Method                                      Mean latency at 4× (ms)   Peak memory at 4× (MB, approx.)
  ----------------------------------------- ------------------------- ---------------------------------
  Nearest neighbour                                                 0                               0.6
  Bilinear                                                          0                               0.6
  Bicubic                                                           0                               0.6
  Joint bilateral (JBU)                                            50                               4.8
  Guided filter                                                    21                              16.2
  Deep learning (EdgeDepthNet)                                    153                               2.3
  Hybrid (fusion + edge-aware refinement)                         383                              20.7

```{=html}
<!-- /AUTO:efficiency -->
```
### Key findings

See the [design document](docs/design_document.md#12-results-discussion)
and [ablation study](docs/ablation_study.md) for details.

-   The hybrid method records the lowest RMSE, MAE, and boundary RMSE
    across the reported upsampling factors and conditions, including 5%
    sparse input and the tested unseen sensors (TUM, SUN RGB-D, and
    Redwood). Overall, it reduces RMSE by 29% and boundary RMSE by 27%
    compared with bicubic interpolation.
-   The learned RGB-depth network reduces RMSE by 22% compared with
    bicubic interpolation. CSPN refinement contributes a further 6%
    reduction, and classical edge-aware back-projection contributes
    another 5%.
-   When the network is available, the learned gate selects it almost
    exclusively. Therefore, the cue-driven fusion itself adds accuracy
    mainly in the classical fallback setting, where it improves RMSE by
    3.5% compared with bicubic interpolation.
-   JBU and the guided filter improve edge sharpness, but do not
    outperform bicubic interpolation on RMSE for NYU. This highlights
    that RGB texture edges do not always correspond to depth edges.
-   Evaluated failure modes include less than 1% sparsity, large holes,
    noise ×8, residual texture copying, and RGB-depth misalignment. See
    the [failure analysis](docs/failure_analysis.md).

Detailed per-condition tables, cross-dataset results (TUM RGB-D, SUN
RGB-D, and Redwood), and ablation results are available in
[`docs/results_tables.md`](docs/results_tables.md) and
[`eval_P4_TeamID.md`](eval_P4_TeamID.md).

## Technology stack

Python 3.11 · NumPy · OpenCV · SciPy · scikit-image · PyTorch (CPU) ·
Streamlit · Plotly · Matplotlib · NYU Depth V2 (+ TUM RGB-D, SUN RGB-D,
and Redwood samples) · Docker / Hugging Face Spaces

## Installation

``` bash
git clone <your-repo-url> EdgeDepth
cd EdgeDepth
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The requirements file installs the CPU PyTorch wheels.

## Run the app

Start the Streamlit application:

``` bash
streamlit run app.py
```

Or build and run it with Docker:

``` bash
docker build -t edgedepth .
docker run -p 7860:7860 edgedepth
```

The local Streamlit app is available at `http://localhost:8501`. The
Docker app is available at `http://localhost:7860`.

The repository includes the trained checkpoint
(`checkpoints/edgedepthnet.pt`, approximately 3 MB), fitted fusion
coefficients, 24 unseen test frames, and cross-dataset samples. The app
and evaluation scripts can run using these included assets. If the
checkpoint is removed, the app displays a warning and the hybrid method
switches to its classical fallback.

## Reproduce the results

Run the following commands in order:

``` bash
python scripts/prepare_dataset.py        # Prepare the NYU mirror and scene-disjoint splits (320x240)
python scripts/prepare_samples.py        # Prepare demo samples (requires Open3D sample ZIP files)
python scripts/tune_classical.py         # Tune JBU and guided-filter parameters on validation data (~3 min)
python scripts/train.py --iters 5000     # Train EdgeDepthNet on CPU (~60 min, 2 cores)
python scripts/fit_fusion.py             # Fit hybrid fusion coefficients on validation data (~7 min)
python scripts/evaluate.py               # Evaluate on the test split -> results/summary.json (~5 min)
python scripts/ablation.py               # Run ablation study -> results/ablation.json
python scripts/failure_cases.py          # Evaluate failure cases -> results/failure_cases.json
python scripts/make_report.py            # Regenerate tables in the documentation and evaluation file
python -m pytest -q tests                # Run unit and headless UI tests
python scripts/infer.py --rgb a.png --depth lr.png --out hr.png  # Run command-line inference
```

For GPU users (Colab/Kaggle), training can be run with:

``` bash
python scripts/train.py --device cuda --iters 20000 --batch 16
```

The included checkpoint was trained on CPU only; see
`results/training_log.json`.

## Architecture

``` text
app.py, ui/            Streamlit front-end (6 sections)
edgedepth/
  degradation.py       Observation, low-resolution / sparse / noise / hole simulation
  classical/           Interpolation, filters (JBU, guided), edges, denoising, holes, boundary refinement
  model/               EdgeDepthNet, losses, dataset, inference
  hybrid/fusion.py     Experts, cues, softmax mixture, edge-aware back-projection, fallback
  pipeline.py          run_method / run_methods (+ ablation variants)
  metrics.py           RMSE, MAE, PSNR, SSIM, boundary RMSE, edge F1, sharpness, latency, memory
scripts/                Data preparation, tuning, training, fusion fitting, evaluation, ablation, failure analysis, reporting
docs/                   Design document, architecture (+ Mermaid), ablation, failure analysis, demo, viva
results/                JSON/CSV results and figures produced by the scripts
checkpoints/            edgedepthnet.pt, fusion_params.json
data/                   nyu/test (unseen), samples (demo + cross-dataset)
```

For architecture diagrams, system flow, network details, and fusion
design, see [`docs/architecture.md`](docs/architecture.md).

## Deployment

**Live app:** <https://edgedepth-p4.streamlit.app/> (Streamlit Community
Cloud, free tier, CPU only).

-   The app is deployed from this GitHub repository using `app.py` as
    the main file, Python 3.11, and dependencies from
    `requirements.txt`. Every push to `main` triggers an automatic
    redeployment.
-   No secrets or API keys are required.
-   The checkpoint and images are stored with Git LFS and are fetched
    automatically by Streamlit Cloud.
-   Free-tier apps may sleep after approximately 12 hours without
    visitors. The first visit after a sleep period may take about a
    minute to load.
-   Alternatively, the included Dockerfile can run the same app on a
    Docker host:

``` bash
docker build -t edgedepth .
docker run -p 7860:7860 edgedepth
```

## Limitations

-   The model was trained on 126 frames from six NYU scenes under a
    CPU-only training budget. Generalisation to outdoor LiDAR datasets
    such as KITTI has not been evaluated.
-   The implementation assumes that RGB and depth images are
    pixel-aligned. Misalignment can reduce the performance of all
    RGB-guided methods (failure case F5).
-   Below approximately 1% sparse density, and within large missing
    regions, there may not be enough input evidence to recover the true
    geometry. The network may produce plausible but incorrect surfaces
    (F1, F2).
-   NYU ground truth is itself inpainted in areas where the Kinect
    sensor missed depth. Metrics in these regions therefore measure
    agreement with an inpainted reference, not necessarily the true
    scene geometry.
-   Latency and memory measurements were collected on a 2-core CPU and
    may vary across hardware.

## Ethics

Depth maps of indoor spaces can reveal private room layouts. The app
processes uploaded data in memory and does not store it.

Upsampled depth maps are estimates. Surfaces filled in missing regions
may look plausible without representing the true geometry. Therefore,
the output should not be used for safety-critical tasks, such as robot
navigation or medical or legal measurement, without validation against
real sensor data.

The datasets used are public research datasets and remain subject to
their respective licences: NYU Depth V2, TUM RGB-D, SUN RGB-D, and
Redwood.

## References

-   Kopf et al., *Joint Bilateral Upsampling*, SIGGRAPH 2007.
-   He, Sun, and Tang, *Guided Image Filtering*, TPAMI 2013.
-   Silberman et al., *NYU Depth V2*, ECCV 2012.
-   Cheng, Wang, and Yang, *CSPN*, ECCV 2018.
-   Nguyen, Izadi, and Lovell, *Modeling Kinect Sensor Noise*, 3DIMPVT
    2012.
-   Jacobs et al., *Adaptive Mixtures of Local Experts*, 1991.
