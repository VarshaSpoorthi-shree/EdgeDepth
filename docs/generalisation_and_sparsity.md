# Generalisation beyond NYU and behaviour on extremely sparse input

Two follow-up studies, added after reviewer feedback. Nothing in them was used to train or tune the reconstruction methods. Every number below is read from `results/cross_dataset_eval.json` and `results/sparse_confidence.json`.

## 1. Cross-dataset generalisation

**Data.** 133 frames from 5 scenes outside NYU Depth V2, fetched by `scripts/fetch_cross_dataset.py` from the Open3D sample-data releases on GitHub (the original dataset servers, Hugging Face, TUM and SUN RGB-D were unreachable from our build environment). All frames are resized to 320×240, like the NYU frames.

| Source | Frames | Type | Reference used for scoring |
|---|---|---|---|
| ICL-NUIM living room | 25 | synthetic, exact GT | exact rendered depth (no sensor noise) |
| ICL-NUIM office | 25 | synthetic, exact GT | exact rendered depth (no sensor noise) |
| Stanford Lounge | 25 | real sensor | raw sensor depth (holes excluded; its noise counts as error) |
| Redwood Bedroom | 25 | real sensor | raw sensor depth (holes excluded; its noise counts as error) |
| Stanford Fountain | 33 | real sensor | raw sensor depth (holes excluded; its noise counts as error) |

Frames are evenly spaced through each sequence. The Augmented ICL-NUIM scenes are photorealistic renders (Choi et al.); they matter because they come with exact, hole-free ground truth and a simulated depth-sensor stream.

**Protocol.** Same methods and code as the NYU benchmark.

* Synthetic scenes: exactly the NYU test conditions and seeds (2×/4×/8× × clean / noisy+missing, plus 5 % sparse), plus *simulated sensor* input (the noisy, holey simulated depth downsampled 4× and 8×, scored against the exact GT).
* Real-sensor scenes: the raw depth box-downsampled 2×/4×/8× (holes ignored), plus 5 % sparse samples.
* Per source, a paired per-frame comparison: relative RMSE change of the hybrid vs bicubic and vs the network alone, with a bootstrap 95 % interval over frames and the share of frames where the hybrid wins.

### Results

RMSE (m), mean over all conditions of that source:

| Source | Bicubic | Guided filter | Network only | **Hybrid** | Hybrid vs bicubic | Hybrid vs network (95 % CI) | Hybrid wins vs network |
|---|---|---|---|---|---|---|---|
| ICL-NUIM living room | 0.0611 | 0.0609 | 0.0453 | **0.0429** | -28.4 % | -5.5 % [-6.2, -4.7] | 100 % of 25 |
| ICL-NUIM office | 0.0994 | 0.0990 | 0.0755 | **0.0723** | -25.1 % | -5.4 % [-7.0, -4.1] | 100 % of 25 |
| Stanford Lounge | 0.0622 | 0.0638 | 0.0487 | **0.0462** | -27.8 % | -5.7 % [-6.5, -4.9] | 100 % of 25 |
| Redwood Bedroom | 0.0394 | 0.0411 | 0.0313 | **0.0298** | -24.0 % | -5.0 % [-5.8, -4.2] | 100 % of 25 |
| Stanford Fountain | 0.0263 | 0.0276 | 0.0211 | **0.0198** | -24.2 % | -6.1 % [-6.9, -5.3] | 100 % of 33 |

Boundary RMSE (m) and edge F1:

| Source | B-RMSE bicubic | B-RMSE network | **B-RMSE hybrid** | Edge F1 bicubic | Edge F1 network | **Edge F1 hybrid** |
|---|---|---|---|---|---|---|
| ICL-NUIM living room | 0.2297 | 0.1650 | **0.1586** | 0.576 | 0.866 | **0.873** |
| ICL-NUIM office | 0.3967 | 0.3095 | **0.2975** | 0.635 | 0.862 | **0.869** |
| Stanford Lounge | 0.1746 | 0.1390 | **0.1333** | 0.703 | 0.919 | **0.931** |
| Redwood Bedroom | 0.1138 | 0.0928 | **0.0900** | 0.666 | 0.863 | **0.876** |
| Stanford Fountain | 0.1160 | 0.0959 | **0.0924** | 0.701 | 0.897 | **0.906** |

Synthetic scenes by condition, RMSE (m) — directly comparable with the NYU test tables:

| Condition | Bicubic | Guided filter | Network only | **Hybrid** |
|---|---|---|---|---|
| 2x/clean | 0.0380 | 0.0403 | 0.0284 | **0.0264** |
| 2x/noisy missing | 0.0596 | 0.0576 | 0.0434 | **0.0414** |
| 4x/clean | 0.0568 | 0.0591 | 0.0411 | **0.0391** |
| 4x/noisy missing | 0.0862 | 0.0855 | 0.0589 | **0.0567** |
| 8x/clean | 0.0820 | 0.0854 | 0.0601 | **0.0563** |
| 8x/noisy missing | 0.1282 | 0.1278 | 0.0939 | **0.0892** |
| sparse/sparse 5pct | 0.0909 | 0.0793 | 0.0685 | **0.0643** |
| 4x/sim sensor | 0.0811 | 0.0825 | 0.0682 | **0.0670** |
| 8x/sim sensor | 0.0996 | 0.1021 | 0.0809 | **0.0783** |

![Cross-dataset results](../results/figures/new/cross_dataset.png)

### Confidence map on unseen data

| Source | AUSE (lower is better) | AUSE random | Rank correlation | Predicted / true error |
|---|---|---|---|---|
| ICL-NUIM living room | 0.095 | 0.788 | 0.56 | 1.11 |
| ICL-NUIM office | 0.073 | 0.809 | 0.56 | 1.00 |
| Stanford Lounge | 0.052 | 0.828 | 0.59 | 0.98 |
| Redwood Bedroom | 0.075 | 0.816 | 0.57 | 1.00 |
| Stanford Fountain | 0.067 | 0.799 | 0.46 | 1.15 |

On the NYU test split the AUSE was 0.107 and the rank correlation 0.59, so the map transfers. One exception: with the *simulated sensor* input scored against exact GT, it still ranks errors (AUSE 0.146) but predicts only 0.53× the true error size. Sensor-style noise and holes it was not fitted on make it over-confident in absolute terms.

### What this shows, and what it does not

* The ranking of methods found on NYU holds on every unseen scene: hybrid < network < classical methods in RMSE, boundary RMSE and edge F1.
* The hybrid's gain over the network alone is small (about 5–6 % RMSE, 3–4 % boundary RMSE) but consistent: it wins on 96–100 % of frames, and every bootstrap interval excludes zero.
* Absolute numbers differ between sources because depth ranges differ (Fountain is a close-range scene at roughly 0.8–1.7 m; ICL office has long depth jumps). Compare methods within a row, not across rows.
* For real-sensor scenes the reference is itself noisy and has holes. Errors there combine reconstruction error and sensor noise.
* All scenes are indoor or close-range and captured with structured-light-type sensors or simulations of them. Outdoor LiDAR (KITTI) and time-of-flight sensors are still not tested; the KITTI and Middlebury servers were unreachable.
* TUM RGB-D and SUN RGB-D are represented by only one frame each (the quick check in `data/samples`), because only single sample frames were reachable.

## 2. Extremely sparse input

**Protocol.** Random samples at 10 %, 5 %, 2 %, 1 %, 0.5 % and 0.2 % of the 320×240 pixels, on **all** 24 test frames (the earlier F1 table used 8). The same was run on the 29 validation frames, but only to choose the app's warning threshold. The confidence model was not refitted: it was fitted at 5 % density and above, so lower densities test extrapolation. An *unusable* image is one whose hybrid RMSE exceeds 0.15 m; an *unreliable* pixel has error above 10 cm.

| Density | Samples per image | Bicubic RMSE | Network RMSE | **Hybrid RMSE** | Unusable images | Pixels with error > 10 cm | Confidence rank correlation | Predicted / true error |
|---|---|---|---|---|---|---|---|---|
| 10% | 7674 | 0.072 | 0.066 | **0.058** | 0.0 % | 3.3 % | 0.57 | 0.97 |
| 5% | 3840 | 0.095 | 0.083 | **0.075** | 12.5 % | 5.0 % | 0.59 | 0.91 |
| 2% | 1539 | 0.136 | 0.111 | **0.103** | 33.3 % | 8.4 % | 0.59 | 0.84 |
| 1% | 763 | 0.178 | 0.135 | **0.127** | 33.3 % | 12.3 % | 0.60 | 0.79 |
| 0.5% | 380 | 0.227 | 0.171 | **0.164** | 37.5 % | 18.2 % | 0.59 | 0.68 |
| 0.2% | 150 | 0.295 | 0.239 | **0.234** | 58.3 % | 31.9 % | 0.49 | 0.42 |

Mean absolute error (m) by distance to the nearest measurement, test split (true / predicted):

| Density | 0-2 | 2-4 | 4-8 | 8-16 | 16-32 | >=32 px |
|---|---|---|---|---|---|---|
| 10% | 0.015 / 0.016 | 0.025 / 0.019 | 0.038 / 0.025 | — | — | — |
| 5% | 0.017 / 0.019 | 0.030 / 0.023 | 0.040 / 0.027 | 0.090 / 0.038 | — | — |
| 2% | 0.020 / 0.026 | 0.036 / 0.029 | 0.051 / 0.034 | 0.073 / 0.043 | 0.054 / 0.040 | — |
| 1% | 0.022 / 0.032 | 0.041 / 0.034 | 0.061 / 0.040 | 0.080 / 0.048 | 0.112 / 0.042 | — |
| 0.5% | 0.023 / 0.038 | 0.046 / 0.041 | 0.071 / 0.046 | 0.100 / 0.055 | 0.141 / 0.046 | 0.065 / 0.016 |
| 0.2% | 0.025 / 0.039 | 0.053 / 0.042 | 0.087 / 0.048 | 0.132 / 0.056 | 0.198 / 0.045 | 0.180 / 0.045 |

![Sparse inputs and the confidence map](../results/figures/new/sparse_confidence.png)

### Findings

* **Where it becomes unreliable.** The hybrid stays the most accurate method at every density, but the mean test RMSE passes 0.15 m between 1 % (0.127 m) and 0.5 % (0.164 m). The share of unusable test images rises from 0 % at 10 % density to 58 % at 0.2 %. Error grows steadily with distance from the nearest measurement: at 0.2 % density it is about 2.5 cm next to a sample and about 20 cm at 16–32 px.
* **The confidence map keeps ranking errors** (rank correlation about 0.59 from 10 % down to 0.5 %, 0.49 at 0.2 %), so it still marks the right regions as least trustworthy.
* **But it underestimates their size at low density.** Predicted/true mean error falls from 0.97 at 10 % to 0.79 at 1 % and 0.43 at 0.2 %. Its prediction stops growing beyond about 8–16 px from a sample, while the true error keeps rising. At very low density the map should be read as relative, not absolute.
* **Image-level warning (added to the app).** The Reconstruction page now warns when the mean predicted error exceeds 3.6 cm. This threshold was chosen on the validation split (balanced accuracy 0.90 there). On the test split it flags 41 of 42 unusable reconstructions, with precision 85 % and a false-alarm rate of 7 %. Validation had fewer unusable images (17 of 174), so its precision was lower (42 %).
* The default app inputs (4× low resolution, and 5 % sparse) do not trigger the warning.

### Possible improvement (not done)

Refit the confidence model on validation data that also includes 2 %, 1 %, 0.5 % and 0.2 % density, so that its predicted error can keep growing with distance from the samples, as the true error does. This would likely reduce the under-estimation. It would change the confidence results reported earlier, so it should be followed by a fresh test-split evaluation.

## Reproduce

```bash
python scripts/fetch_cross_dataset.py     # ~1 min, 133 frames -> data/cross (not committed)
python scripts/eval_cross_dataset.py      # took 31 min on 2 CPU cores (while sharing them with the next script)
python scripts/eval_sparse_confidence.py  # took ~25 min on 2 CPU cores (same conditions)
```
