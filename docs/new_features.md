# EdgeDepth — Unique features: confidence map, 3-D flying-pixel viewer and automatic misalignment correction

Three features that go beyond producing a depth map. Each answers a weakness we measured in our own failure analysis.

- **Confidence map:** EdgeDepth tells you *where not to trust* its output.
- **3-D flying-pixel viewer:** shows boundary quality as it matters in 3-D.
- **Automatic misalignment correction:** EdgeDepth detects when the colour and depth cameras are out of alignment, and corrects it.

Neither feature needed the network to be retrained. Every number below was measured on the unseen NYU Depth V2
test split by `scripts/eval_new_features.py` (results in `results/new_features.json`).

---

## 1. Confidence map — "where to trust this depth"

### What it does
For every pixel of the hybrid reconstruction, EdgeDepth outputs:
- a **predicted error in centimetres**, and
- a **confidence score** from 0 to 1 (1 = trusted, about 0.37 = expected to be 5 % off, near 0 = a guess).

It is computed **without the ground truth**. It uses only signals the hybrid already produces while reconstructing:

| Signal | Why it indicates error |
| :--- | :--- |
| Disagreement between the four methods | If bicubic, guided filter, JBU and the network disagree, at least some of them are wrong |
| Size of the back-projection correction | Large corrections mean the estimate was far from the measurements |
| Network vs. bicubic difference | Where the network departs strongly from plain interpolation, it is inferring structure |
| Network boundary probability | Errors concentrate at object boundaries |
| Discontinuity probability of the output | Same, from the final depth map |
| Distance to the nearest real measurement | Far from any measurement means filled-in rather than measured |
| Estimated sensor noise | Noisier input means larger errors everywhere |
| Measurement spacing (2×/4×/8×/sparse) | Coarser input means larger errors |
| Missing-measurement indicator | Holes are reconstructed, not measured |

**How the model is fitted:**
- A linear model in these 9 signals predicts the logarithm of the relative error.
- It was fitted by least squares on 1.74 million pixels of the **validation** split (`scripts/fit_confidence.py`, `checkpoints/confidence_params.json`).
- A log-normal mean correction, fitted on the same validation data, makes the predicted errors unbiased in centimetres.
- The test split was never used for fitting.

### How well it works (unseen test split, 168 inputs)

| Uncertainty estimate | AUSE ↓ | Rank correlation with true error ↑ |
| :--- | :---: | :---: |
| **EdgeDepth confidence map** | **0.107** | **0.59** |
| Disagreement between the methods only (naive baseline) | 0.160 | 0.51 |
| Random map (useless) | 0.778 | ≈ 0 |

**AUSE (area under the sparsification error):**
- *The test:* remove pixels from least to most confident, and track the error of what remains.
- *The comparison:* a perfect map (sorting pixels by their true error) is the reference, and AUSE measures how far below it a map falls. Lower is better; 0 means perfect.
- *Result:* EdgeDepth's map is **33 % better than the naive baseline**.

| Least-confident pixels removed | 5 % | 10 % | 15 % | 20 % | 30 % |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Remaining RMSE (fraction of full RMSE) — EdgeDepth | 0.57 | 0.44 | **0.36** | 0.32 | 0.25 |
| Naive baseline | 0.67 | 0.54 | 0.46 | 0.40 | 0.32 |
| Perfect map (oracle) | 0.37 | 0.26 | 0.21 | 0.18 | 0.13 |

**Hiding just the 15 % least-confident pixels removes about two thirds of the RMSE (−64 %).**

| Input type | 2× | 4× | 8× | 5 % sparse |
| :--- | :---: | :---: | :---: | :---: |
| AUSE — EdgeDepth | 0.079 | 0.092 | 0.153 | 0.099 |
| AUSE — naive baseline | 0.120 | 0.152 | 0.224 | 0.129 |

**Calibration:**
- *The check:* we split the test pixels into ten bands by predicted error and compared the average predicted error with the average true error in each band.
- *Result:* they match to within about 0.5 cm in every band, from 0.3 cm vs 0.2 cm in the lowest band to 15.4 cm vs 14.9 cm in the highest.
- *Meaning:* a predicted error of "5 cm" really does mean about 5 cm.

![Confidence evaluation](../results/figures/new/confidence_evaluation.png)

### Why it matters
- **Robotics, AR and 3-D capture:** they need to know which depth values are measured and which are guessed. A robot can slow down near low-confidence regions, and an AR app can avoid placing objects there.
- **Our own failure cases:** large holes, extreme sparsity, very high noise and misalignment are exactly the regions the map flags.

### Limitations
- **Imperfect ranking:** rank correlation is 0.59, not 1. Some errors stay hidden, especially where all methods make the *same* mistake (for example, a wrong but agreed-upon depth inside a large hole).
- **Not a flying-pixel filter:** hiding the least-confident pixels does **not** specifically remove flying pixels (25.5 % → 22.7 % after hiding 10 %). Flying pixels are smaller errors than the large outliers the map ranks first.
- **Hybrid only:** the map is computed for the hybrid output, because it relies on the hybrid's internal signals.

### Where to see it
- **Depth Reconstruction page** (method = Hybrid): the confidence map, predicted error, true error (when ground truth exists), the rank correlation, and this image's sparsification curve. The confidence map can be downloaded as a PNG.
- **Evaluation Dashboard → "Confidence & flying pixels"** tab: the full test-split results.

---

## 2. 3-D flying-pixel viewer and metric

### What a flying pixel is
At an occlusion boundary (for example, a chair in front of a wall), interpolation and smoothing create depths
*between* the two surfaces that belong to neither object. When the depth map is turned into a 3-D point cloud, these
**flying pixels** appear as streaks hanging in the air between foreground and background. They are the most visible
artefact in 3-D reconstruction, AR occlusion and robot mapping.

**Our metric:**
- *Where it looks:* in every 7×7 window where the ground truth has a real depth jump (more than 10 %).
- *What counts as flying:* a predicted pixel that is more than 25 % of the gap away from **both** the near and the far surface.
- *What is reported:* the percentage of pixels in these discontinuity regions that are flying (lower is better). Pixels where the ground truth itself is intermediate are excluded.
- *Without ground truth:* for uploads, an approximate version uses the output itself.

### Results (unseen test split, 168 inputs)

| Method | All inputs | 2× | 4× | 8× | 5 % sparse |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Nearest neighbour | 39.0 % | 26.9 % | 44.8 % | 59.0 % | 11.3 % |
| Bilinear | 57.0 % | 32.0 % | 60.7 % | 83.3 % | 46.7 % |
| **Bicubic (baseline)** | 49.5 % | 26.2 % | 49.6 % | 78.0 % | 39.2 % |
| Joint bilateral | 46.9 % | 22.7 % | 47.2 % | 67.3 % | 54.1 % |
| Guided filter | 58.7 % | 35.2 % | 62.4 % | 83.4 % | 48.6 % |
| Deep learning only | 26.4 % | 12.8 % | 23.0 % | **42.1 %** | 29.2 % |
| **Hybrid (proposed)** | **25.5 %** | **12.0 %** | **22.8 %** | 43.0 % | **22.6 %** |

- **The hybrid halves the flying pixels of bicubic** (−49 % overall).
- **It matches the network alone** on low-resolution input: slightly better at 2× and 4×, slightly worse at 8×.
- **It is clearly better than the network on sparse input** (22.6 % vs 29.2 %).
- **Nearest neighbour on sparse input (11.3 %)** looks good on this metric because it only ever copies real samples, which are never in-between values. Its boundaries are in the wrong place, though, which is why its RMSE is the worst of all methods.
- **We also tested the optional weighted-median "snapping":** it did not reduce flying pixels meaningfully on validation, so it is not presented as a fix.

![Flying pixels per method](../results/figures/new/flying_pixels.png)

### The viewer
App page **"🧊 3-D Flying-Pixel Viewer"**:
- *Layout:* two methods side by side (default: bicubic vs hybrid), optionally with the ground-truth cloud, as rotatable 3-D point clouds.
- *Highlighting:* flying pixels are shown in red, and the flying-pixel percentage is shown for each method.
- *Colouring modes:* flying pixels, true colour, or **confidence** (green = trusted, red = uncertain), which combines both features.
- *Camera:* the default view looks along the object edges from the side, where flying pixels are most visible.

---

## 3. Automatic RGB–depth misalignment correction

### The problem it solves
RGB-guided methods assume the colour image and the depth map line up pixel for pixel. In real devices a calibration
or timing error can shift one against the other by a few pixels. Our failure analysis (case F5) showed that every
RGB-guided method, including the hybrid, then draws depth boundaries in the wrong place; unguided bicubic is
unaffected.

### How it works (`edgedepth/alignment.py`)
1. **RGB-free depth estimate:** build a depth estimate *without* the RGB image (bicubic interpolation, or a smoothed fill for sparse input). Its edges are blurry but sit in the **correct** positions, because no RGB was used.
2. **Edge maps:** compute depth edges (relative depth gradient) and colour edges (Sobel), with the colour edges blurred to match the depth blur.
3. **Search:** try every shift of the colour edges from −10 to +10 px in both directions (441 candidates). For each, compute the normalised cross-correlation with the depth edges. Real object boundaries line up at the correct shift, so the correlation peaks there.
4. **Safety rule:** correct only if the best shift is larger than 1 px **and** beats "no shift" by a clear margin (correlation gain > 0.05). Otherwise the image is left untouched.
5. **Correction:** shift the RGB image back by the estimate. The depth grid defines the output geometry and is never moved.

**Speed:** about 0.1 s per image, with no training. The threshold was chosen on the **validation** split, where it changed none of the 60 correctly aligned inputs.

### Results (unseen test split)
- *Shifted inputs:* 24 test frames × {2×, 4×, 8×, 5 % sparse} × 5 shifts = **480 misaligned inputs**.
- *Aligned controls:* 96 correctly aligned inputs. The hybrid is used throughout.

| Hybrid on misaligned RGB | No correction | **With auto-alignment** | Perfectly aligned (best possible) |
| :--- | :---: | :---: | :---: |
| RMSE (m) | 0.0616 | **0.0575** | 0.0554 |
| Boundary RMSE (m) | 0.1523 | **0.1424** | 0.1370 |
| Edge F1 | 0.881 | **0.894** | 0.903 |

**Auto-alignment recovers 65 % of the boundary error caused by misalignment** (−6.5 % boundary RMSE).

| RGB shift (vertical, horizontal) | 2 px right | 4 px right | 8 px right | 4 px down | 6 px up + 6 px right |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Boundary RMSE: misaligned → corrected (aligned) | 0.1379 → 0.1392 (0.1370) | 0.1531 → 0.1439 | 0.1589 → 0.1399 | 0.1492 → 0.1479 | 0.1624 → 0.1413 |
| Share of the loss recovered | −146 % (slightly worse) | 57 % | **87 %** | 11 % | **83 %** |
| Inputs where a correction was applied | 18 % | 70 % | 100 % | 36 % | 99 % |

| Input type | 2× | 4× | 8× | 5 % sparse |
| :--- | :---: | :---: | :---: | :---: |
| Share of the boundary loss recovered | 80 % | 67 % | 63 % | 65 % |

![Misalignment correction before/after](../results/figures/new/alignment_before_after.png)

### Honest limitations
- **Vertical shifts are poorly corrected** (11 % recovered). The likely reason: indoor scenes have mostly *vertical* depth edges (wall corners, furniture sides, door frames), which pin down horizontal shifts well, while the few horizontal depth edges leave vertical shifts poorly constrained — the classic aperture problem.
- **Tiny shifts (2 px) are not worth correcting.** The estimate is only accurate to about ±1–2 px, and the NYU data itself has about 1 px of native registration offset. So a 2 px correction can slightly overshoot (boundary RMSE 0.1379 → 0.1392). The hybrid is barely affected by a 2 px shift anyway.
- **False corrections on aligned test images:** 20 of the 96 correctly aligned test inputs received a small correction, mostly 2–3 px — none did on validation. The effect was negligible (boundary RMSE 0.1370 → 0.1373, +0.25 %). We report this instead of re-tuning the threshold on the test set.
- **Translation only:** the method estimates a single global shift. Rotation, scale or depth-dependent parallax errors are not modelled.
- **Estimate accuracy:** 59 % of estimates on misaligned test inputs land within 2.5 px of the ideal correction (2×: 83 %, 4×: 63 %, 8×: 42 %, sparse: 48 %). Even approximate corrections recover most of the loss for larger shifts.

### Where to see it
- **Depth Reconstruction** and **Method Comparison** pages: the switch **"Auto-correct RGB–depth misalignment"** is on by default. Open **"Simulate RGB–depth misalignment"** to shift the RGB yourself (for example, 8 px horizontally) and see the detected shift, with a correlation heat-map in **Alignment diagnostics**.
- **Failure Analysis → "Incorrect boundaries (RGB shifted 8 px)"** now includes a **"Hybrid + auto-alignment"** result.
- **Evaluation Dashboard → Unique features:** the full test-split numbers.

### Reproduce
```bash
python scripts/tune_alignment.py     # threshold on VAL               (~12 min, CPU)
python scripts/eval_alignment.py     # before/after on TEST + figure  (~12 min, CPU)
```

### Viva questions on misalignment correction

**Q. How can you estimate the misalignment if the depth is low resolution and you don't know the true geometry?**
We build a depth estimate that uses no RGB at all, so its edges cannot be misaligned. They are only blurry. Then we
find the shift of the colour image whose edges correlate best with those depth edges. Blur spreads the correlation
peak but does not move it.

**Q. How do you avoid "correcting" images that were already aligned?**
Two safety rules: ignore shifts of 1 px or less, and require the best shift to beat "no shift" by a correlation gain
of 0.05, chosen on validation (0 of 60 aligned inputs changed there). On the test set, 20 of 96 aligned inputs still
got small 2–3 px corrections, with a negligible effect (+0.25 % boundary RMSE). We report that openly.

**Q. Why does it fail on vertical shifts?**
Correlation can only measure a shift *across* edges. Indoor scenes are dominated by vertical depth edges, so
horizontal shifts are well constrained and vertical ones are not — the aperture problem from optical flow.

**Q. Why not let the network learn to handle misalignment?**
That would mean retraining with random misalignments. It could make the network more tolerant, but it could not tell
the user that the cameras are miscalibrated. The explicit estimate is interpretable: it reports the shift, which is
also a useful calibration diagnostic.

---

## 4. Reproduce

```bash
python scripts/fit_confidence.py       # fit the confidence model on VAL      (~4 min, CPU)
python scripts/eval_new_features.py    # test-split evaluation + figures      (~3 min, CPU)
python -m pytest -q tests              # includes tests for both features
```

## 5. Viva questions on the confidence map and flying pixels

**Q. How can you predict the error without the ground truth?**
We don't predict the error of a specific pixel from the answer. We learn, on the validation split, how the hybrid's
own internal signals relate to its error: method disagreement, correction size, distance to measurements, edge
probability, noise and spacing. On unseen test images, this relationship ranks pixels by error with a rank
correlation of 0.59 and is calibrated to within about 0.5 cm per band.

**Q. Why fit in log space, and what is the 1.84 correction?**
Errors span several orders of magnitude (sub-millimetre on flat surfaces, tens of centimetres in holes and at edges) and are roughly log-normal, so a linear model of
log-error fits much better. Back-transforming a log prediction gives the *median*, not the mean. Multiplying by
exp(σ²/2) = 1.84, where σ² is the residual variance on validation, gives an unbiased mean. Before this correction,
the predicted errors were about 1.7× too small.

**Q. What is AUSE?**
Area Under the Sparsification Error. Remove pixels in order of decreasing predicted uncertainty and plot the error of
what remains. Do the same with the true error (the oracle). The area between the two curves, normalised by the full
error, is the AUSE. Lower is better, a random map gives about 0.78, and ours gives 0.107.

**Q. Isn't the confidence just the disagreement between methods?**
Disagreement is one of nine signals. On its own it gives an AUSE of 0.160; the fitted combination reaches 0.107,
33 % better. Distance to measurements and the edge probabilities add information disagreement cannot, for example
when all methods agree on a wrong value inside a hole.

**Q. How do you define a flying pixel, and why is it a fair metric?**
A pixel inside a real depth jump (a 7×7 window whose ground-truth range exceeds 10 %) whose predicted depth is more
than 25 % of the gap away from both surfaces. It is computed for all methods on the same inputs, against the same
ground truth, and excludes pixels where the ground truth itself is intermediate.

**Q. Why doesn't hiding low-confidence pixels remove the flying pixels?**
The confidence map ranks pixels by expected error size. The largest errors are outliers in holes and at badly
reconstructed objects. Flying pixels are many moderate errors along edges, so they are not among the first pixels
removed. We report this rather than claim the two features fix the same problem.

## 6. Demo-script addition (≈ 25 s)
- **Depth Reconstruction**, Hybrid, 8× noisy → scroll to the confidence map. *"EdgeDepth also tells you where not to trust it. This map is computed without the ground truth. Next to it is the true error — they match. On unseen test images, hiding just 15 % of the least-confident pixels removes two thirds of the error."*
- **3-D Flying-Pixel Viewer** → Build, then rotate. *"Interpolation leaves depths hanging between objects — the red streaks. Our hybrid halves them, from 49.5 % to 25.5 % on the test set."*
- **Depth Reconstruction** → open *Simulate RGB–depth misalignment*, set the horizontal shift to 8 px → Run. *"Real cameras drift out of calibration. EdgeDepth detects the shift by itself and corrects it — on the test set this recovers 65 % of the boundary error that misalignment causes."*
