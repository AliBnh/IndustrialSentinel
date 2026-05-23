# Engineering Decisions & Lessons Learned

This document records key design decisions, failures encountered, and pivots made during development. It exists to demonstrate engineering judgment and intellectual honesty — not every first attempt works, and recognizing why is more valuable than pretending it did.

---

## 1. Row-Level vs Engine-Level Train/Val Split

**Initial approach:** Random 85/15 row-level split for XGBoost and LSTM validation.

**What happened:** Validation RMSE was suspiciously low (~11). The model appeared to generalize perfectly.

**Root cause:** Temporal data leakage. When you split rows randomly, cycle 150 of engine 5 might be in training while cycle 151 is in validation. The model memorizes the trajectory rather than learning degradation patterns.

**Fix:** Split at the engine level — 85 engines for training, 15 entire engines for validation. Validation RMSE jumped to ~16 (honest), and test RMSE was 14.14 (genuine generalization).

**Lesson:** In time-series problems, always split by the entity (engine, patient, user), never by individual observations.

---

## 2. Anomaly Threshold: Mean+2σ vs 95th Percentile

**Initial approach:** Set the autoencoder anomaly threshold as `mean + 2*std` of reconstruction errors on healthy data.

**What happened:** A few noisy sequences in the healthy training set had disproportionately high reconstruction errors, pulling the mean and std up. This made the threshold too lenient — real anomalies were missed.

**Fix:** Use the 95th percentile of healthy reconstruction errors instead. This is robust to outliers because it's based on rank, not magnitude.

**Result:** Threshold of 0.00501 — correctly flags degrading engines while keeping false alert rate at 0%.

---

## 3. Ensemble Weights: Equal vs Performance-Based

**Initial approach:** 50/50 equal weighting of XGBoost and LSTM.

**What happened:** Ensemble RMSE was 15.2 — barely better than LSTM alone (15.04). The XGBoost's higher error (17.97) was dragging the ensemble down.

**Analysis:** LSTM dominates on critical-zone predictions (RMSE 3.69 vs 15.61 for XGBoost in the RUL<30 zone). XGBoost is only useful for stability on well-separated healthy engines.

**Fix:** Shifted to 0.3/0.7 (XGB/LSTM) weighting based on empirical performance gap. Ensemble RMSE dropped to 14.14.

**Lesson:** Ensemble weights should reflect actual model strengths, not naive equality.

---

## 4. RUL Clipping: Why 125?

**Considered alternatives:** No clipping, clip at 100, clip at 150.

**No clipping problem:** The model wastes capacity trying to distinguish between RUL=300 and RUL=200 — both are "healthy" and the sensor readings are nearly identical. This adds noise to the loss function.

**Clip at 100 problem:** Some engines show early degradation signals around cycle 100-125. Clipping too aggressively loses this information.

**Decision:** Clip at 125. This is supported by domain literature (Heimes 2008, Zheng 2017) and empirically validated — the LSTM training loss converges faster with clipping at 125 than at 100 or 150.

---

## 5. Feature Engineering: Why Not Just Raw Sensors?

**Tested:** Training LSTM on raw 14 sensors (no rolling features).

**Result:** Test RMSE was ~19. The model struggled to capture degradation trends from noisy single-cycle readings.

**Fix:** Added rolling statistics (mean, std) at windows [5, 10, 15] and lag features. This gives the model explicit trend information — the rolling std captures increasing volatility as engines degrade.

**Result:** 126 features, test RMSE dropped to 14.14. The rolling std features ranked in the top 10 by XGBoost feature importance.

---

## 6. Docker: CPU-Only PyTorch

**Problem:** Full PyTorch with CUDA support is ~2GB. Combined with XGBoost's nvidia-nccl dependency (~300MB), Docker images were 3GB+.

**Decision:** Use `--index-url https://download.pytorch.org/whl/cpu` for CPU-only PyTorch (~200MB). Training takes ~10 minutes on CPU which is acceptable for a demo system.

**Tradeoff:** If deploying at scale with GPU inference, you'd switch to the CUDA build. For this project's scope, CPU keeps images practical and CI fast.

---

## 7. API Statelessness: Dependency Injection vs Global State

**Initial approach:** Load models as module-level globals in `api/main.py`.

**Problem:** This makes testing difficult (can't mock models), creates import-time side effects, and violates the principle of explicit dependencies.

**Fix:** Used FastAPI's `Depends()` system with a singleton `ModelContainer`. Models are loaded once on first request, injected into handlers, and easily mockable in tests.

---

## Summary

| Decision | Naive Approach | What Went Wrong | Final Approach |
|----------|---------------|-----------------|----------------|
| Data split | Row-level random | Temporal leakage, RMSE=11 (fake) | Engine-level split |
| AE threshold | Mean + 2σ | Outlier-sensitive | 95th percentile |
| Ensemble weights | 50/50 | XGBoost dragging down | 30/70 based on performance |
| RUL target | Raw (no clip) | Noisy high-RUL predictions | Clip at 125 |
| Features | Raw sensors only | RMSE=19 | Rolling + lag (126 features) |
| Docker | Full CUDA PyTorch | 3GB images | CPU-only (~200MB) |
| Model loading | Global variables | Untestable | Dependency injection |
