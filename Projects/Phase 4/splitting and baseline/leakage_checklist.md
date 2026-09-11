# Leakage-Detector Checklist

- [ ] Split done BEFORE any preprocessing/scaling
- [ ] Scaler/normalizer fit ONLY on train, then applied to val/test
- [ ] No duplicate rows across train/val/test
- [ ] No feature that's a disguised version of the target (proxy label)
- [ ] No feature containing info not available at prediction time
- [ ] If time-series data: split chronologically, not randomly shuffled
- [ ] Random seed fixed for reproducibility