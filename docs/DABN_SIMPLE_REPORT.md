# DABN Implementation and Test Summary

## 1) What DABN is supposed to do
DABN (Domain Adaptive Batch Normalization) is meant to reduce domain shift between training and testing data.

In simple terms:
- Keep model inference mode for normal layers.
- Let BatchNorm layers adapt their running mean/variance using test-domain inputs.
- Expected outcome: better cross-dataset generalization.

## 2) How we implemented it here
We added DABN-style BN control for EfficientNet and Xception.

Code updates:
- `backend/models/wrappers/efficientnet.py`
  - Added `set_bn_adaptive_mode(model, enabled, momentum)`.
  - Model loads in eval mode; DABN is now controlled at runtime.
- `backend/models/wrappers/xception.py`
  - Added `set_bn_adaptive_mode(model, enabled, momentum)`.
- `backend/handlers/facial_analyzer.py`
  - Added `get_dabn_settings(model_cfg, default_enabled)`.
  - Applies DABN settings before EfficientNet/Xception inference.
- `backend/config/ensemble.yaml`
  - Added/used model-level DABN config blocks (`enabled`, `momentum`).

## 3) How we tested it
We ran A/B evaluations on the same canonical FakeAVCeleb slice (4,000 videos):
- Categories: `RealVideo-RealAudio`, `RealVideo-FakeAudio`, `FakeVideo-FakeAudio`, `FakeVideo-RealAudio`
- Same file order and same code, only DABN mode changed.

Run modes:
- `off`: DABN disabled for EfficientNet and Xception
- `efficientnet_only`: DABN on EfficientNet only
- `both`: DABN on EfficientNet and Xception

Evaluation script:
- `/tmp/run_ab_fakeav_eval_v2.py`

Logs:
- `/tmp/ab_v2_off.log`
- `/tmp/ab_v2_effonly.log`
- `/tmp/ab_v2_both.log`

## 4) What actually happened
### Ensemble accuracy (4,000-video slice)
- `off`: `0.0900` (360/4000)
- `efficientnet_only`: `0.0508` (203/4000)
- `both`: `0.0190` (76/4000)

### Observed behavior
- Turning DABN on reduced accuracy in this branch setup.
- Xception especially collapsed when DABN was enabled (`both` run showed very poor behavior).
- AASIST stayed unchanged across runs (DABN was only applied to video models).

## 5) Why this likely happened
Most likely causes in this code path:
- BN adaptation is happening online with very small per-video face batches, which can make BN stats noisy/unstable.
- This branch’s fusion/threshold setup is sensitive, so score distribution shifts from DABN can hurt voting outcomes.
- DABN assumptions (stable target-domain stats, enough adaptation data) are not well matched by per-video micro-batches.

## 6) Current conclusion
In this branch and current evaluation setup, DABN did **not** improve results.

Recommended default for now:
- Keep DABN **off** unless we add a safer adaptation strategy (e.g., calibration pass and threshold recalibration).
