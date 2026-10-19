# Reconstruction notes

Recovered from prior project history:

- Original upload names: `trainingtool_IV4.zip`, later `codes_for_gpt.zip`
- Last known source folder name: `trainingtool_20250809`
- Last known Step4 files:
  - `Step4_1_make_confusion_training.py`
  - `Step4_2_1_make_confusion_inference_master.py`
  - `Step4_2_2_make_confusion_inference.py`
  - `Step4_3_make_confusion_Step3infer.py`
  - `pydealer_detector.py`
  - `pydealer_predict.py`

Known problems addressed in this reconstruction:

1. Logits/loss mismatch: the classifier returns logits and uses `from_logits=True`.
2. Per-split class-index drift: one `class_names.json` controls every stage.
3. `0_nocard` / `54_zerocard`: merged before splitting.
4. BGR/RGB mismatch: pydealer/OpenCV path converts BGR to RGB explicitly.
5. Preprocessing mismatch: train/valid/Step3/Step4 share one implementation.
6. Hard-pool leakage: hashes already present in valid/test are rejected from training.
7. Dataset duplicate leakage: Step1 assigns only the first SHA-1 occurrence.

This file is intentionally explicit so that the reconstructed package is not mistaken for the unrecovered original ZIP.
