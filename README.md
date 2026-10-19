# Inception V4 Training Tool — reconstructed latest version

This package reconstructs the last known project structure from the 2025 work:

- `Step1_split_dataset.py`
- `Step2_train.py`
- `Step3_test_single_image.py`
- `Step4_1_make_confusion_training.py`
- `Step4_2_1_make_confusion_inference_master.py`
- `Step4_2_2_make_confusion_inference.py`
- `Step4_3_make_confusion_Step3infer.py`
- `pydealer_detector.py`
- `pydealer_predict.py`
- `prepare_data.py`, `to_tfrecord.py`, `parse_tfrecord.py`
- `models/inception_v4.py`

## Important status

The original uploaded `codes_for_gpt.zip` / `trainingtool_IV4.zip` binary could not be recovered from File Library. This is therefore a **code-complete reconstruction based on the last known filenames, settings and fixes**, not a byte-for-byte copy of the old ZIP.

## Preserved final decisions

- Python target: **3.10.12**
- TensorFlow 2
- Input: **299 × 299 × 3**
- Batch size: **16**
- Epochs: **2000**
- `Step2_train.py` uses `import configuration as cfg`
- Model head returns logits and the loss uses `from_logits=True`
- Stable class mapping shared by train/valid/test/inference
- `0_nocard` and `54_zerocard` merge into one `0_nocard` class
- latest SavedModel every epoch
- periodic snapshots
- F12 manual-save request
- `inspect_worse_valid` output
- hard-pool is automatically used, but SHA-1 duplicates found in valid/test are excluded
- Step4_2 calls `pydealer_detector.do_predict_one_label_original`
- OpenCV BGR is explicitly converted to RGB
- Step4 reports include CSV/TXT/XLSX confusion matrices, per-class accuracy and copied mistakes

## Folder layout expected before Step1

```text
original_dataset/
├─ 0_nocard/
├─ 1/
├─ 2/
├─ ...
├─ 52/
└─ 54_zerocard/   # optional; merged into 0_nocard
```

## Run order

### 1. Edit paths/settings

Edit `configuration.py` first.

### 2. Split dataset

```bat
python Step1_split_dataset.py
```

Or split and create TFRecords together:

```bat
python prepare_data.py
```

### 3. Train

```bat
python Step2_train.py
```

Outputs:

```text
saved_model/
checkpoints/latest.weights.h5
training_history.csv
snapshots/epoch_XXXX/
inspect_worse_valid/
```

Press **F12** during training to request a manual latest-model save. The save occurs at a safe batch boundary.

### 4. Step3 test

```bat
python Step3_test_single_image.py path\to\image.jpg
python Step3_test_single_image.py path\to\folder
```

### 5. Accuracy reports

Direct SavedModel path:

```bat
python Step4_1_make_confusion_training.py
```

pydealerclient/OpenCV path:

```bat
python Step4_2_1_make_confusion_inference_master.py --workers 1
```

Step3 function path:

```bat
python Step4_3_make_confusion_Step3infer.py
```

Smoke test, first 10 images per class:

```bat
python Step4_1_make_confusion_training.py --smoke-max-per-class 10 --output reports\step4_1_smoke10
python Step4_2_1_make_confusion_inference_master.py --smoke-max-per-class 10 --output reports\step4_2_smoke10
```

Compare two reports:

```bat
python compare_step4_reports.py reports\step4_1_training reports\step4_2_pydealer
```

## Step4 output

Each report folder contains:

```text
predictions.csv
summary.json
confusion_matrix.csv
confusion_matrix_labeled.csv
confusion_matrix_labeled.txt
confusion_matrix.xlsx
confusion_report.txt
per_class_accuracy.csv
wrongly_classified_images/
```

## Why Step4_1 and Step4_2 should now match

Both paths use the same model, class mapping, resize mode and normalization. Step4_2 intentionally starts from an OpenCV BGR image, then converts BGR → RGB before shared preprocessing. This preserves the pydealerclient-style entry point while removing the earlier color-order mismatch.
