# Stage 2 Overview

Stage 2 consumes Stage 1 landmark CSVs and exports the trained runtime package for Stage 3.

## Input

```text
Feature_Data/*_landmarks.csv
```

Expected CSV schema:

```text
hand0_label, hand0_score, hand0_0 ... hand0_62
hand1_label, hand1_score, hand1_0 ... hand1_62
```

## Canonical Motion Feature

`build_windows.py` converts detection-order hands into this model input:

```text
right hand 63 + left hand 63 = 126
```

Default transform:

```text
hand_order: label
normalize_to_wrist: true
normalize_scale_landmark: 9
include_velocity: false
```

## Output

```text
Window_Data/*.npz
Window_Data/window_manifest.csv
Window_Data/window_summary.json
Model_Outputs/<run_name>/export_for_runtime/
```

Stage 3 should set `runtime_model.artifact_dir` to `Model_Outputs/<run_name>/export_for_runtime` so live features match the training features.
