# Stage 2 Overview

Stage 2 consumes Stage 1 landmark CSVs and exports the trained runtime package for Stage 3.

## Input

```text
Feature_Data/*_landmarks.csv
Video_Data/*.{mp4,wav,aif,aiff,flac,mp3,m4a,mov}
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

## Audio Target

The full AV training config expects per-window audio targets:

```text
audio_schema: log_mel_12
audio_dim: 12
audio_quality: 1.0 when decoded/aligned, 0.0 when missing
```

`build_windows.py --audio-mode log_mel` uses `ffmpeg` to decode mono 16 kHz audio from the source video/audio file, computes 12 normalized log-mel bands, and averages them over each motion window's timestamp span. Audio is used only as a training target; Stage 3 runtime inference uses motion landmarks only.

## Output

```text
Window_Data/*.npz
Window_Data/window_manifest.csv
Window_Data/window_summary.json
Model_Outputs/<run_name>/export_for_runtime/
```

Stage 3 should set `runtime_model.artifact_dir` to `Model_Outputs/<run_name>/export_for_runtime` so live features match the training features.
