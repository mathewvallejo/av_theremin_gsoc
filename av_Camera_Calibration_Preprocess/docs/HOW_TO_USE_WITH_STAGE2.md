# Connecting Stage 1 to Stage 2

Stage 2 (`av_GRU_autoencoder`) expects pre-windowed `.npz` files containing `motion`, `audio`, and `audio_quality` arrays. Stage 1 produces the raw landmark CSV files that feed into that windowing step.

## What Stage 1 outputs

```text
Feature_Data/
├── ses01_cam01_vid01_landmarks.csv
├── ses01_cam01_vid02_landmarks.csv
└── ...
```

Each CSV has one row per video frame with columns:
- `frame`, `timestamp_ms`, `num_hands`
- `hand0_0` … `hand0_62` — right hand xyz landmarks (21 × 3 = 63 values)
- `hand1_0` … `hand1_62` — left hand xyz landmarks

Frames with no detected hand have `NaN` in those columns.

## Windowing step (your responsibility)

Stage 2 does not do windowing internally. You need an intermediate step that:

1. Reads the landmark CSVs from Stage 1.
2. Extracts audio features for the same time range (e.g. MFCCs).
3. Slices into overlapping windows of `sequence_length` frames.
4. Saves each window as a `.npz` with keys `motion`, `audio`, `audio_quality`.
5. Optionally writes a `window_manifest.csv`.

Then point Stage 2's config at:

```yaml
data:
  dataset_dir: ../Feature_Data/windowed_sequences
  manifest_csv: ../Feature_Data/window_manifest.csv
```

## Calibration file for Stage 3

The same `.npz` calibration files produced in Step B of Stage 1 should be copied to Stage 3's `models/` folder so the live camera is undistorted with the same parameters used during training.
