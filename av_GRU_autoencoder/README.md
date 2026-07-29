# av_GRU_autoencoder

Stage 2 for the audiovisual autoencoder system.

This folder converts Stage 1 landmark CSVs into fixed temporal windows, trains the GRU autoencoder, embeds windows into latent gesture vectors, clusters those vectors, and exports the model package consumed by Stage 3.

## Folder Structure

```text
av_GRU_autoencoder/
├── README.md
├── build_windows.py
├── train.py
├── embed.py
├── cluster.py
├── evaluate.py
├── export_for_runtime.py
├── plot_metrics.py
├── configs/
│   ├── small_test.yaml
│   ├── motion_only.yaml
│   ├── default.yaml
│   ├── full_av.yaml
│   ├── latent_16.yaml
│   ├── latent_32.yaml
│   └── gru_hidden_256.yaml
├── docs/
│   └── STAGE_OVERVIEW.md
├── models/
│   └── av_gru_autoencoder.py
├── src/
│   ├── config.py
│   ├── dataset.py
│   └── losses.py
├── requirements.txt
└── requirements_frozen.txt
```

Recommended sibling data layout:

```text
av_autoencoder/
├── Feature_Data/
├── Window_Data/
├── Model_Outputs/
├── av_Camera_Calibration_Preprocess/
├── av_GRU_autoencoder/
└── av_Gesture_OSC_runtime/
```

## Setup

```bash
python -m venv ~/venvs/av_GRU_env
source ~/venvs/av_GRU_env/bin/activate

cd "/Volumes/MP_1/GSoC 2026/av_autoencoder/av_GRU_autoencoder"
pip install -r requirements.txt
python -c "import torch, numpy, pandas, sklearn; print('Stage 2 OK')"
```

## Feature Contract

Stage 1 CSVs contain detection-order hands (`hand0`, `hand1`) plus MediaPipe handedness labels. Stage 2 converts each frame to the canonical model input:

```text
right hand 63 + left hand 63 = 126 motion features
```

By default, `build_windows.py` also normalizes each hand relative to its wrist and scales it by landmark 9. This is the same transform Stage 3 applies live when a newly exported `runtime_model_config.json` says:

```json
{
  "include_velocity": false,
  "normalize_to_wrist": true,
  "normalize_scale_landmark": 9,
  "hand_order": "label",
  "feature_order": "right_63_left_63"
}
```

If you are using CSVs created before `hand0_label` and `hand1_label` existed, the builder falls back to `hand0=right` and `hand1=left`. Re-extract landmarks when two-hand identity matters.

## Build Windows

Motion-only placeholder windows:

```bash
python build_windows.py \
  --input-dir "../Feature_Data" \
  --output-dir "../Window_Data" \
  --sequence-length 60 \
  --hop-length 30 \
  --audio-dim 1 \
  --audio-quality 0.0 \
  --overwrite
```

This creates:

```text
Window_Data/
├── ses01_cam01_vid01_win000000.npz
├── ses01_cam01_vid01_win000001.npz
├── window_manifest.csv
└── window_summary.json
```

Each `.npz` contains:

```text
motion             shape [60, 126]
audio              shape [audio_dim]
audio_quality      scalar
feature_contract   JSON string
```

For the current placeholder pipeline, use `configs/small_test.yaml` or `configs/motion_only.yaml`. The `default.yaml` and `full_av.yaml` configs expect `audio_dim: 12`; use those only after building windows with matching 12-D audio placeholders or real audio features.

### Responsive 15-Frame Small Test

For a more responsive live model, rebuild windows at the shorter sequence length before training:

```bash
python build_windows.py \
  --input-dir "../Feature_Data" \
  --output-dir "../Window_Data_seq15" \
  --sequence-length 15 \
  --hop-length 1 \
  --audio-dim 1 \
  --audio-quality 0.0 \
  --overwrite
```

Then train, embed, cluster, and export with the matching config:

```bash
python train.py --config configs/small_test_seq15.yaml
python embed.py --config configs/small_test_seq15.yaml
python cluster.py --config configs/small_test_seq15.yaml --method hdbscan --min_cluster_size 10
python export_for_runtime.py --config configs/small_test_seq15.yaml
```

Do not point `configs/small_test_seq15.yaml` at the original `Window_Data` folder; `train.py` validates that the saved window shape matches `features.sequence_length`.

## Train

Small smoke test:

```bash
python train.py --config configs/small_test.yaml
```

Motion-only training:

```bash
python train.py --config configs/motion_only.yaml
```

The trainer now validates the first window against the config. If `motion_dim`, `audio_dim`, or `sequence_length` do not match, it fails before training.

Outputs are written under the config's `data.output_dir`, for example:

```text
Model_Outputs/small_test/
├── checkpoints/best_model.pt
├── scalers/feature_scaler.joblib
├── metrics/training_history.json
├── train_split.csv
├── val_split.csv
└── test_split.csv
```

## Embed And Cluster

```bash
python embed.py --config configs/small_test.yaml

python cluster.py \
  --config configs/small_test.yaml \
  --method hdbscan \
  --min_cluster_size 10
```

Outputs:

```text
Model_Outputs/small_test/
├── embeddings/
│   ├── embeddings.npy
│   └── embeddings.csv
└── clustering/
    ├── cluster_assignments.csv
    ├── cluster_model.joblib
    ├── embedding_scaler.joblib
    └── umap_model.joblib
```

## Evaluate

```bash
python evaluate.py --config configs/small_test.yaml
```

Output:

```text
Model_Outputs/small_test/reports/evaluation_report.json
```

## Export For Stage 3

Run the export with the same config used for training:

```bash
python export_for_runtime.py --config configs/small_test.yaml
```

Exported package:

```text
Model_Outputs/small_test/export_for_runtime/
├── encoder.pt
├── av_gru_encoder.pt
├── feature_scaler.joblib
├── embedding_scaler.joblib
├── cluster_model.joblib
├── cluster_names.json
├── runtime_model_config.json
└── runtime_export_manifest.json
```

Point Stage 3 at that folder:

```yaml
runtime_model:
  artifact_dir: "../Model_Outputs/small_test/export_for_runtime"
```

`runtime_model_config.json` carries the feature contract, sequence length, and GRU architecture. Stage 3 reads it and overrides local fallback settings so live input matches training.

## End-To-End Order

```text
Stage 1
raw video -> undistorted MediaPipe CSVs with handedness labels

Stage 2
CSV landmarks -> canonical windows -> train -> embed -> cluster -> export

Stage 3
camera/video -> optional undistortion -> MediaPipe -> same feature contract -> OSC
```
