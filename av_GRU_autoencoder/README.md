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
│   ├── small_test_seq15.yaml
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
ffmpeg -version
```

## Recommended Workflow

Start with `configs/small_test.yaml` to confirm that window building, training, embedding, clustering, and export all run on your machine. It is intentionally motion-only and short, so it is the fastest way to catch path, dependency, and feature-shape problems.

After that sanity check is successful, users can move to any other config. The `configs/full_av.yaml` config is the intended full-capability model path: motion is still the live runtime input, but audio is included during training as an auxiliary target so the latent gesture space can learn audiovisual structure.

Use the same config for every Stage 2 step in a run:

```text
build matching windows -> train -> embed -> cluster -> evaluate -> export
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

## Audio Target Contract

The full AV configs use a concrete audio target:

```text
audio_schema: log_mel_12
audio shape: [12]
audio[0] = lowest mel-frequency band energy
audio[11] = highest mel-frequency band energy
```

`build_windows.py --audio-mode log_mel` reads the source video/audio file, converts audio to mono 16 kHz with `ffmpeg`, computes 12 normalized log-mel bands, and averages those bands over the same timestamp span as the motion window. The result is stored as the window's `audio` target.

`audio_quality` is extraction/sync confidence, not loudness. A quiet or silent window can still have `audio_quality: 1.0` if the audio was decoded and aligned correctly. Missing or unreadable audio should be `0.0`, and `full_av.yaml` rejects windows below its `audio_quality_threshold`.

Stage 3 does not require live audio. Runtime inference still uses motion landmarks only; audio is an auxiliary target used during training to shape the latent gesture space.

## Window Length And Realtime Latency

The `features.sequence_length` value is both the offline training window length and the live Stage 3 rolling-buffer length. A 60-frame model gives the GRU more temporal context, but it also means Stage 3 cannot produce the first model-based gesture until 60 valid hand frames have filled the rolling window.

Approximate warm-up latency:

```text
60 frames at 30 FPS = about 2.0 seconds
60 frames at 60 FPS = about 1.0 second
15 frames at 30 FPS = about 0.5 seconds
```

After the rolling window is full, Stage 3 advances the buffer one frame at a time and can run inference every frame. So the 60-frame setting does not force predictions to update only once per 60 frames. The tradeoff is that each prediction summarizes the last 60 frames, so abrupt gesture changes can feel smeared by the longer temporal context.

The offline `--hop-length` setting only controls how many training windows are generated from recorded CSVs. It does not control the live inference rate.

Gesture selection also has a separate smoothing layer in Stage 3: `smoothing.cluster_history` is a majority vote over recent cluster predictions. Lowering that value can make cluster changes more responsive, while increasing it makes them steadier. For live interaction, tune both the trained `sequence_length` and the runtime smoothing history; do not train a 60-frame model and then run it with a 15-frame runtime window.

## Build Windows

### Small Test Windows

Build motion-only windows for the first smoke test:

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

This matches `configs/small_test.yaml`:

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

### Full AV Windows

`configs/full_av.yaml` expects 60-frame motion windows with `audio_schema: log_mel_12`:

```bash
python build_windows.py \
  --input-dir "../Feature_Data" \
  --output-dir "../Window_Data" \
  --sequence-length 60 \
  --hop-length 30 \
  --audio-mode log_mel \
  --audio-dim 12 \
  --audio-source-dir "../Video_Data" \
  --overwrite
```

This intentionally rebuilds the shared `../Window_Data` directory for `full_av.yaml`. After this, `configs/small_test.yaml` will no longer match that directory until you rebuild the 1-D smoke-test windows or point the configs at separate output folders.

The builder first tries the `video` path recorded in each Stage 1 CSV, then falls back to matching filenames in `--audio-source-dir`, for example `ses01_cam01_vid01_landmarks.csv` -> `ses01_cam01_vid01.mp4`.

For strict full AV training, leave missing audio as an error. If you intentionally need to keep windows whose audio cannot be decoded, add `--allow-missing-audio`; those windows get zero audio targets and `audio_quality: 0.0`, so the audio loss ignores them.

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

## Train, Embed, Cluster

### Smoke Test

```bash
python train.py --config configs/small_test.yaml
python embed.py --config configs/small_test.yaml
python cluster.py --config configs/small_test.yaml --method hdbscan --min_cluster_size 10
python evaluate.py --config configs/small_test.yaml
python export_for_runtime.py --config configs/small_test.yaml
```

### Full AV Run

```bash
python train.py --config configs/full_av.yaml
python embed.py --config configs/full_av.yaml
python cluster.py --config configs/full_av.yaml --method hdbscan --min_cluster_size 20
python evaluate.py --config configs/full_av.yaml
python export_for_runtime.py --config configs/full_av.yaml
```

Use `full_av.yaml` after your windows contain `audio_schema: log_mel_12` and `audio_dim: 12`. The trainer checks that schema before training and rejects placeholder windows for the full AV config.

The trainer validates the first window against the config. If `motion_dim`, `audio_dim`, or `sequence_length` do not match, it fails before training.

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

For the full AV run, the same structure is written under:

```text
Model_Outputs/full_av/
```

Embedding and clustering add:

```text
Model_Outputs/<run_name>/
├── embeddings/
│   ├── embeddings.npy
│   └── embeddings.csv
└── clustering/
    ├── cluster_assignments.csv
    ├── cluster_model.joblib
    ├── embedding_scaler.joblib
    └── umap_model.joblib
```

## Export For Stage 3

Run the export with the same config used for training:

```bash
python export_for_runtime.py --config configs/small_test.yaml
python export_for_runtime.py --config configs/full_av.yaml
```

Exported package:

```text
Model_Outputs/<run_name>/export_for_runtime/
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
  artifact_dir: "../Model_Outputs/full_av/export_for_runtime"
```

`runtime_model_config.json` carries the feature contract, audio schema, sequence length, and GRU architecture. Stage 3 reads it and overrides local fallback settings so live input matches training. Runtime inference still uses motion landmarks only; audio is a training target, not a required live input.

## End-To-End Order

```text
Stage 1
raw video -> undistorted MediaPipe CSVs with handedness labels

Stage 2
CSV landmarks -> canonical windows -> train -> embed -> cluster -> export

Stage 3
camera/video -> optional undistortion -> MediaPipe -> same feature contract -> OSC
```
