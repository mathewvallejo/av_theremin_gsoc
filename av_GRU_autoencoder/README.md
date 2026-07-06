# av_GRU_autoencoder

**Audio-video guided GRU autoencoder for self-supervised gesture representation learning and evaluation.**

This is the **Stage 2 training and evaluation stack** for a larger gesture-recognition system.

Stage 1 (av_Camera_Calibration_Preprocess) converts raw video into calibrated MediaPipe landmark features. Stage 2 transforms those features into temporal training windows, trains an audio-guided GRU autoencoder, evaluates the learned latent space, clusters discovered gesture states, and exports a compact runtime package for Stage 3 real-time deployment.

NOTE: All file path names must be converted to local user's filepath(s) to run Python scripts.

---

# What this Stack Does

```text
Stage 1
(raw video)
        ↓
camera calibration
        ↓
MediaPipe landmark extraction
        ↓
Feature_Data/*.csv
        ↓

Stage 2
(build_windows.py)
        ↓
Window_Data/*.npz
        ↓
window_manifest.csv
        ↓
GRU autoencoder training
        ↓
latent gesture embeddings
        ↓
unsupervised clustering
        ↓
evaluation plots/reports
        ↓
export_for_runtime/
```

At runtime, the deployed system can use:

```text
camera
↓
MediaPipe
↓
trained encoder
↓
gesture embedding
↓
cluster/state
↓
OSC
↓
Max/MSP
```

Audio is used during training as a self-supervised auxiliary signal and is not required during live performance.

---

# Recommended Project Layout

```text
/Volumes/MP_1/GSoC 2026/

├── Feature_Data/
│
├── Window_Data/
│
├── Model_Outputs/
│
├── av_Camera_Calibration_Preprocess_v2/
│
├── av_GRU_autoencoder/
│
└── av_Gesture_OSC_Runtime/
```

Python virtual environments should remain on the internal drive:

```text
~/venvs/av_GRU_env
```

---

# Environment Setup

Create:

```bash
mkdir -p ~/venvs
python -m venv ~/venvs/av_GRU_env
```

Activate:

```bash
source ~/venvs/av_GRU_env/bin/activate
```

Verify:

```bash
which python
which pip
```

Expected:

```text
/Users/<user>/venvs/av_GRU_env/bin/python
/Users/<user>/venvs/av_GRU_env/bin/pip
```

Install requirements:

```bash
cd /Volumes/MP_1/GSoC_2026/av_autoencoder/av_GRU_autoencoder
pip install -r requirements.txt
```

If Conda is active:

```bash
conda deactivate
source ~/venvs/av_GRU_env/bin/activate
```

---

# Stage 1 Output (from av_Camera_Calibration_Preprocess)

Stage 1 produces landmark CSV files.

Example:

```text
Feature_Data/
├── ses01_cam01_vid01_landmarks.csv
├── ses01_cam01_vid02_landmarks.csv
├── ses01_cam02_vid01_landmarks.csv
└── ...
```

Each row corresponds to one video frame.

Typical landmark dimensionality:

```text
2 hands
× 21 landmarks
× 3 coordinates (x,y,z)

= 126 motion features
```

This should be done before beginning Stage 2.

---

# Stage 2 Preprocessing (Begin here once Stage 1 is complete)

Before training, landmark CSV files must be converted into fixed-length temporal windows.

This is performed using:

```text
scripts/build_windows.py
```

Run:

```bash
python scripts/build_windows.py \
  --input-dir "/Volumes/MP_1/GSoC 2026/Feature_Data" \
  --output-dir "/Volumes/MP_1/GSoC 2026/Window_Data" \
  --sequence-length 60 \
  --hop-length 30
```

Example:

```text
60 frame window
30 frame hop
```

produces overlapping gesture sequences.

---

# Window Output Format

Generated windows:

```text
Window_Data/
├── ses01_cam01_vid01_win000000.npz
├── ses01_cam01_vid01_win000001.npz
├── ...
```

Each `.npz` contains:

```python
motion          # [sequence_length, motion_dim]
audio           # audio features or placeholder vector
audio_quality   # scalar [0,1]
```

Example:

```text
motion.shape = (60,126)
audio.shape = (1,)
audio_quality = 0.0
```

The placeholder format above is used during motion-only validation.

---

# Manifest Format

`build_windows.py` automatically generates:

```text
Window_Data/window_manifest.csv
```

Required column:

```text
path
```

Additional columns:

```text
window_file
source_csv
source_video
camera_id
session_id
start_frame
end_frame
start_time_ms
end_time_ms
split
```

The Stage 2 dataset loader requires:

```text
path
```

to exist.

---

# Motion-Only Validation Workflow

Before integrating audio features, validate the pipeline using landmark motion only.

The generated windows contain:

```python
motion
audio            # placeholder vector
audio_quality    # 0.0
```

This validates:

```text
window generation
dataset loading
GRU training
embedding generation
clustering
runtime export
```

without requiring audio preprocessing.

---

# Configuration Files

Available configs:

```text
configs/default.yaml
    Full AV training

configs/small_test.yaml
    Motion-only validation

configs/motion_only.yaml
    Motion-only training

configs/latent_16.yaml
    Small latent space experiment

configs/latent_32.yaml
    Larger latent space experiment

configs/gru_hidden_256.yaml
    Larger recurrent model

configs/full_av.yaml
    Audio-guided training
```

---

# Example Motion-Only Configuration

```yaml
project_name: av_GRU_autoencoder
seed: 42

data:
  dataset_dir: "/Volumes/MP_1/GSoC 2026/Window_Data"
  manifest_csv: "/Volumes/MP_1/GSoC 2026/Window_Data/window_manifest.csv"
  output_dir: "/Volumes/MP_1/GSoC 2026/Model_Outputs"
  val_fraction: 0.15
  test_fraction: 0.15

features:
  motion_dim: 126
  audio_dim: 1
  sequence_length: 60
  use_audio_guidance: false
  audio_quality_threshold: 0.35

model:
  hidden_dim: 64
  latent_dim: 16
  num_layers: 1
  dropout: 0.10
  bidirectional: true

training:
  batch_size: 8
  epochs: 5
  learning_rate: 0.001
  weight_decay: 0.00001
  grad_clip: 1.0
  patience: 5
  device: auto

loss:
  motion_reconstruction_weight: 1.0
  audio_prediction_weight: 0.0
  latent_smoothness_weight: 0.02

clustering:
  method: hdbscan
  min_cluster_size: 10
  kmeans_clusters: 8
  umap_neighbors: 15
  umap_min_dist: 0.1

runtime_export:
  export_dir: "/Volumes/MP_1/GSoC 2026/Model_Outputs/export_for_runtime"
```

---

# Train

Run:

```bash
python train.py --config configs/small_test.yaml
```

Outputs:

```text
outputs/checkpoints/best_model.pt
outputs/scalers/feature_scaler.joblib
outputs/metrics/training_history.json
outputs/train_split.csv
outputs/val_split.csv
outputs/test_split.csv
```

---

# Generate Embeddings

```bash
python embed.py --config configs/default.yaml
```

Outputs:

```text
outputs/embeddings/embeddings.npy
outputs/embeddings/embeddings.csv
```

---

# Cluster the Latent Space

HDBSCAN:

```bash
python cluster.py --method hdbscan --min_cluster_size 20
```

KMeans:

```bash
python cluster.py --method kmeans --k 8
```

Outputs:

```text
outputs/clustering/
├── cluster_assignments.csv
├── cluster_model.joblib
├── embedding_scaler.joblib
└── umap_model.joblib
```

---

# Evaluate

```bash
python evaluate.py
```

Outputs:

```text
outputs/reports/evaluation_report.json
```

---

# Plot Metrics

```bash
python plot_metrics.py
```

Outputs:

```text
outputs/plots/train_val_loss.png
outputs/plots/embedding_umap_clusters.png
outputs/plots/cluster_distribution.png
```

---

# Export Runtime Package

Run the export command using the same config that was used for training:

```bash
python export_for_runtime.py --config configs/small_test.yaml
```

Outputs:

```text
outputs/export_for_runtime/
├── encoder.pt                         # canonical Stage 3 checkpoint name
├── av_gru_encoder.pt                  # backwards-compatible alias
├── feature_scaler.joblib
├── embedding_scaler.joblib            # optional, if clustering produced it
├── cluster_model.joblib
├── cluster_names.json
├── runtime_model_config.json          # model + feature metadata for Stage 3
└── runtime_export_manifest.json
```

This folder is consumed by Stage 3. Copy the contents of `export_for_runtime/` into the Stage 3 `models/` folder. The important integration fix is that Stage 2 now exports both `encoder.pt` and `runtime_model_config.json`, which are the names Stage 3 expects.

---

# Model Architecture

```text
motion sequence
      ↓
GRU encoder
      ↓
latent gesture vector
     / \
    /   \
motion decoder   audio prediction head
```

Training objective:

```text
motion reconstruction loss
+
audio-quality-gated prediction loss
+
latent smoothness regularization
```

Audio is not treated as a human label.

Audio functions as a self-supervised training signal which encourages the latent space to organize gestures according to their acoustic consequences.

---

# Evaluation Philosophy

This is an unsupervised learning system.

Useful metrics include:

* training reconstruction loss
* validation reconstruction loss
* audio prediction loss
* silhouette score
* Davies-Bouldin score
* cluster size distribution
* UMAP visualization quality
* human review of sampled clips

Human review remains important because discovered gesture states must ultimately be interpreted musically.

---

# Troubleshooting

## TypeError: 'NoneType' object is not subscriptable

Cause:

```text
Invalid YAML formatting
or
missing required configuration keys
```

Fix:

```text
Check indentation and verify all sections exist.
```

---

## manifest_csv must include a path column

Cause:

```text
Old manifest generated by an earlier windowing script.
```

Fix:

```text
Re-run build_windows.py.
```

---

## IsADirectoryError: [Errno 21] Is a directory: '.'

Cause:

```text
manifest_csv was set to an empty string.
```

Fix:

```text
Point manifest_csv to Window_Data/window_manifest.csv
```

---

## Could not build wheels for llvmlite

Cause:

```text
Numba / llvmlite compatibility issue.
```

Fix:

```text
Upgrade pip, setuptools, wheel
and install compatible versions.
```

---

## NumPy 2.x / PyTorch Compatibility Warnings

Cause:

```text
Some packages compiled against NumPy 1.x.
```

Fix:

```text
Use package versions specified in requirements.txt.
```

---

# Relationship to the Full System

```text
01_av_Camera_Calibration_Preprocess_v2
    calibration
    undistortion
    MediaPipe landmark extraction

02_av_GRU_autoencoder
    build_windows.py
    train
    evaluate
    cluster
    export runtime package

03_av_Gesture_OSC_Runtime
    live camera
    MediaPipe
    trained encoder
    gesture state
    OSC
    Max/MSP
```

---

# Future HPC Deployment

This stack is designed so training can later be moved to:

```text
Digital Research Alliance of Canada
or
other HPC environments
```

by changing paths inside a config file.

Example:

```bash
python train.py --config configs/hpc.yaml
```

The primary artifact remains:

```text
outputs/export_for_runtime/
```

which is imported by the Stage 3 runtime.
