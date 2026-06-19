# av_GRU_autoencoder

**Stage 2: audio-video guided GRU autoencoder for self-supervised gesture representation learning.**

This stack trains a GRU autoencoder on windowed motion sequences from Stage 1, evaluates the learned latent space, clusters discovered gesture states, plots metrics, and exports a TorchScript encoder for Stage 3.

---

## What this stack does

```text
Stage 1 preprocessing output
        ↓
windowed MediaPipe motion features + audio features
        ↓
av_GRU_autoencoder training
        ↓
latent gesture embeddings
        ↓
unsupervised clustering (HDBSCAN or k-means)
        ↓
evaluation plots and reports
        ↓
TorchScript encoder export for Stage 3
```

At runtime, the deployed system uses **video/MediaPipe only**. Audio guides training as a self-supervised auxiliary signal; it is not required at inference time.

---

## Folder Structure

```text
av_GRU_autoencoder/
├── README.md
├── requirements.txt
├── train.py
├── embed.py
├── cluster.py
├── evaluate.py
├── plot_metrics.py
├── export_for_runtime.py
├── configs/
│   └── default.yaml
├── docs/
│   └── STAGE_OVERVIEW.md
├── src/
│   ├── config.py
│   ├── dataset.py
│   └── losses.py
├── models/
│   └── av_gru_autoencoder.py
└── outputs/
```

---

## Install

```bash
mkdir -p ~/venvs
python -m venv ~/venvs/av_GRU_env
source ~/venvs/av_GRU_env/bin/activate

cd /Volumes/MP_1/av_GRU_autoencoder
pip install -r requirements.txt
```

If your terminal shows `(base)` alongside the venv, deactivate conda first:

```bash
conda deactivate
source ~/venvs/av_GRU_env/bin/activate
```

---

## Expected Input Format

Stage 1 produces per-frame landmark CSVs. An intermediate windowing step (your responsibility) converts these into `.npz` files:

```python
motion        # shape: [sequence_length, motion_dim]
audio         # shape: [audio_dim] or [sequence_length, audio_dim]
audio_quality # scalar in [0, 1]
```

```text
data/windowed_sequences/
├── session01_cam01_win000001.npz
├── session01_cam01_win000002.npz
└── ...
```

Optional manifest:

```text
data/window_manifest.csv
```

Recommended columns: `path, source_video, camera_id, start_time, end_time, split`

If `split` is included, use `train`, `val`, `test`. Otherwise the stack creates random splits.

---

## Configure an Experiment

Edit `configs/default.yaml`. Key settings:

```yaml
features:
  motion_dim: 126       # 2 hands × 21 landmarks × 3 xyz = 126
  audio_dim: 12
  sequence_length: 60

model:
  hidden_dim: 128
  latent_dim: 24
```

Create additional configs for experiments:

```text
configs/small_test.yaml
configs/latent_16.yaml
configs/gru_hidden_256.yaml
```

---

## Full Training and Evaluation Pipeline

### 1. Train

```bash
python train.py --config configs/default.yaml
```

Outputs:

```text
outputs/checkpoints/best_model.pt
outputs/scalers/feature_scaler.joblib
outputs/metrics/training_history.json
outputs/train_split.csv  outputs/val_split.csv  outputs/test_split.csv
```

### 2. Embed all windows

```bash
python embed.py --config configs/default.yaml
```

Outputs:

```text
outputs/embeddings/embeddings.npy
outputs/embeddings/embeddings.csv
```

### 3. Cluster the latent space

```bash
python cluster.py --method hdbscan --min_cluster_size 20
# or
python cluster.py --method kmeans --k 8
```

Outputs:

```text
outputs/clustering/cluster_assignments.csv
outputs/clustering/cluster_model.joblib
outputs/clustering/embedding_scaler.joblib
outputs/clustering/umap_model.joblib
```

### 4. Evaluate clusters

```bash
python evaluate.py
```

Outputs:

```text
outputs/reports/evaluation_report.json
```

### 5. Plot metrics

```bash
python plot_metrics.py
```

Outputs:

```text
outputs/plots/train_val_loss.png
outputs/plots/embedding_umap_clusters.png
outputs/plots/cluster_distribution.png
```

### 6. Export for Stage 3

```bash
python export_for_runtime.py --config configs/default.yaml
```

Outputs:

```text
outputs/export_for_runtime/
├── encoder_scripted.pt          ← TorchScript encoder (not a raw state dict)
├── feature_scaler.joblib
├── embedding_scaler.joblib
├── cluster_model.joblib
├── cluster_names.json
└── runtime_model_config.json
```

Copy the contents of `export_for_runtime/` into `av_Gesture_OSC_runtime/models/`.

---

## Model Architecture

```text
motion sequence  [B, T, motion_dim]
      ↓
bidirectional GRU encoder
      ↓
mean pool over time steps  [B, enc_out_dim]
      ↓
LayerNorm → Linear          [B, latent_dim]
      ↓ z
     / \
    /   \
motion decoder    audio prediction head
```

Training loss:

```text
motion reconstruction loss
+ audio-quality-gated audio prediction loss
+ latent smoothness penalty
```

---

## TorchScript Export

`export_for_runtime.py` exports the `encode()` method as a TorchScript traced module (`encoder_scripted.pt`). This is important for correctness: it captures the exact computation graph — mean pooling over time and LayerNorm — so Stage 3 cannot accidentally diverge by redefining the architecture. Stage 3 loads it with `torch.jit.load()`.

---

## Evaluation Philosophy

Because this is unsupervised, evaluation is not simple accuracy. Useful metrics include:

- Train/validation reconstruction loss
- Audio prediction loss when audio quality is high
- Silhouette score of latent clusters
- Davies-Bouldin score
- UMAP cluster separability
- Cluster size distribution
- Human review of clips sampled from clusters

Human review remains important. The model discovers gesture states; humans decide whether those states are musically meaningful.

---

## Notes for HPC Use

Move training to the Digital Research Alliance of Canada (or similar) by changing paths in a dedicated config:

```bash
python train.py --config configs/hpc.yaml
```

The key exported artifact — `outputs/export_for_runtime/` — is the same regardless of where training runs.
