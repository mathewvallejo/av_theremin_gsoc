# Stage 2 — Overview

This project is Stage 2 only: model training and evaluation.

It expects Stage 1 to handle raw video, optional undistortion/calibration, MediaPipe landmark extraction, audio feature extraction, and windowing into `.npz` files.

It exports artifacts for Stage 3 realtime OSC deployment via `export_for_runtime.py`.

## Encoder pooling

The encoder uses **mean pooling** over all GRU time steps, followed by `LayerNorm` and a `Linear` projection. This is defined in `models/av_gru_autoencoder.py :: AVGRUAutoencoder.encode()`.

`export_for_runtime.py` exports this method as a TorchScript traced module so Stage 3 loads the exact computation graph without redefining the architecture.

## Key output artifacts

| File | Used by |
|---|---|
| `outputs/checkpoints/best_model.pt` | `embed.py`, `export_for_runtime.py` |
| `outputs/scalers/feature_scaler.joblib` | `embed.py`, `export_for_runtime.py` |
| `outputs/embeddings/embeddings.csv` | `cluster.py` |
| `outputs/clustering/cluster_model.joblib` | `evaluate.py`, `export_for_runtime.py` |
| `outputs/export_for_runtime/` | Stage 3 `av_Gesture_OSC_runtime/models/` |
