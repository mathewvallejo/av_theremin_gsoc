# Exporting from Stage 2 into Stage 3

After training and evaluation in `av_GRU_autoencoder`, run:

```bash
python export_for_runtime.py --config configs/default.yaml
```

This writes:

```text
outputs/export_for_runtime/
├── encoder_scripted.pt       ← TorchScript traced encoder
├── feature_scaler.joblib
├── embedding_scaler.joblib
├── cluster_model.joblib
├── cluster_names.json
└── runtime_model_config.json
```

Then copy the contents into Stage 3:

```bash
cp outputs/export_for_runtime/* /path/to/av_Gesture_OSC_runtime/models/
```

Also copy the MediaPipe hand model if not already present:

```text
models/hand_landmarker.task
```

## About encoder_scripted.pt

The exported encoder is a **TorchScript traced module**, not a raw state dict. Stage 3 loads it with:

```python
self.encoder = torch.jit.load(encoder_path, map_location=device)
```

This is the correct and only supported way to load it — do not try to reconstruct the architecture manually. The TorchScript module contains the exact computation graph from Stage 2, including mean pooling and LayerNorm, and will produce identical latent vectors to training.

## Feature dimension check

If Stage 3 crashes with a shape mismatch on the first frame, the `feature_dim` in `runtime_config.yaml` does not match what Stage 2 was trained on. Check:

- `feature_dim: 252` if `include_velocity: true` (default)
- `feature_dim: 126` if `include_velocity: false`

The `runtime_model_config.json` written by `export_for_runtime.py` records the training feature dimension for reference.
