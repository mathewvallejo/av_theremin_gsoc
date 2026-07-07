# Exporting Stage 2 Artifacts for Stage 3

In the Stage 2 `av_GRU_autoencoder` folder, run the export command with the same config used for training:

```bash
python export_for_runtime.py --config configs/small_test.yaml
```

The export folder should contain:

```text
encoder.pt
av_gru_encoder.pt
feature_scaler.joblib
embedding_scaler.joblib
cluster_model.joblib
cluster_names.json
runtime_model_config.json
runtime_export_manifest.json
```

Copy those files into the Stage 3 runtime folder:

```bash
cp /path/to/Model_Outputs/small_test/export_for_runtime/* \
   /path/to/av_Gesture_OSC_runtime/models/
```

Also make sure `models/hand_landmarker.task` exists in Stage 3.

`runtime_model_config.json` is required. It stores the motion dimension, sequence length, GRU hidden size, latent size, number of layers, dropout, and bidirectional flag used by the trained Stage 2 model. Stage 3 reads this file so the live runtime matches the training architecture.
