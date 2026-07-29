# Exporting Stage 2 Artifacts For Stage 3

In `av_GRU_autoencoder`, run the export command with the same config used for training:

```bash
python export_for_runtime.py --config configs/small_test.yaml
```

The export folder should contain:

```text
../Model_Outputs/small_test/export_for_runtime/
├── encoder.pt
├── av_gru_encoder.pt
├── feature_scaler.joblib
├── embedding_scaler.joblib
├── cluster_model.joblib
├── cluster_names.json
├── runtime_model_config.json
└── runtime_export_manifest.json
```

Do not copy these files into Stage 3 for normal use. Instead, point Stage 3 at this folder:

```yaml
runtime_model:
  artifact_dir: "../Model_Outputs/small_test/export_for_runtime"
```

To audition another trained run without editing YAML:

```bash
python runtime/live_camera_to_osc.py \
  --config configs/runtime_config.yaml \
  --camera 0 \
  --model-dir "../Model_Outputs/motion_only/export_for_runtime"
```

Stage 3 still expects the MediaPipe hand model here:

```text
av_Gesture_OSC_runtime/models/hand_landmarker.task
```

`runtime_model_config.json` is required. It stores:

```text
sequence_length
feature_dim
include_velocity
normalize_to_wrist
normalize_scale_landmark
hand_order
feature_order
GRU architecture settings
```

Stage 3 reads this file from `artifact_dir` and overrides local fallback feature settings before inference.
