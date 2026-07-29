Place the MediaPipe hand model here:

- hand_landmarker.task

Trained Stage 2 artifacts should normally stay in:

../Model_Outputs/<run_name>/export_for_runtime/

Then point configs/runtime_config.yaml at that folder:

runtime_model:
  artifact_dir: "../Model_Outputs/<run_name>/export_for_runtime"

Local encoder/scaler/cluster files in this models folder are still supported as explicit per-file overrides for older setups.
