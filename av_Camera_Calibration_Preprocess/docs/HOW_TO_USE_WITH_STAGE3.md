# How Stage 1 Feeds Stage 3

Stage 3 can use the same calibration `.npz` that Stage 1 used for offline preprocessing.

In `av_Gesture_OSC_runtime/configs/runtime_config.yaml`:

```yaml
calibration:
  enabled: true
  camera_calibration_path: "../av_Camera_Calibration_Preprocess/outputs/calibration/cam01_intrinsics.npz"
  alpha: 0.0
```

Enable this when the Stage 2 model was trained from Stage 1 undistorted landmarks and you want live/replay input to pass through the same camera correction before MediaPipe.
