# Connecting Stage 1 to Stage 3

Stage 3 (`av_Gesture_OSC_runtime`) uses the same camera calibration file that was used during landmark extraction in Stage 1. This ensures the live camera geometry matches the training data geometry.

## What to copy

Copy the `.npz` calibration file for your deployment camera into Stage 3:

```bash
cp outputs/calibration/cam01_intrinsics.npz \
   ../av_Gesture_OSC_runtime/models/cam01_intrinsics.npz
```

## How Stage 3 uses it

If you add optional undistortion to the Stage 3 live pipeline, pass the `.npz` file path to `cv2.undistort()` using the same `camera_matrix` and `dist_coeffs` keys that Stage 1 writes.

Currently, Stage 3's `live_camera_to_osc.py` runs MediaPipe directly on raw frames. If you find tracking quality is noticeably worse on the live camera than it was during training (where frames were undistorted), add undistortion to the pipeline loop in `runtime/pipeline.py` using the calibration file.
