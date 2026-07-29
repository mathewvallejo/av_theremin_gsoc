# av_Camera_Calibration_Preprocess

Stage 1 for the audiovisual autoencoder system.

This folder calibrates cameras, undistorts frames, runs MediaPipe hand tracking, and writes per-frame landmark CSVs for Stage 2. The CSV schema now includes handedness metadata so Stage 2 can build the same right/left feature order that Stage 3 uses live.

## Folder Structure

```text
av_Camera_Calibration_Preprocess/
├── README.md
├── configs/
│   └── calibration_config.yaml
├── calibration/
│   └── cam01_calib_*.png
├── docs/
│   ├── HOW_TO_USE_WITH_STAGE2.md
│   └── HOW_TO_USE_WITH_STAGE3.md
├── models/
│   └── hand_landmarker.task
├── outputs/
│   ├── camera_manifest.csv
│   └── calibration/
│       └── cam01_intrinsics.npz
├── requirements.txt
└── scripts/
    ├── capture_chessboard_images.py
    ├── calibrate_camera.py
    ├── make_camera_manifest.py
    ├── extract_landmarks_undistorted.py
    ├── extract_landmarks_undistorted_group.py
    └── undistort_video.py
```

Recommended sibling data layout:

```text
av_autoencoder/
├── Video_Data/
├── Feature_Data/
├── Window_Data/
├── Model_Outputs/
├── av_Camera_Calibration_Preprocess/
├── av_GRU_autoencoder/
└── av_Gesture_OSC_runtime/
```

## Setup

```bash
python -m venv ~/venvs/av_calibration_env
source ~/venvs/av_calibration_env/bin/activate

cd "/Volumes/MP_1/GSoC 2026/av_autoencoder/av_Camera_Calibration_Preprocess"
pip install -r requirements.txt
python -c "import cv2, mediapipe, numpy, pandas, yaml; print('Stage 1 OK')"
```

Place the MediaPipe model here:

```text
models/hand_landmarker.task
```

## 1. Capture Calibration Images

Edit `configs/calibration_config.yaml` for your chessboard.

```bash
python scripts/capture_chessboard_images.py \
  --camera 0 \
  --camera-id cam01 \
  --output-dir "../Calibration_Data/cam01" \
  --config configs/calibration_config.yaml
```

Controls:

```text
SPACE  save current frame when corners are detected
a      toggle autosave
q      quit
```

## 2. Calibrate Camera

```bash
python scripts/calibrate_camera.py \
  --images "../Calibration_Data/cam01" \
  --camera-id cam01 \
  --config configs/calibration_config.yaml \
  --output outputs/calibration/cam01_intrinsics.npz
```

The `.npz` contains `camera_matrix` and `dist_coeffs`. Stage 3 can use the same file for live undistortion.

## 3. Build A Camera Manifest

```bash
python scripts/make_camera_manifest.py \
  --calibration-dir outputs/calibration \
  --output outputs/camera_manifest.csv
```

## 4. Extract Landmarks From One Video

```bash
python scripts/extract_landmarks_undistorted.py \
  --video "../Video_Data/ses01_cam01_vid01.mp4" \
  --camera-calibration outputs/calibration/cam01_intrinsics.npz \
  --output "../Feature_Data/ses01_cam01_vid01_landmarks.csv"
```

## 5. Extract Landmarks From A Folder

Use this for large batches.

```bash
python scripts/extract_landmarks_undistorted_group.py \
  --video-dir "../Video_Data" \
  --camera-calibration outputs/calibration/cam01_intrinsics.npz \
  --output-dir "../Feature_Data" \
  --recursive \
  --skip-existing
```

Notes for long runs:

- The group extractor writes each in-progress video to `*_landmarks.csv.partial`.
- When a video finishes, the `.partial` file is renamed to `*_landmarks.csv`.
- `--skip-existing` lets you restart a batch without reprocessing finished CSVs.
- Use one run per camera calibration. If videos come from different cameras, run each camera group with its matching `.npz`.

## CSV Feature Contract

Each row is one video frame. Columns:

```text
video
frame
timestamp_ms
num_hands
hand0_label
hand0_score
hand0_0 ... hand0_62
hand1_label
hand1_score
hand1_0 ... hand1_62
```

Landmark vectors are MediaPipe normalized `x, y, z` values in this order:

```text
x0 y0 z0 x1 y1 z1 ... x20 y20 z20
```

`hand*_label` is `right`, `left`, or empty. Stage 2 uses this label to build the canonical model input:

```text
right hand 63 values + left hand 63 values = 126 motion features
```

## Optional Undistorted Video Export

Only use this when you need corrected videos on disk.

```bash
python scripts/undistort_video.py \
  --input "../Video_Data/ses01_cam01_vid01.mp4" \
  --camera-calibration outputs/calibration/cam01_intrinsics.npz \
  --output "../Video_Data/ses01_cam01_vid01_undistorted.mp4"
```
