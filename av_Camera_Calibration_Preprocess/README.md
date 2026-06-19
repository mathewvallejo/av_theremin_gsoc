# av_Camera_Calibration_Preprocess

**Stage 1: camera calibration, live chessboard capture, and preprocessing for AV-GRU gesture systems.**

This stack feeds:

- Stage 2: `av_GRU_autoencoder`
- Stage 3: `av_Gesture_OSC_runtime`

---

## Recommended Layout

Keep data and all three stages at the same hierarchy level:

```text
GSoC 2026/
├── Video_Data/
├── Calibration_Data/
├── Feature_Data/
├── Model_Outputs/
├── av_Camera_Calibration_Preprocess/
├── av_GRU_autoencoder/
└── av_Gesture_OSC_runtime/
```

---

## Folder Structure

```text
av_Camera_Calibration_Preprocess/
├── README.md
├── requirements.txt
├── configs/
│   └── calibration_config.yaml
├── calibration/
│   └── README_PLACE_CALIBRATION_IMAGES_HERE.txt
├── models/
│   └── README_PLACE_MEDIAPIPE_MODEL_HERE.txt
├── scripts/
│   ├── capture_chessboard_images.py
│   ├── calibrate_camera.py
│   ├── make_camera_manifest.py
│   ├── extract_landmarks_undistorted.py
│   └── undistort_video.py
├── docs/
│   ├── HOW_TO_USE_WITH_STAGE2.md
│   └── HOW_TO_USE_WITH_STAGE3.md
└── outputs/
    ├── calibration/
    └── features/
```

---

## Install

```bash
mkdir -p ~/venvs
python -m venv ~/venvs/av_calibration_env
source ~/venvs/av_calibration_env/bin/activate

cd "/Volumes/MP_1/GSoC 2026/av_autoencoder/av_Camera_Calibration_Preprocess"
pip install -r requirements.txt
```

Verify:

```bash
python -c "import cv2, mediapipe, numpy, pandas, yaml; print('Stage 1 OK')"
```

---

## Configure Chessboard

Edit `configs/calibration_config.yaml`:

```yaml
checkerboard:
  inner_corners_x: 9
  inner_corners_y: 6
  square_size_m: 0.024
```

`inner_corners_x` and `inner_corners_y` are the number of **inside corners**, not the number of printed squares.

---

## Step A — Live Chessboard Capture

Use this script to capture calibration images directly from a live camera with visual corner feedback.

```bash
python scripts/capture_chessboard_images.py \
  --camera 0 \
  --camera-id cam01 \
  --output-dir "../Calibration_Data/cam01" \
  --config configs/calibration_config.yaml
```

Controls:

```text
SPACE  save current frame (only if chessboard is detected)
a      toggle autosave
q      quit
```

Capture 20–50 good images per camera. Cover center, edges, corners, tilted angles, near, and far distances.

---

## Step B — Calibrate Each Camera

```bash
python scripts/calibrate_camera.py \
  --images "../Calibration_Data/cam01" \
  --camera-id cam01 \
  --config configs/calibration_config.yaml \
  --output outputs/calibration/cam01_intrinsics.npz
```

Repeat for each camera (`cam02`, `cam03`, etc.).

The script prints calibration quality guidance based on mean reprojection error:

- **< 0.5 px** — good
- **0.5–1.0 px** — acceptable but consider recapturing
- **> 1.0 px** — poor; recapture with more diverse board poses

The output `.npz` contains the camera matrix, distortion coefficients, image size, reprojection error, and board settings.

---

## Step C — Build a Camera Manifest

```bash
python scripts/make_camera_manifest.py \
  --calibration-dir outputs/calibration \
  --output outputs/camera_manifest.csv
```

---

## Step D — Extract Landmarks from Undistorted Frames (preferred)

Place MediaPipe's hand model at `models/hand_landmarker.task`.

```bash
python scripts/extract_landmarks_undistorted.py \
  --video "../Video_Data/ses01_cam01_vid01.mp4" \
  --camera-calibration outputs/calibration/cam01_intrinsics.npz \
  --output "../Feature_Data/ses01_cam01_vid01_landmarks.csv"
```

This reads the raw video, undistorts each frame in memory, runs MediaPipe, and writes landmark features — without creating a corrected `.mp4`.

---

## Step E — Optional Undistorted Video Export

Use this only when you actually need corrected video files.

```bash
python scripts/undistort_video.py \
  --input "../Video_Data/ses01_cam01_vid01.mp4" \
  --camera-calibration outputs/calibration/cam01_intrinsics.npz \
  --output "../Video_Data/ses01_cam01_vid01_undistorted.mp4"
```

---

## Notes

- Chessboard calibration is sufficient for lens distortion and intrinsic calibration. For future stereo 3D reconstruction, ChArUco may be worth adding later.
- All scripts use proper `main()` guards and are safe to import as modules without side effects.
- See `docs/HOW_TO_USE_WITH_STAGE2.md` and `docs/HOW_TO_USE_WITH_STAGE3.md` for cross-stage wiring.
