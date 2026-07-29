# av_Gesture_OSC_runtime

Stage 3 for the audiovisual autoencoder system.

This folder runs live camera or replay video input through MediaPipe, builds the same motion feature contract used in Stage 2, embeds a rolling window with the trained GRU encoder, assigns a cluster, and sends OSC messages to Max/MSP.

## Folder Structure

```text
av_Gesture_OSC_runtime/
├── README.md
├── configs/
│   └── runtime_config.yaml
├── docs/
│   ├── EXPORT_FROM_STAGE2.md
│   └── OSC_MESSAGES.md
├── models/
│   └── hand_landmarker.task
├── requirements.txt
└── runtime/
    ├── calibration_runtime.py
    ├── feature_runtime.py
    ├── gesture_model_runtime.py
    ├── live_camera_to_osc.py
    ├── replay_video_to_osc.py
    ├── osc_receive_test.py
    ├── osc_sender.py
    └── smoothing.py
```

## Setup

```bash
python -m venv ~/venvs/av_gesture_runtime_env
source ~/venvs/av_gesture_runtime_env/bin/activate

cd "/Volumes/MP_1/GSoC 2026/av_autoencoder/av_Gesture_OSC_runtime"
pip install -r requirements.txt
python -c "import torch, cv2, mediapipe, pythonosc; print('Stage 3 OK')"
```

The requirements pin `numpy<2` because some MediaPipe/Matplotlib wheels used by this runtime are still built against the NumPy 1.x C API.

## Model Package

Stage 3 now reads trained model artifacts directly from the Stage 2 export folder. Keep only the MediaPipe model in Stage 3:

```text
hand_landmarker.task
```

Stage 2 writes trained artifacts here:

```text
../Model_Outputs/<run_name>/export_for_runtime/
```

That folder should contain:

```text
encoder.pt
feature_scaler.joblib
cluster_model.joblib
runtime_model_config.json
embedding_scaler.joblib
cluster_names.json
runtime_export_manifest.json
```

Point Stage 3 at the chosen export in `configs/runtime_config.yaml`:

```yaml
runtime_model:
  artifact_dir: "../Model_Outputs/small_test/export_for_runtime"
```

You can also override it at launch:

```bash
python runtime/live_camera_to_osc.py \
  --config configs/runtime_config.yaml \
  --camera 0 \
  --model-dir "../Model_Outputs/motion_only/export_for_runtime"
```

## Runtime Feature Contract

Stage 3 reads `runtime_model_config.json` from `runtime_model.artifact_dir` and lets it override fallback values in `configs/runtime_config.yaml`.

Canonical new exports use:

```text
feature_order: right_63_left_63
hand_order: label
normalize_to_wrist: true
normalize_scale_landmark: 9
include_velocity: false
feature_dim: 126
sequence_length: 60
```

Older exports may use:

```text
hand_order: index
normalize_to_wrist: false
```

That is supported for compatibility, but newly trained models should use the canonical label-based contract.

## Optional Camera Undistortion

If Stage 2 was trained from Stage 1 undistorted landmarks, enable the same camera calibration before MediaPipe in live/replay mode.

Edit `configs/runtime_config.yaml`:

```yaml
calibration:
  enabled: true
  camera_calibration_path: "../av_Camera_Calibration_Preprocess/outputs/calibration/cam01_intrinsics.npz"
  alpha: 0.0
```

Keep this disabled if your model was trained on distorted/raw camera coordinates.

## OSC Settings

```yaml
osc:
  host: "127.0.0.1"
  port: 9000
  prefix: "/av_gesture"
```

In Max/MSP, listen with:

```text
udpreceive 9000
```

## Test OSC

Terminal 1:

```bash
python runtime/osc_receive_test.py
```

Terminal 2:

```bash
python runtime/live_camera_to_osc.py --config configs/runtime_config.yaml --camera 0
```

Synthetic Max/MSP synth mapping test:

```bash
python runtime/osc_synth_test.py \
  --host 127.0.0.1 \
  --port 9000 \
  --prefix /av_gesture \
  --mode performer \
  --fps 30 \
  --duration 60
```

Useful modes:

```text
performer  phrase-like clusters, hand movement, energy, confidence, latent data
pulse      sharp rhythmic energy spikes for envelope/gate mapping
sweep      slower continuous changes for range scaling
idle       no-hand state for gate/reset testing
```

## Run Live Camera

```bash
python runtime/live_camera_to_osc.py --config configs/runtime_config.yaml --camera 0
```

Quit the preview with `q`.

The preview overlay shows the rolling model window. It fills from `0/60` to `60/60`, then stays at `window ready 60/60` because the runtime keeps a full rolling buffer and replaces the oldest frame each tick.

Cluster `-1` means HDBSCAN noise/unassigned or the explicit no-hand idle state. During active hand tracking, a persistent `-1` usually means the live embedding is outside the learned clusters or the model package was trained/clusted in a way that does not match the live feature stream.

## Run Video Replay

```bash
python runtime/replay_video_to_osc.py \
  --config configs/runtime_config.yaml \
  --video "/path/to/test_video.mp4"
```

Paced replay:

```bash
python runtime/replay_video_to_osc.py \
  --config configs/runtime_config.yaml \
  --video "/path/to/test_video.mp4" \
  --realtime \
  --drop-late-frames \
  --debug
```

If the video metadata reports the wrong FPS, override it:

```bash
python runtime/replay_video_to_osc.py \
  --config configs/runtime_config.yaml \
  --video "/path/to/test_video.mp4" \
  --realtime \
  --drop-late-frames \
  --fps 30 \
  --debug
```

Replay uses the same preview overlay and no-hand idle behavior as live camera mode. Set `preview.draw_landmarks: true` in `configs/runtime_config.yaml` to draw MediaPipe hand landmarks.

## Runtime Flow

```text
camera/video
-> optional undistortion
-> MediaPipe Hand Landmarker
-> canonical motion feature
-> rolling 60-frame window
-> feature scaler
-> GRU encoder
-> latent vector
-> cluster model
-> OSC messages
```

## Notes

- Stage 3 does not use audio live.
- The trained encoder is loaded from `runtime_model.artifact_dir/encoder.pt`.
- The runtime validates feature dimension before inference.
- If you rebuild Stage 2 windows with a different feature contract, retrain, re-export, and point `runtime_model.artifact_dir` at the new export folder.
- Local files such as `models/encoder.pt` are still supported as explicit per-file overrides for older setups.
