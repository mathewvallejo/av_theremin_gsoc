# av_Gesture_OSC_runtime

**Stage 3: realtime deployment bridge for MediaPipe + trained AV-GRU gesture model + Max/MSP OSC output.**

This stack is designed for live performance or live testing. It does not train a model. It loads exported artifacts from Stage 2, receives camera or video input, runs MediaPipe hand tracking, builds rolling motion windows, runs the trained encoder, assigns a gesture cluster, and sends hand landmark and gesture data to Max/MSP over OSC.

---

## Runtime Architecture

```text
Camera or video file
  ↓
MediaPipe Hand Landmarker
  ↓
wrist-relative landmark normalization + velocity features
  ↓
rolling sequence buffer (window_size frames)
  ↓
TorchScript AV-GRU encoder (from Stage 2 export)
  ↓
cluster assignment + temporal smoothing
  ↓
OSC messages to Max/MSP
```

The encoder is loaded as a TorchScript module (`encoder_scripted.pt`) exported by Stage 2's `export_for_runtime.py`. This guarantees the exact computation graph — including mean pooling and LayerNorm — is identical to what was trained.

---

## Folder Structure

```text
av_Gesture_OSC_runtime/
├── README.md
├── requirements.txt
├── configs/
│   └── runtime_config.yaml
├── docs/
│   ├── OSC_MESSAGES.md
│   └── EXPORT_FROM_STAGE2.md
├── models/
│   ├── hand_landmarker.task          ← add manually (MediaPipe model)
│   ├── encoder_scripted.pt           ← TorchScript encoder from Stage 2
│   ├── feature_scaler.joblib
│   ├── cluster_model.joblib
│   ├── embedding_scaler.joblib       ← optional
│   ├── cluster_names.json            ← optional
│   └── runtime_model_config.json
└── runtime/
    ├── pipeline.py                   ← shared inference pipeline
    ├── feature_runtime.py
    ├── gesture_model_runtime.py
    ├── osc_sender.py
    ├── smoothing.py
    ├── live_camera_to_osc.py
    └── replay_video_to_osc.py
```

---

## Required Model Artifacts

Minimum required in `models/`:

```text
hand_landmarker.task
encoder_scripted.pt
feature_scaler.joblib
cluster_model.joblib
runtime_model_config.json
```

Optional but recommended:

```text
cluster_names.json
embedding_scaler.joblib
```

Download the MediaPipe hand model if you do not already have it:

```bash
curl -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
```

---

## Setup

```bash
mkdir -p ~/venvs
python -m venv ~/venvs/av_gesture_runtime_env
source ~/venvs/av_gesture_runtime_env/bin/activate

cd /path/to/av_Gesture_OSC_runtime
pip install -r requirements.txt
```

Test imports:

```bash
python -c "from mediapipe.tasks.python import vision; print('mediapipe OK')"
python -c "import torch, cv2, pythonosc; print('runtime imports OK')"
```

---

## Configure

Edit `configs/runtime_config.yaml`.

### Feature dimensions

`feature_dim` must match what Stage 2 was trained on:

- Base: 2 hands × 21 landmarks × 3 (xyz) = **126**
- With `include_velocity: true`: 126 × 2 = **252** (base + velocity delta)

The default config uses 252 (velocity enabled). If you trained Stage 2 without velocity, set `feature_dim: 126` and `include_velocity: false`.

### OSC

```yaml
osc:
  host: "127.0.0.1"
  port: 9000
  prefix: "/av_gesture"
```

In Max/MSP, listen with `udpreceive 9000` and route messages such as `/av_gesture/gesture/cluster`.

---
### OSC Test in Terminal

A quick check of OSC functionality. Requires two terminal windows. In terminal window 1, run:

```bash
python runtime/osc_receive_test.py
```

In terminal window 2, run:

```bash
python runtime/live_camera_to_osc.py --config configs/runtime_config.yaml --camera 0
```

The user should see values change in the terminal window 1 as hands enter the video frame.

---

## Run Live Camera Mode

```bash
python runtime/live_camera_to_osc.py --config configs/runtime_config.yaml
```

Use a specific camera index:

```bash
python runtime/live_camera_to_osc.py --config configs/runtime_config.yaml --camera 1
```

Quit with `q` in the OpenCV preview window.

---

## Run Video Replay Mode

Useful for testing the full pipeline before using a live camera:

```bash
python runtime/replay_video_to_osc.py \
  --config configs/runtime_config.yaml \
  --video /path/to/test_video.mp4

# Optionally pace output to match source FPS:
python runtime/replay_video_to_osc.py \
  --config configs/runtime_config.yaml \
  --video /path/to/test_video.mp4 \
  --realtime
```

---

## Code Structure

Both entry points (`live_camera_to_osc.py` and `replay_video_to_osc.py`) are thin wrappers that open their respective source and delegate to the shared `runtime/pipeline.py`. The pipeline handles MediaPipe, feature extraction, windowing, model inference, smoothing, and OSC sending. To modify runtime behaviour, edit `pipeline.py`.

---

## Important Notes

### Runtime does not use audio

Audio guided Stage 2 training. In Stage 3, only camera input is used:

```text
camera → MediaPipe → trained encoder → gesture state → OSC
```

This is intentional. The audio was a training-time teacher signal so the latent space reflects musical gesture patterns, but the live instrument must respond before sound is produced.

### Feature dimensions must match Stage 2

If you change window length, normalization, velocity features, or landmark ordering in Stage 2, export a matching `runtime_model_config.json` and update `runtime_config.yaml`. Mismatches will cause a hard crash at the first inference step.

### Error handling

The pipeline catches per-frame exceptions and logs them without crashing, so a single bad frame or transient OSC error will not interrupt a live performance.

---

## Suggested Max/MSP Mapping

Use raw landmarks for continuous control:

```text
hand position → pitch / volume
motion energy → articulation intensity
```

Use gesture state for context:

```text
cluster ID → mapping mode
gesture name → envelope / modulation / timbre
latent vector → continuous synthesis parameters
```

See `docs/OSC_MESSAGES.md` for the full message reference.
