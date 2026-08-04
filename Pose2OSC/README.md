# Pose2OSC

Pose2OSC is a standalone live gesture-control runtime. It does not depend on
the earlier camera preprocessing, GRU autoencoder, or OSC runtime pipeline.

It provides lightweight gesture enrollment and OSC output for a
MediaPipe-controlled Max/MSP instrument. It is designed for the expanded
theremin idea: continuous landmark streams stay available for expressive
control, while user-defined held gestures act as patch states or triggers.

The recognizer is intentionally small:

- no neural network at runtime
- one-frame KNN prediction
- translation and scale invariant hand-shape features
- default `enter_frames=1`, so gestures can enter on the first matching frame
- optional `exit_frames=2` or `3` only if you want more dropout tolerance

## Install

Core model code uses only the Python standard library:

```bash
python -m pip install -e .
```

For live camera + OSC:

```bash
python -m pip install -e '.[live]'
```

## Enroll A Gesture

Hold the gesture in front of the camera while recording:

```bash
pose2osc enroll delay_hold --model models/gestures.json --seconds 2 --show
pose2osc enroll filter_grab --model models/gestures.json --seconds 2 --show
```

Each enrollment stores up to 64 frames by default. That keeps KNN fast while
still capturing small variations in the user's held pose.

## Run OSC

```bash
pose2osc run --model models/gestures.json --host 127.0.0.1 --port 8000 --split-axes
```

Lowest-latency state settings are the defaults:

```bash
pose2osc run --enter-frames 1 --exit-frames 1 --switch-frames 1
```

If the active gesture flickers, keep entry immediate and only relax release:

```bash
pose2osc run --enter-frames 1 --exit-frames 2
```

## OSC Shape

Gesture state:

```text
/pose2osc/state/active delay_hold 0.92
/pose2osc/state/event enter delay_hold 0.92
/pose2osc/gesture/delay_hold/active 1
/pose2osc/gesture/delay_hold/confidence 0.92
```

Continuous landmark vectors:

```text
/pose2osc/hand/right/index_mcp 0.42 0.71 -0.18
```

Axis-specific messages for Max dropdown routing with `--split-axes`:

```text
/pose2osc/hand/right/index_mcp/x 0.42
/pose2osc/hand/right/index_mcp/y 0.71
/pose2osc/hand/right/index_mcp/z -0.18
```

## Model Design

Gesture recognition does not use raw camera position. A frame is converted into
normalized hand-shape features:

1. Translate landmarks so the wrist or palm center is the origin.
2. Divide by palm scale so near/far camera distance has less effect.
3. Optionally mirror left hands into the same canonical shape space.
4. Compare normalized shape vectors with KNN.

That means the same held gesture can be recognized anywhere in the frame.

Raw MediaPipe `x/y/z` values are still sent to Max/MSP for continuous theremin
control. In practice, Max owns the musical mapping:

| Gesture | On Enter | While Held | On Exit |
| --- | --- | --- | --- |
| `delay_hold` | enable delay | `index_mcp/x` -> delay time | disable delay |
| `filter_grab` | enable filter mode | `index_mcp/y` -> cutoff | release |
| `freeze_pose` | trigger freeze | `wrist/z` -> grain size | unfreeze |

## Inspect Or Remove Gestures

```bash
pose2osc inspect --model models/gestures.json
pose2osc remove delay_hold --model models/gestures.json
```

## Notes For Max/MSP

Use the gesture messages as state gates, then route continuous landmark values
inside the matrix. For realtime performance, start with `enter_frames=1` and
only add smoothing or release hysteresis on the Max side where it is musically
useful.
