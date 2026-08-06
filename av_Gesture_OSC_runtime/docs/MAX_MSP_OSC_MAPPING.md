# Max/MSP OSC Mapping Guide

This runtime sends OSC over UDP. The default destination is:

```text
host: 127.0.0.1
port: 9000
prefix: /av_gesture
```

## Receiving In Max/MSP

There are two common Max patch styles. Use the one that matches your patch.

### If Your Patch Routes Full OSC Addresses

Some OSC externals decode the packet and leave the full address as the first route item. In that case, start with:

```text
[udpreceive 9000]
|
[route /av_gesture/gesture/cluster /av_gesture/motion/energy /av_gesture/gesture/confidence /av_gesture/hand/right/index_tip]
```

Use the full OSC address in `route`. For example, route `/av_gesture/gesture/cluster`, not just `gesture/cluster`.

### If You Use Native Max `[oscparse]`

The common native Max pattern is:

```text
[udpreceive 9000]
|
[oscparse]
|
[route av_gesture]
|
[route gesture motion hand latent]
```

With this style, route address parts without slashes:

```text
gesture outlet -> [route cluster confidence name changed active]
motion outlet  -> [route energy window_ready]
hand outlet    -> [route num_hands right left]
right outlet   -> [route present wrist thumb_tip index_tip middle_tip ring_tip pinky_tip landmarks]
left outlet    -> [route present wrist thumb_tip index_tip middle_tip ring_tip pinky_tip landmarks]
latent outlet  -> [unpack 0. 0. 0. ...]
```

If you are not sure which style you have, connect a `[print osc]` after your OSC receive/parse objects and look at the first atom. If it starts with `/av_gesture/...`, route full addresses. If it starts with `av_gesture`, route by address parts.

## Core Synth Mapping Messages

These are the best first signals to map into a synth engine.

| OSC address | Type | Values | Max unpacking | Suggested synth use |
| --- | --- | --- | --- | --- |
| `/av_gesture/gesture/cluster` | int | `-1`, `0`, `1`, ... | `[i]` | Discrete synth state, preset, scene, or gesture class selector. `-1` means no hand/noise/idle. |
| `/av_gesture/gesture/active` | int | `0` or `1` | `[i]` | Master gate. Use `0` to silence, freeze, or reset gesture-driven modulation. |
| `/av_gesture/motion/energy` | float | usually `0.0` to about `1.0` | `[f]` | Amplitude, brightness, density, delay feedback, reverb send, grain rate. |
| `/av_gesture/gesture/confidence` | float | `0.0` to `1.0` | `[f]` | Modulation depth, wet/dry confidence, gate threshold, smoothing amount. |
| `/av_gesture/gesture/changed` | int | `0` or `1` | `[i]` or `[sel 1]` | Trigger event when the gesture cluster changes. |
| `/av_gesture/hand/num_hands` | int | `0`, `1`, or `2` | `[i]` | Switch one-hand/two-hand synth behavior. |
| `/av_gesture/hand/right/index_tip` | 3 floats | `x y z` | `[unpack 0. 0. 0.]` | Continuous right-hand control. Good first landmark for pitch/filter/spatial position. |
| `/av_gesture/hand/left/index_tip` | 3 floats | `x y z` | `[unpack 0. 0. 0.]` | Continuous left-hand control. Good for secondary modulation. |
| `/av_gesture/latent` | float list | 16 floats by default | `[unpack 0. 0. ...]` | Advanced continuous control from the model embedding. Best after the simpler signals work. |

## Message Types

The sender deliberately uses ints for discrete values:

```text
/av_gesture/gesture/cluster      int
/av_gesture/gesture/active       int
/av_gesture/gesture/changed      int
/av_gesture/hand/num_hands       int
/av_gesture/hand/right/present   int
/av_gesture/hand/left/present    int
/av_gesture/motion/window_ready  int
```

It uses floats for continuous values:

```text
/av_gesture/gesture/confidence   float
/av_gesture/motion/energy        float
/av_gesture/hand/...             x y z floats
/av_gesture/latent               float list
```

It uses a symbol/string for the gesture label:

```text
/av_gesture/gesture/name         symbol
```

If Max displays an integer as `0.` or `1.`, something in the patch is probably coercing it to a float, such as `[unpack 0.]`, `[flonum]`, or `[scale]`. For discrete values, send the output through `[i]`.

## Gesture Messages

### `/av_gesture/gesture/cluster`

```text
int
```

The main predicted gesture class. This comes from clustering the GRU latent embedding.

```text
-1  idle, no hand, noise, or unassigned
0   cluster 0
1   cluster 1
2   cluster 2
...
```

Recommended Max pattern when routing full addresses:

```text
[route /av_gesture/gesture/cluster]
|
[i]
|
[sel -1 0 1 2 3 4 5 6 7]
```

Native `[oscparse]` pattern:

```text
[route av_gesture]
|
[route gesture]
|
[route cluster]
|
[i]
|
[sel -1 0 1 2 3 4 5 6 7]
```

Use this as a discrete selector, not a continuous modulation signal.

### `/av_gesture/gesture/name`

```text
symbol
```

Human-readable cluster name. The live model may send names from `cluster_names.json`; the synthetic test script sends names like:

```text
open_sweep
pinch_pulse
rising_arc
tight_tremor
wide_hold
falling_release
left_answer
two_hand_push
```

This is useful for debugging displays, but `gesture/cluster` is better for robust routing.

### `/av_gesture/gesture/confidence`

```text
float 0.0-1.0
```

Confidence or assignment strength. In the synthetic test script, this is shaped to feel plausible. In live model mode, the exact meaning depends on the cluster model:

```text
KMeans: usually 1.0 unless probability support exists
HDBSCAN: approximate assignment strength when available
```

Recommended Max pattern when routing full addresses:

```text
[route /av_gesture/gesture/confidence]
|
[f]
|
[scale 0. 1. 0. 127.]
```

### `/av_gesture/gesture/changed`

```text
int 0 or 1
```

Sends `1` when the smoothed gesture cluster changes. Use it for triggering one-shot synth events.

Recommended Max pattern when routing full addresses:

```text
[route /av_gesture/gesture/changed]
|
[i]
|
[sel 1]
```

### `/av_gesture/gesture/active`

```text
int 0 or 1
```

Master activity gate:

```text
0  no active hand/gesture
1  active hand/gesture
```

Use this to stop stale gestures from continuing to control the synth after the performer leaves the frame.

## Motion Messages

### `/av_gesture/motion/energy`

```text
float
```

Mean frame-to-frame motion over the rolling window. Higher means more movement.

Good mappings:

```text
amplitude
filter cutoff
noise amount
grain density
delay feedback
reverb send
```

Recommended Max pattern when routing full addresses:

```text
[route /av_gesture/motion/energy]
|
[f]
|
[slide 3 10]
|
[scale 0. 1. 100. 6000.]
```

### `/av_gesture/motion/window_ready`

```text
int 0 or 1
```

The model only predicts after the rolling window is full.

```text
0  warming up or no hand
1  model window is full and predictions are valid
```

Use this as a safety gate if you want to ignore startup values.

## Hand Presence Messages

### `/av_gesture/hand/num_hands`

```text
int 0, 1, or 2
```

Number of hands detected or simulated.

### `/av_gesture/hand/right/present`

```text
int 0 or 1
```

Right hand presence flag.

### `/av_gesture/hand/left/present`

```text
int 0 or 1
```

Left hand presence flag.

Recommended Max pattern when routing full addresses:

```text
[route /av_gesture/hand/right/present]
|
[i]
|
[sel 0 1]
```

## Selected Landmark Messages

Each selected landmark sends three floats:

```text
x y z
```

Available selected landmarks:

```text
/av_gesture/hand/right/wrist
/av_gesture/hand/right/thumb_tip
/av_gesture/hand/right/index_tip
/av_gesture/hand/right/middle_tip
/av_gesture/hand/right/ring_tip
/av_gesture/hand/right/pinky_tip

/av_gesture/hand/left/wrist
/av_gesture/hand/left/thumb_tip
/av_gesture/hand/left/index_tip
/av_gesture/hand/left/middle_tip
/av_gesture/hand/left/ring_tip
/av_gesture/hand/left/pinky_tip
```

Recommended Max pattern when routing full addresses:

```text
[route /av_gesture/hand/right/index_tip]
|
[unpack 0. 0. 0.]
|       |       |
x       y       z
```

Coordinates are wrist-relative and scale-normalized when using the current runtime export contract. That means:

```text
x  left/right offset from wrist
y  up/down offset from wrist
z  relative depth from wrist
```

These are not screen pixels. Treat them as normalized gesture-space values.

Good mappings:

```text
index_tip x  stereo pan, wavetable position, filter balance
index_tip y  pitch bend, filter cutoff, brightness
index_tip z  depth, reverb send, FM index, proximity effect
thumb_tip-index_tip distance  pinch control
left/right hand distance       spread, stereo width, harmonic tension
```

## Full Landmark Arrays

If full landmark output is enabled:

```text
/av_gesture/hand/right/landmarks
/av_gesture/hand/left/landmarks
```

Each message carries 63 floats:

```text
x0 y0 z0 x1 y1 z1 ... x20 y20 z20
```

MediaPipe landmark indices:

```text
0   wrist
4   thumb tip
8   index tip
12  middle tip
16  ring tip
20  pinky tip
```

Use full arrays only if you need detailed hand-shape mapping. For most Max synth work, the selected landmark messages are easier to patch and debug.

## Latent Vector

### `/av_gesture/latent`

```text
float list
```

The live model sends the GRU latent embedding when `send_latent` is enabled. The synthetic test script sends 16 floats by default and can be changed with:

```bash
python runtime/osc_synth_test.py --latent-dim 8
```

Recommended Max pattern for 16 values:

```text
[route /av_gesture/latent]
|
[unpack 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
```

Use latent values for advanced continuous mapping after the core signals are working. They are expressive, but less interpretable than `cluster`, `energy`, and landmarks.

## Synthetic Test Sender

Run this while building the Max patch:

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
performer  phrase-like cluster changes, hand movement, energy, confidence, latent data
pulse      sharp rhythmic energy spikes for envelope/gate mapping
sweep      slow continuous range changes
idle       no-hand state for gate/reset testing
```

Optional flags:

```text
--one-hand          send only right-hand data
--full-landmarks    also send 63-float landmark arrays
--clusters 4        limit synthetic cluster ids to 0-3
--latent-dim 8      send 8 latent floats instead of 16
--duration 0        run until Ctrl-C
```

## Minimal Max Patch Plan

Start with these four signals:

```text
/av_gesture/gesture/cluster
/av_gesture/motion/energy
/av_gesture/gesture/confidence
/av_gesture/hand/right/index_tip
```

If your patch routes full OSC addresses:

```text
[udpreceive 9000]
|
[route /av_gesture/gesture/cluster /av_gesture/motion/energy /av_gesture/gesture/confidence /av_gesture/hand/right/index_tip]
```

If you use native `[oscparse]`, route by address parts:

```text
[udpreceive 9000]
|
[oscparse]
|
[route av_gesture]
|
[route gesture motion hand]
```

Then unpack each outlet:

```text
cluster outlet      -> [i] -> [sel -1 0 1 2 3 4 5 6 7]
energy outlet       -> [f] -> [slide 3 10] -> synth continuous parameter
confidence outlet   -> [f] -> confidence gate or modulation depth
index_tip outlet    -> [unpack 0. 0. 0.] -> x/y/z synth controls
```

Once those work, add:

```text
/av_gesture/gesture/active
/av_gesture/gesture/changed
/av_gesture/hand/left/index_tip
/av_gesture/latent
```

## Common Max Gotchas

### `udpreceive 900` vs `udpreceive 9000`

The runtime and synthetic sender default to port `9000`. Use:

```text
[udpreceive 9000]
```

Or run the sender with a matching port:

```bash
python runtime/osc_synth_test.py --port 900
```

### Everything Looks Like A Float

The Python sender sends cluster, active, changed, presence, and hand count as ints. Max may display them as floats if the patch coerces them. Use `[i]` for discrete values.

### `[unpack 0. 0. 0.]` Only Fits XYZ Messages

This is correct for landmark messages:

```text
/av_gesture/hand/right/index_tip
```

It is not correct for scalar messages:

```text
/av_gesture/gesture/cluster
/av_gesture/motion/energy
/av_gesture/gesture/confidence
```

Use `[i]` or `[f]` for scalar messages.
