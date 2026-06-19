# OSC Message Reference

Default prefix:

```text
/av_gesture
```

## Gesture Messages

```text
/av_gesture/gesture/cluster <int>
/av_gesture/gesture/confidence <float>
/av_gesture/gesture/name <symbol>
/av_gesture/gesture/changed <0_or_1>
```

## Latent Vector

```text
/av_gesture/latent <float_0> <float_1> ... <float_n>
```

## Motion

```text
/av_gesture/motion/energy <float>
/av_gesture/motion/window_ready <0_or_1>
```

## Hand Presence

```text
/av_gesture/hand/num_hands <int>
/av_gesture/hand/right/present <0_or_1>
/av_gesture/hand/left/present <0_or_1>
```

## Selected Landmarks

Coordinates are normalized MediaPipe coordinates unless your feature configuration changes them.

```text
/av_gesture/hand/right/wrist <x> <y> <z>
/av_gesture/hand/right/index_tip <x> <y> <z>
/av_gesture/hand/right/middle_tip <x> <y> <z>
/av_gesture/hand/right/thumb_tip <x> <y> <z>

/av_gesture/hand/left/wrist <x> <y> <z>
/av_gesture/hand/left/index_tip <x> <y> <z>
/av_gesture/hand/left/middle_tip <x> <y> <z>
/av_gesture/hand/left/thumb_tip <x> <y> <z>
```

## Full Landmark Array

If enabled:

```text
/av_gesture/hand/right/landmarks <63 floats>
/av_gesture/hand/left/landmarks <63 floats>
```

The order is:

```text
x0 y0 z0 x1 y1 z1 ... x20 y20 z20
```

MediaPipe hand landmark indices:

```text
0 wrist
4 thumb tip
8 index tip
12 middle tip
16 ring tip
20 pinky tip
```
