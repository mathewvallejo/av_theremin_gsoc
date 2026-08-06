import argparse
import math
import random
import time

from osc_sender import AVGestureOSCSender


GESTURE_NAMES = [
    "open_sweep",
    "pinch_pulse",
    "rising_arc",
    "tight_tremor",
    "wide_hold",
    "falling_release",
    "left_answer",
    "two_hand_push",
]

FINGER_TIPS = {
    "thumb_tip": 4,
    "index_tip": 8,
    "middle_tip": 12,
    "ring_tip": 16,
    "pinky_tip": 20,
}


def clamp(value, lo=0.0, hi=1.0):
    return max(lo, min(hi, value))


def smoothstep(x):
    x = clamp(x)
    return x * x * (3.0 - 2.0 * x)


def make_hand(t, side="right", amplitude=1.0, curl=0.2, tremor=0.0):
    """Return 21 wrist-relative, scale-normalized MediaPipe-like xyz points."""
    side_sign = 1.0 if side == "right" else -1.0
    phase = t * 2.0 * math.pi
    lateral = side_sign * 0.18 * math.sin(phase * 0.31)
    lift = 0.14 * math.sin(phase * 0.23 + side_sign * 0.7)
    depth = 0.08 * math.sin(phase * 0.19)
    jitter = tremor * math.sin(phase * 13.0 + side_sign)

    pts = [(0.0, 0.0, 0.0)] * 21
    bases = {
        "thumb": (0.24 * side_sign, -0.08, -0.02),
        "index": (0.18 * side_sign, -0.36, 0.0),
        "middle": (0.02 * side_sign, -0.42, 0.02),
        "ring": (-0.14 * side_sign, -0.36, 0.0),
        "pinky": (-0.28 * side_sign, -0.28, -0.03),
    }
    chains = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20],
    }
    lengths = {
        "thumb": 0.17,
        "index": 0.20,
        "middle": 0.22,
        "ring": 0.19,
        "pinky": 0.16,
    }

    for finger, ids in chains.items():
        base_x, base_y, base_z = bases[finger]
        length = lengths[finger] * amplitude
        finger_curl = curl * (1.15 if finger in ("ring", "pinky") else 1.0)
        spread = 0.025 * math.sin(phase * 0.47 + len(finger))
        for joint_i, landmark_i in enumerate(ids, start=1):
            frac = joint_i / 4.0
            bend = finger_curl * frac * frac
            x = base_x + spread * joint_i * side_sign + lateral + jitter
            y = base_y - length * frac * (1.0 - 0.55 * bend) + lift
            z = base_z + depth + bend * 0.16 + jitter * 0.4
            pts[landmark_i] = (x, y, z)

    return [coord for point in pts for coord in point]


def send_hand(sender, side, vec, present=True, full_landmarks=False):
    sender.send(f"/hand/{side}/present", int(bool(present)))
    if not present:
        return

    for name, landmark_i in {"wrist": 0, **FINGER_TIPS}.items():
        i = landmark_i * 3
        sender.send(f"/hand/{side}/{name}", vec[i], vec[i + 1], vec[i + 2])

    if full_landmarks:
        sender.send(f"/hand/{side}/landmarks", *vec)


def phrase_state(t, clusters):
    phrase_len = 2.4
    phrase_pos = (t % phrase_len) / phrase_len
    phrase_idx = int(t / phrase_len)
    cluster = phrase_idx % max(1, clusters)
    next_cluster = (phrase_idx + 1) % max(1, clusters)

    attack = smoothstep(min(1.0, phrase_pos / 0.22))
    release = 1.0 - smoothstep(max(0.0, (phrase_pos - 0.72) / 0.28))
    envelope = attack * release
    micro = 0.5 + 0.5 * math.sin(t * 2.0 * math.pi * 7.0)

    changed = phrase_pos < 0.04
    confidence = 0.55 + 0.4 * envelope
    energy = 0.06 + 0.62 * envelope + 0.10 * micro
    curl = 0.15 + 0.65 * (1.0 - envelope)
    tremor = 0.004 + 0.018 * micro if cluster in (1, 3) else 0.004

    if phrase_pos > 0.92:
        cluster = next_cluster
        confidence *= 0.75
        energy *= 0.45

    return {
        "cluster": cluster,
        "name": GESTURE_NAMES[cluster % len(GESTURE_NAMES)],
        "changed": changed,
        "confidence": confidence,
        "energy": energy,
        "curl": curl,
        "tremor": tremor,
        "envelope": envelope,
    }


def pulse_state(t, clusters):
    beat = 0.5 + 0.5 * math.sin(t * 2.0 * math.pi * 1.4)
    hit = beat > 0.92
    cluster = int(t * 1.4) % max(1, clusters)
    return {
        "cluster": cluster,
        "name": "pulse_hit" if hit else "pulse_wait",
        "changed": hit,
        "confidence": 0.65 + 0.3 * beat,
        "energy": 0.05 + 0.85 * (beat**5),
        "curl": 0.25 + 0.5 * beat,
        "tremor": 0.006,
        "envelope": beat,
    }


def sweep_state(t, clusters):
    cycle = 0.5 + 0.5 * math.sin(t * 2.0 * math.pi * 0.22)
    cluster = int(cycle * max(1, clusters - 1))
    return {
        "cluster": cluster,
        "name": "continuous_sweep",
        "changed": False,
        "confidence": 0.78,
        "energy": 0.16 + 0.34 * abs(math.cos(t * 2.0 * math.pi * 0.22)),
        "curl": 0.15 + 0.5 * cycle,
        "tremor": 0.002,
        "envelope": cycle,
    }


def idle_state():
    return {
        "cluster": -1,
        "name": "no_hand",
        "changed": False,
        "confidence": 0.0,
        "energy": 0.0,
        "curl": 0.5,
        "tremor": 0.0,
        "envelope": 0.0,
    }


def make_latent(t, cluster, latent_dim, energy):
    values = []
    base = max(0, cluster)
    for i in range(latent_dim):
        slow = math.sin(t * (0.37 + i * 0.031) + base * 0.9 + i)
        fast = math.sin(t * (2.1 + i * 0.07) + base)
        values.append((0.55 * slow + 0.18 * fast) * (0.3 + energy))
    return values


def state_for_mode(mode, t, clusters):
    if mode == "idle":
        return idle_state()
    if mode == "pulse":
        return pulse_state(t, clusters)
    if mode == "sweep":
        return sweep_state(t, clusters)
    return phrase_state(t, clusters)


def main():
    parser = argparse.ArgumentParser(description="Send synthetic av_gesture OSC data for Max/MSP synth mapping tests.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--prefix", default="/av_gesture")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=60.0, help="Seconds to run. Use 0 for forever.")
    parser.add_argument("--mode", choices=["performer", "pulse", "sweep", "idle"], default="performer")
    parser.add_argument("--clusters", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--one-hand", action="store_true", help="Only send right-hand data.")
    parser.add_argument("--full-landmarks", action="store_true", help="Also send 63-float landmark arrays for each active hand.")
    parser.add_argument("--print-every", type=float, default=1.0, help="Seconds between console status lines. Use 0 to disable.")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    fps = max(1.0, float(args.fps))
    frame_interval = 1.0 / fps
    sender = AVGestureOSCSender(host=args.host, port=args.port, prefix=args.prefix)

    print(
        f"Sending synthetic OSC to {args.host}:{args.port}{args.prefix} "
        f"mode={args.mode} fps={fps:.1f} duration={args.duration:g}s"
    )

    start = time.perf_counter()
    next_frame = start
    next_print = start
    last_cluster = None
    frame_idx = 0

    try:
        while args.duration <= 0.0 or (time.perf_counter() - start) < args.duration:
            now = time.perf_counter()
            if now < next_frame:
                time.sleep(next_frame - now)
                continue

            t = now - start
            state = state_for_mode(args.mode, t, args.clusters)
            active = state["cluster"] >= 0
            cluster_changed = state["changed"] or (state["cluster"] != last_cluster)
            last_cluster = state["cluster"]

            num_hands = 1 if args.one_hand else 2
            if not active:
                num_hands = 0

            sender.send("/hand/num_hands", num_hands)
            sender.send("/gesture/active", int(active))
            sender.send_motion(state["energy"], active)
            sender.send_gesture(state["cluster"], state["confidence"], state["name"], cluster_changed)
            sender.send("/latent", *make_latent(t, state["cluster"], args.latent_dim, state["energy"]))

            if active:
                right = make_hand(t, "right", amplitude=1.0 + 0.25 * state["envelope"], curl=state["curl"], tremor=state["tremor"])
                left = make_hand(t + 0.37, "left", amplitude=0.86 + 0.18 * state["envelope"], curl=1.0 - state["curl"] * 0.55, tremor=state["tremor"] * 0.7)
                send_hand(sender, "right", right, present=True, full_landmarks=args.full_landmarks)
                send_hand(sender, "left", left, present=not args.one_hand, full_landmarks=args.full_landmarks)
            else:
                send_hand(sender, "right", [], present=False)
                send_hand(sender, "left", [], present=False)

            if args.print_every > 0 and now >= next_print:
                print(
                    f"t={t:6.2f}s cluster={state['cluster']:2d} "
                    f"name={state['name']:<16} energy={state['energy']:.3f} "
                    f"confidence={state['confidence']:.3f} hands={num_hands}"
                )
                next_print = now + args.print_every

            frame_idx += 1
            next_frame = start + frame_idx * frame_interval

    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        sender.send("/hand/num_hands", 0)
        sender.send("/gesture/active", 0)
        sender.send("/hand/right/present", 0)
        sender.send("/hand/left/present", 0)
        sender.send_motion(0.0, False)
        sender.send_gesture(-1, 0.0, "no_hand", True)


if __name__ == "__main__":
    main()
