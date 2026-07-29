import argparse
import time

import cv2
import mediapipe as mp
import yaml
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from calibration_runtime import OptionalUndistorter
from feature_runtime import RollingWindow, make_frame_feature
from gesture_model_runtime import GestureRuntimeModel
from live_camera_to_osc import draw_cluster_overlay, draw_mediapipe_landmarks, osc_safe_meta
from osc_sender import AVGestureOSCSender
from smoothing import ClusterSmoother


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/runtime_config.yaml")
    parser.add_argument("--video", required=True)
    parser.add_argument("--realtime", action="store_true", help="Sleep according to source FPS while sending OSC.")
    parser.add_argument("--fps", type=float, default=None, help="Override video FPS metadata for replay pacing and MediaPipe timestamps.")
    parser.add_argument(
        "--drop-late-frames",
        action="store_true",
        help="When --realtime is active, skip unread video frames if processing falls behind recorded time.",
    )
    parser.add_argument(
        "--max-frame-drops",
        type=int,
        default=120,
        help="Maximum consecutive frames to skip while catching up with --drop-late-frames.",
    )
    parser.add_argument("--model-dir", default=None, help="Override runtime_model.artifact_dir with a Stage 2 export_for_runtime folder.")
    parser.add_argument("--debug", action="store_true", help="Print replay timing and hand detection diagnostics.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model_dir:
        cfg.setdefault("runtime_model", {})["artifact_dir"] = args.model_dir

    mp_cfg = cfg["mediapipe"]
    feat_cfg = cfg["features"]
    osc_cfg = cfg["osc"]
    smooth_cfg = cfg["smoothing"]
    prev_cfg = cfg.get("preview", {})
    undistorter = OptionalUndistorter(cfg.get("calibration", {}))

    idle_cluster = int(osc_cfg.get("idle_cluster", -1))
    send_idle_every_frame = bool(osc_cfg.get("send_idle_every_frame", True))

    BaseOptions = python.BaseOptions
    RunningMode = vision.RunningMode

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=mp_cfg["model_path"]),
        running_mode=RunningMode.VIDEO,
        num_hands=int(mp_cfg.get("num_hands", 2)),
        min_hand_detection_confidence=float(mp_cfg.get("min_hand_detection_confidence", 0.5)),
        min_hand_presence_confidence=float(mp_cfg.get("min_hand_presence_confidence", 0.5)),
        min_tracking_confidence=float(mp_cfg.get("min_tracking_confidence", 0.5)),
    )

    model_cfg = cfg["runtime_model"]
    gesture_model = GestureRuntimeModel.from_config(model_cfg)

    # Let the Stage 2 runtime_model_config.json override feature settings so
    # the rolling window matches the trained model. YAML values remain fallbacks.
    exported_feat_cfg = gesture_model.cfg.get("features", {})
    feat_cfg = {**feat_cfg, **{k: v for k, v in exported_feat_cfg.items() if v is not None}}
    if "sequence_length" in exported_feat_cfg:
        feat_cfg["window_size"] = exported_feat_cfg["sequence_length"]
    if "feature_dim" not in feat_cfg and "motion_dim" in exported_feat_cfg:
        feat_cfg["feature_dim"] = exported_feat_cfg["motion_dim"]

    sender = AVGestureOSCSender(
        host=osc_cfg.get("host", "127.0.0.1"),
        port=osc_cfg.get("port", 9000),
        prefix=osc_cfg.get("prefix", "/av_gesture"),
    )

    window = RollingWindow(feat_cfg["window_size"], feat_cfg["feature_dim"])
    smoother = ClusterSmoother(
        history=smooth_cfg.get("cluster_history", 7),
        hold_last_valid=smooth_cfg.get("hold_last_valid", True),
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {args.video}")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    fps = float(args.fps) if args.fps else source_fps
    if fps <= 1e-6:
        fps = 30.0
    frame_interval = 1.0 / fps
    previous_feature = None
    frame_idx = 0
    playback_start = time.perf_counter()
    was_active = False
    display_cluster = None
    display_confidence = 0.0
    display_name = "warming"
    display_changed = False
    display_energy = 0.0

    if args.debug:
        print(f"Replay source FPS: {source_fps:.3f}; pacing FPS: {fps:.3f}")

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            frame_start = time.perf_counter()
            ok, frame_bgr = cap.read()
            if not ok:
                break

            timestamp_ms = int((frame_idx / fps) * 1000)
            frame_bgr = undistorter.apply(frame_bgr)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0

            sender.send("/hand/num_hands", int(num_hands))
            sender.send("/gesture/active", int(num_hands > 0))

            if num_hands == 0:
                previous_feature = None
                window = RollingWindow(feat_cfg["window_size"], feat_cfg["feature_dim"])
                smoother = ClusterSmoother(
                    history=smooth_cfg.get("cluster_history", 7),
                    hold_last_valid=smooth_cfg.get("hold_last_valid", True),
                )

                if send_idle_every_frame or was_active:
                    sender.send("/hand/right/present", 0)
                    sender.send("/hand/left/present", 0)
                    sender.send("/motion/energy", 0.0)
                    sender.send("/motion/window_ready", 0)
                    sender.send("/gesture/cluster", idle_cluster)
                    sender.send("/gesture/confidence", 0.0)
                    sender.send("/gesture/name", "no_hand")
                    sender.send("/gesture/changed", int(was_active))

                was_active = False
                display_cluster = idle_cluster
                display_confidence = 0.0
                display_name = "no_hand"
                display_changed = False
                display_energy = 0.0
            else:
                was_active = True

                feature, meta = make_frame_feature(
                    result,
                    include_velocity=feat_cfg.get("include_velocity", False),
                    previous_feature=previous_feature,
                    normalize=feat_cfg.get("normalize_to_wrist", True),
                    normalize_scale_landmark=feat_cfg.get("normalize_scale_landmark", 9),
                    hand_order=feat_cfg.get("hand_order", "label"),
                )
                previous_feature = feature.copy()

                window.append(feature)
                meta_for_osc = osc_safe_meta(meta)
                sender.send_hand_meta(meta_for_osc)
                sender.send_selected_landmarks(meta_for_osc)

                if osc_cfg.get("send_full_landmarks", True):
                    sender.send_full_landmarks(meta_for_osc)

                energy = window.motion_energy()
                display_energy = energy
                sender.send_motion(energy, window.ready)

                if window.ready:
                    pred = gesture_model.infer(window.array())
                    smoothed_cluster, changed = smoother.update(pred["cluster"])
                    name = pred["name"]
                    if smoothed_cluster != pred["cluster"]:
                        name = f"cluster_{smoothed_cluster}"

                    sender.send_gesture(smoothed_cluster, pred["confidence"], name, changed)
                    display_cluster = smoothed_cluster
                    display_confidence = pred["confidence"]
                    display_name = name
                    display_changed = changed

                    if osc_cfg.get("send_latent", True):
                        sender.send_latent(pred["latent"])

            if prev_cfg.get("show_window", True):
                preview_frame = frame_bgr.copy()
                if prev_cfg.get("draw_landmarks", False):
                    preview_frame = draw_mediapipe_landmarks(preview_frame, result)

                preview_frame = draw_cluster_overlay(
                    preview_frame,
                    cluster=display_cluster,
                    confidence=display_confidence,
                    name=display_name,
                    changed=display_changed,
                    active=num_hands > 0,
                    window_fill=len(window.buf),
                    window_size=feat_cfg["window_size"],
                    motion_energy=display_energy,
                )
                cv2.imshow("av_Gesture_OSC_runtime replay", preview_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.debug and frame_idx % max(1, int(round(fps))) == 0:
                processing_ms = (time.perf_counter() - frame_start) * 1000.0
                ready = int(window.ready)
                print(
                    f"frame={frame_idx} t={timestamp_ms}ms hands={num_hands} "
                    f"window={len(window.buf)}/{feat_cfg['window_size']} ready={ready} "
                    f"cluster={display_cluster} processing={processing_ms:.1f}ms"
                )

            frame_idx += 1

            if args.realtime:
                target_time = playback_start + (frame_idx * frame_interval)
                delay = target_time - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                elif args.drop_late_frames:
                    dropped = 0
                    while dropped < max(0, int(args.max_frame_drops)):
                        next_frame_target = playback_start + ((frame_idx + 1) * frame_interval)
                        if next_frame_target > time.perf_counter():
                            break
                        if not cap.grab():
                            break
                        frame_idx += 1
                        dropped += 1

                    if args.debug and dropped:
                        late_ms = max(0.0, (time.perf_counter() - (playback_start + (frame_idx * frame_interval))) * 1000.0)
                        print(f"dropped={dropped} next_frame={frame_idx} still_late={late_ms:.1f}ms")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
