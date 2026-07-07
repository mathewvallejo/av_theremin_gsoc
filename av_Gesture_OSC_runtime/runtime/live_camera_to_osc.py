import argparse
from pathlib import Path
import time

import cv2
import mediapipe as mp
import yaml
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from feature_runtime import RollingWindow, make_frame_feature
from gesture_model_runtime import GestureRuntimeModel
from osc_sender import AVGestureOSCSender
from smoothing import ClusterSmoother


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def osc_safe_meta(meta):
    """Do not send zero-filled fake landmarks for an absent hand."""
    out = dict(meta)
    if not int(out.get("right_present", 0)):
        out["right"] = None
    if not int(out.get("left_present", 0)):
        out["left"] = None
    return out


def draw_mediapipe_landmarks(preview_frame, result):
    if not result.hand_landmarks:
        return preview_frame

    for hand_landmarks in result.hand_landmarks:
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z)
            for lm in hand_landmarks
        ])

        solutions.drawing_utils.draw_landmarks(
            preview_frame,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

    return preview_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/runtime_config.yaml")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)

    mp_cfg = cfg["mediapipe"]
    feat_cfg = cfg["features"]
    osc_cfg = cfg["osc"]
    smooth_cfg = cfg["smoothing"]
    prev_cfg = cfg.get("preview", {})

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
    gesture_model = GestureRuntimeModel(
        encoder_path=model_cfg["encoder_path"],
        feature_scaler_path=model_cfg["feature_scaler_path"],
        cluster_model_path=model_cfg["cluster_model_path"],
        embedding_scaler_path=model_cfg.get("embedding_scaler_path"),
        cluster_names_path=model_cfg.get("cluster_names_path"),
        runtime_model_config_path=model_cfg["runtime_model_config_path"],
    )

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

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    previous_feature = None
    start = time.time()
    was_active = False

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            timestamp_ms = int((time.time() - start) * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            num_hands = len(result.hand_landmarks) if result.hand_landmarks else 0

            # Always report hand presence first. Max can use this as the master gate.
            sender.send("/hand/num_hands", int(num_hands))
            sender.send("/gesture/active", int(num_hands > 0))

            if num_hands == 0:
                # Clear temporal state so old frames do not keep producing gestures
                # after the hand leaves the camera frame.
                previous_feature = None
                window = RollingWindow(feat_cfg["window_size"], feat_cfg["feature_dim"])
                smoother = ClusterSmoother(
                    history=smooth_cfg.get("cluster_history", 7),
                    hold_last_valid=smooth_cfg.get("hold_last_valid", True),
                )

                # Send an explicit idle/no-hand state. This prevents Max from
                # continuing to act on the last valid gesture.
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

                if prev_cfg.get("show_window", True):
                    cv2.imshow("av_Gesture_OSC_runtime", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            was_active = True

            feature, meta = make_frame_feature(
                result,
                include_velocity=feat_cfg.get("include_velocity", False),
                previous_feature=previous_feature,
                normalize=feat_cfg.get("normalize_to_wrist", True),
            )
            previous_feature = feature.copy()

            meta_for_osc = osc_safe_meta(meta)
            sender.send_hand_meta(meta_for_osc)
            sender.send_selected_landmarks(meta_for_osc)

            if osc_cfg.get("send_full_landmarks", True):
                sender.send_full_landmarks(meta_for_osc)

            window.append(feature)
            energy = window.motion_energy()
            sender.send_motion(energy, window.ready)

            if window.ready:
                pred = gesture_model.infer(window.array())
                smoothed_cluster, changed = smoother.update(pred["cluster"])
                name = pred["name"]
                if smoothed_cluster != pred["cluster"]:
                    name = f"cluster_{smoothed_cluster}"

                sender.send_gesture(smoothed_cluster, pred["confidence"], name, changed)

                if osc_cfg.get("send_latent", True):
                    sender.send_latent(pred["latent"])

            if prev_cfg.get("show_window", True):
                preview_frame = frame_bgr.copy()
                if prev_cfg.get("draw_landmarks", False):
                    preview_frame = draw_mediapipe_landmarks(preview_frame, result)

                cv2.imshow("av_Gesture_OSC_runtime", preview_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
