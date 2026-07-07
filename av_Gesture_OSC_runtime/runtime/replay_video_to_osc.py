import argparse
import time

import cv2
import mediapipe as mp
import yaml
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from feature_runtime import RollingWindow, make_frame_feature
from gesture_model_runtime import GestureRuntimeModel
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
    args = parser.parse_args()

    cfg = load_config(args.config)

    mp_cfg = cfg["mediapipe"]
    feat_cfg = cfg["features"]
    osc_cfg = cfg["osc"]
    smooth_cfg = cfg["smoothing"]
    prev_cfg = cfg.get("preview", {})

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

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    previous_feature = None
    frame_idx = 0

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            timestamp_ms = int((frame_idx / fps) * 1000)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            feature, meta = make_frame_feature(
                result,
                include_velocity=feat_cfg.get("include_velocity", False),
                previous_feature=previous_feature,
                normalize=feat_cfg.get("normalize_to_wrist", True),
            )
            previous_feature = feature.copy()

            window.append(feature)
            sender.send_hand_meta(meta)
            sender.send_selected_landmarks(meta)

            if osc_cfg.get("send_full_landmarks", True):
                sender.send_full_landmarks(meta)

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
                cv2.imshow("av_Gesture_OSC_runtime replay", frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.realtime:
                time.sleep(max(0.0, 1.0 / fps))

            frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
