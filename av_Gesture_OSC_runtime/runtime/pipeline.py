"""Shared inference pipeline for live camera and video replay modes.

Both live_camera_to_osc.py and replay_video_to_osc.py call run_pipeline()
with a pre-opened VideoCapture and a timestamp function. This avoids
duplicating the MediaPipe / feature / model / OSC logic in two places.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from feature_runtime import RollingWindow, make_frame_feature
from gesture_model_runtime import GestureRuntimeModel
from osc_sender import AVGestureOSCSender
from smoothing import ClusterSmoother


def build_landmarker(mp_cfg):
    """Construct and return a HandLandmarker from config."""
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=mp_cfg["model_path"]),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=int(mp_cfg.get("num_hands", 2)),
        min_hand_detection_confidence=float(mp_cfg.get("min_hand_detection_confidence", 0.5)),
        min_hand_presence_confidence=float(mp_cfg.get("min_hand_presence_confidence", 0.5)),
        min_tracking_confidence=float(mp_cfg.get("min_tracking_confidence", 0.5)),
    )
    return vision.HandLandmarker.create_from_options(options)


def build_gesture_model(model_cfg):
    """Construct and return the gesture runtime model from config."""
    return GestureRuntimeModel(
        encoder_path=model_cfg["encoder_path"],
        feature_scaler_path=model_cfg["feature_scaler_path"],
        cluster_model_path=model_cfg["cluster_model_path"],
        embedding_scaler_path=model_cfg.get("embedding_scaler_path"),
        cluster_names_path=model_cfg.get("cluster_names_path"),
        runtime_model_config_path=model_cfg["runtime_model_config_path"],
    )


def run_pipeline(cap, cfg, timestamp_fn, window_label="av_Gesture_OSC_runtime"):
    """Run the full inference pipeline on an open VideoCapture.

    Args:
        cap: cv2.VideoCapture already opened by the caller.
        cfg: dict loaded from runtime_config.yaml.
        timestamp_fn: callable(frame_idx) -> int timestamp in milliseconds.
            For live camera, use wall-clock time. For video replay, use
            frame_idx / fps * 1000.
        window_label: title for the OpenCV preview window.
    """
    feat_cfg = cfg["features"]
    osc_cfg = cfg["osc"]
    smooth_cfg = cfg["smoothing"]
    prev_cfg = cfg.get("preview", {})

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

    gesture_model = build_gesture_model(cfg["runtime_model"])
    previous_feature = None
    frame_idx = 0

    with build_landmarker(cfg["mediapipe"]) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            timestamp_ms = timestamp_fn(frame_idx)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            try:
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                feature, meta = make_frame_feature(
                    result,
                    include_velocity=feat_cfg.get("include_velocity", True),
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

                    # Use the model's name unless smoothing changed the cluster.
                    name = pred["name"] if smoothed_cluster == pred["cluster"] else f"cluster_{smoothed_cluster}"

                    sender.send_gesture(smoothed_cluster, pred["confidence"], name, changed)

                    if osc_cfg.get("send_latent", True):
                        sender.send_latent(pred["latent"])

            except Exception as e:
                # Log and skip frame rather than crashing mid-performance.
                print(f"Warning: error on frame {frame_idx}: {e}")

            if prev_cfg.get("show_window", True):
                cv2.imshow(window_label, frame_bgr)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

    cv2.destroyAllWindows()
