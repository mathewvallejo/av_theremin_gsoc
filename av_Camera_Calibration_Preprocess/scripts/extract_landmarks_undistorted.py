import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def load_calibration(path):
    data = np.load(path, allow_pickle=True)
    return data["camera_matrix"], data["dist_coeffs"]


def flatten_hand(hand_landmarks):
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
    return pts.reshape(-1)


def main():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe landmarks from undistorted video frames."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--camera-calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="models/hand_landmarker.task")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="0 crops black borders; 1 keeps full field of view.")
    parser.add_argument("--num-hands", type=int, default=2)
    args = parser.parse_args()

    camera_matrix, dist_coeffs = load_calibration(args.camera_calibration)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), args.alpha, (w, h)
    )

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=args.model),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.num_hands,
    )

    rows = []
    frame_idx = 0

    with vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            undistorted = cv2.undistort(frame_bgr, camera_matrix, dist_coeffs, None, new_camera_matrix)
            frame_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int((frame_idx / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            row = {
                "video": str(args.video),
                "frame": frame_idx,
                "timestamp_ms": timestamp_ms,
                "num_hands": len(result.hand_landmarks),
            }

            for hand_i in range(2):
                if hand_i < len(result.hand_landmarks):
                    vec = flatten_hand(result.hand_landmarks[hand_i])
                else:
                    vec = np.full(63, np.nan, dtype=np.float32)
                for j, value in enumerate(vec):
                    row[f"hand{hand_i}_{j}"] = float(value)

            rows.append(row)
            frame_idx += 1

    cap.release()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Wrote {out}")
    print(f"Frames processed: {frame_idx}")


if __name__ == "__main__":
    main()
