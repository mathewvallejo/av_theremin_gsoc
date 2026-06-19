import argparse
from pathlib import Path

import cv2
import numpy as np


def load_calibration(path):
    data = np.load(path, allow_pickle=True)
    return data["camera_matrix"], data["dist_coeffs"]


def main():
    parser = argparse.ArgumentParser(description="Export an undistorted copy of a video.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--camera-calibration", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="0 crops black borders; 1 keeps full field of view.")
    args = parser.parse_args()

    camera_matrix, dist_coeffs = load_calibration(args.camera_calibration)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), args.alpha, (w, h)
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    frame_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        corrected = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
        writer.write(corrected)
        frame_count += 1

    cap.release()
    writer.release()

    print(f"Wrote {out_path}")
    print(f"Frames: {frame_count}")


if __name__ == "__main__":
    main()
