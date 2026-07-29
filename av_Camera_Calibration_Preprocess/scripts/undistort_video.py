import argparse
from pathlib import Path
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--camera-calibration", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--alpha", type=float, default=0.0)
args = parser.parse_args()

data = np.load(args.camera_calibration, allow_pickle=True)
camera_matrix = data["camera_matrix"]
dist_coeffs = data["dist_coeffs"]

cap = cv2.VideoCapture(args.input)
if not cap.isOpened():
    raise RuntimeError(f"Could not open input video: {args.input}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), args.alpha, (w, h))

out = Path(args.output)
out.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    corrected = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    writer.write(corrected)
    n += 1

cap.release()
writer.release()
print(f"Wrote {out}")
print(f"Frames: {n}")
