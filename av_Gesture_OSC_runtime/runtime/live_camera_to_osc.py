"""Live camera mode: reads from a webcam and sends gesture OSC to Max/MSP."""

import argparse
import time

import cv2
import yaml

from pipeline import run_pipeline


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Live camera gesture to OSC.")
    parser.add_argument("--config", default="configs/runtime_config.yaml")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index.")
    args = parser.parse_args()

    cfg = load_config(args.config)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    start = time.time()

    def timestamp_fn(frame_idx):
        # Use wall-clock time so timestamps are monotonic regardless of frame drops.
        return int((time.time() - start) * 1000)

    try:
        run_pipeline(cap, cfg, timestamp_fn, window_label="av_Gesture_OSC_runtime — live")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
