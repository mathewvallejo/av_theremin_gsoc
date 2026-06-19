"""Video replay mode: replays a recorded video and sends gesture OSC to Max/MSP.

Useful for testing the runtime bridge before using a live camera.
"""

import argparse
import time

import cv2
import yaml

from pipeline import run_pipeline


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Video replay gesture to OSC.")
    parser.add_argument("--config", default="configs/runtime_config.yaml")
    parser.add_argument("--video", required=True, help="Path to recorded video file.")
    parser.add_argument(
        "--realtime", action="store_true",
        help="Sleep between frames to match the source video FPS."
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_duration = 1.0 / fps

    def timestamp_fn(frame_idx):
        if args.realtime:
            time.sleep(max(0.0, frame_duration))
        return int((frame_idx / fps) * 1000)

    try:
        run_pipeline(cap, cfg, timestamp_fn, window_label="av_Gesture_OSC_runtime — replay")
    finally:
        cap.release()


if __name__ == "__main__":
    main()
