import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand landmarks from undistorted video frames. Supports one video or a folder of videos."
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--video", help="Path to one input video file.")
    input_group.add_argument("--video-dir", help="Path to a folder containing video files.")

    parser.add_argument("--camera-calibration", required=True, help="Path to camera intrinsics .npz file.")

    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output", help="Output CSV path. Use this with --video.")
    output_group.add_argument("--output-dir", help="Output folder. Use this with --video-dir.")

    parser.add_argument("--model", default="models/hand_landmarker.task", help="Path to MediaPipe hand landmarker model.")
    parser.add_argument("--alpha", type=float, default=0.0, help="Alpha for cv2.getOptimalNewCameraMatrix.")
    parser.add_argument("--num-hands", type=int, default=2, help="Maximum number of hands to detect.")

    args = parser.parse_args()

    if args.video and args.output_dir:
        parser.error("Use --output with --video, not --output-dir.")
    if args.video_dir and args.output:
        parser.error("Use --output-dir with --video-dir, not --output.")

    return args


def find_videos(video_dir):
    video_dir = Path(video_dir)
    videos = [p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS]
    return sorted(videos)


def extract_landmarks_for_video(
    video_path,
    output_csv,
    camera_matrix,
    dist_coeffs,
    landmarker_options,
    alpha=0.0,
):
    video_path = Path(video_path)
    output_csv = Path(output_csv)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Skipping unreadable video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if w <= 0 or h <= 0:
        cap.release()
        print(f"Skipping video with invalid dimensions: {video_path}")
        return 0

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (w, h),
        alpha,
        (w, h),
    )

    rows = []
    frame_idx = 0

    # Create a fresh VIDEO-mode landmarker for each video.
    # MediaPipe requires timestamps to be monotonically increasing within one landmarker session,
    # so a new session lets each video safely start timestamps again at 0 ms.
    with vision.HandLandmarker.create_from_options(landmarker_options) as landmarker:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            undistorted = cv2.undistort(
                frame_bgr,
                camera_matrix,
                dist_coeffs,
                None,
                new_camera_matrix,
            )
            frame_rgb = cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            timestamp_ms = int((frame_idx / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            row = {
                "video": str(video_path),
                "frame": frame_idx,
                "timestamp_ms": timestamp_ms,
                "num_hands": len(result.hand_landmarks),
            }

            for hand_i in range(2):
                if hand_i < len(result.hand_landmarks):
                    pts = np.array(
                        [[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[hand_i]],
                        dtype=np.float32,
                    ).reshape(-1)
                else:
                    pts = np.full(63, np.nan, dtype=np.float32)

                for j, value in enumerate(pts):
                    row[f"hand{hand_i}_{j}"] = float(value)

            rows.append(row)
            frame_idx += 1

    cap.release()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)

    print(f"Wrote {output_csv}")
    print(f"Frames: {frame_idx}")
    return frame_idx


def main():
    args = parse_args()

    data = np.load(args.camera_calibration, allow_pickle=True)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=args.model),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=args.num_hands,
    )

    if args.video:
        jobs = [(Path(args.video), Path(args.output))]
    else:
        videos = find_videos(args.video_dir)
        if not videos:
            raise RuntimeError(f"No video files found in: {args.video_dir}")

        output_dir = Path(args.output_dir)
        jobs = [(video, output_dir / f"{video.stem}_landmarks.csv") for video in videos]

    for video_path, output_csv in jobs:
        print(f"Processing {video_path}")
        extract_landmarks_for_video(
            video_path=video_path,
            output_csv=output_csv,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            landmarker_options=options,
            alpha=args.alpha,
        )


if __name__ == "__main__":
    main()
