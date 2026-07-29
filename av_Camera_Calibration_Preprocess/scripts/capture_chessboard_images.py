import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def draw_status(frame, lines):
    y = 30
    for line in lines:
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        y += 30


def main():
    parser = argparse.ArgumentParser(description="Live chessboard calibration capture with visual feedback.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/calibration_config.yaml")
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    board = cfg["checkerboard"]
    cap_cfg = cfg.get("capture", {})

    pattern_size = (int(board["inner_corners_x"]), int(board["inner_corners_y"]))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ext = cap_cfg.get("image_extension", ".png")
    autosave_interval = float(cap_cfg.get("autosave_interval_seconds", 1.0))
    min_pose_change = float(cap_cfg.get("min_pose_change_pixels", 25.0))
    mirror_preview = bool(cap_cfg.get("mirror_preview", False))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps:
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    count = len(list(out_dir.glob(f"{args.camera_id}_calib_*{ext}")))
    autosave = False
    last_save_time = 0.0
    last_corner_mean = None

    print("Controls: SPACE save, a autosave, q quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        display = frame.copy()
        if mirror_preview:
            display = cv2.flip(display, 1)

        corner_mean = None
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, pattern_size, corners, found)
            corner_mean = np.mean(corners.reshape(-1, 2), axis=0)

        now = time.time()
        pose_changed = False
        if corner_mean is not None:
            if last_corner_mean is None:
                pose_changed = True
            else:
                pose_changed = np.linalg.norm(corner_mean - last_corner_mean) >= min_pose_change

        if autosave and found and pose_changed and (now - last_save_time) >= autosave_interval:
            fname = out_dir / f"{args.camera_id}_calib_{count:04d}{ext}"
            cv2.imwrite(str(fname), frame)
            print(f"Saved {fname}")
            count += 1
            last_save_time = now
            last_corner_mean = corner_mean

        draw_status(display, [
            f"camera: {args.camera_id} index={args.camera}",
            f"found chessboard: {found}",
            f"saved frames: {count}",
            f"autosave: {'ON' if autosave else 'OFF'}",
            "SPACE save | a autosave | q quit",
        ])

        cv2.imshow("av_Camera_Calibration_Preprocess capture", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        if key == ord("a"):
            autosave = not autosave
            print(f"Autosave {'ON' if autosave else 'OFF'}")
        if key == 32:
            if found:
                fname = out_dir / f"{args.camera_id}_calib_{count:04d}{ext}"
                cv2.imwrite(str(fname), frame)
                print(f"Saved {fname}")
                count += 1
                last_save_time = now
                last_corner_mean = corner_mean
            else:
                print("Chessboard not detected; frame not saved.")

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved {count} total frames in {out_dir}")


if __name__ == "__main__":
    main()
