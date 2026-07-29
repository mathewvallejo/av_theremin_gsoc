import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def collect_images(folder, extensions):
    folder = Path(folder)
    files = []
    for ext in extensions:
        files.extend(folder.rglob(f"*{ext}"))
        files.extend(folder.rglob(f"*{ext.upper()}"))
    return sorted(files)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--config", default="configs/calibration_config.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    board = cfg["checkerboard"]
    cal = cfg["calibration"]

    nx = int(board["inner_corners_x"])
    ny = int(board["inner_corners_y"])
    square = float(board["square_size_m"])
    pattern_size = (nx, ny)

    objp = np.zeros((ny * nx, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    objp *= square

    objpoints, imgpoints = [], []
    images = collect_images(args.images, cal.get("image_extensions", [".jpg", ".png"]))
    if not images:
        raise RuntimeError(f"No calibration images found in {args.images}")

    image_size = None
    good = 0

    for path in images:
        img = cv2.imread(str(path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]
        ok, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ok and cal.get("corner_refinement", True):
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        if ok:
            objpoints.append(objp)
            imgpoints.append(corners)
            good += 1

        if cal.get("show_preview", False):
            vis = img.copy()
            if ok:
                cv2.drawChessboardCorners(vis, pattern_size, corners, ok)
            cv2.imshow("calibration review", vis)
            cv2.waitKey(100)

    if cal.get("show_preview", False):
        cv2.destroyAllWindows()

    if good < 5:
        raise RuntimeError(f"Only found {good} usable chessboard images. Need more views.")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    total_error = 0.0
    total_points = 0
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
        err = cv2.norm(imgpoints[i], projected, cv2.NORM_L2)
        total_error += err * err
        total_points += len(objpoints[i])
    mean_error = float(np.sqrt(total_error / total_points))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        camera_id=args.camera_id,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_width=image_size[0],
        image_height=image_size[1],
        reprojection_error=mean_error,
        inner_corners_x=nx,
        inner_corners_y=ny,
        square_size_m=square,
        usable_images=good,
        total_images=len(images),
        rms_error=float(ret),
    )

    print(f"Saved calibration: {out}")
    print(f"Usable images: {good}/{len(images)}")
    print(f"RMS error: {ret:.4f}")
    print(f"Mean reprojection error: {mean_error:.4f} pixels")


if __name__ == "__main__":
    main()
