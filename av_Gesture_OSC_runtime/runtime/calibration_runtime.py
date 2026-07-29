from pathlib import Path

import cv2
import numpy as np


class OptionalUndistorter:
    def __init__(self, cfg=None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.alpha = float(cfg.get("alpha", 0.0))
        self.camera_matrix = None
        self.dist_coeffs = None
        self.new_camera_matrix = None
        self.size = None

        if not self.enabled:
            return

        calibration_path = cfg.get("camera_calibration_path") or cfg.get("path")
        if not calibration_path:
            raise ValueError("calibration.enabled is true, but no camera_calibration_path was provided.")

        data = np.load(Path(calibration_path), allow_pickle=True)
        self.camera_matrix = data["camera_matrix"]
        self.dist_coeffs = data["dist_coeffs"]

    def apply(self, frame_bgr):
        if not self.enabled:
            return frame_bgr

        h, w = frame_bgr.shape[:2]
        size = (w, h)
        if self.new_camera_matrix is None or self.size != size:
            self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix,
                self.dist_coeffs,
                size,
                self.alpha,
                size,
            )
            self.size = size

        return cv2.undistort(
            frame_bgr,
            self.camera_matrix,
            self.dist_coeffs,
            None,
            self.new_camera_matrix,
        )
