import numpy as np
from collections import deque

MP_LANDMARKS = 21
XYZ = 3
HAND_VEC = MP_LANDMARKS * XYZ


def landmarks_to_vec(hand_landmarks):
    """Convert a MediaPipe hand landmark list to flat xyz vector."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
    return pts.reshape(-1)


def normalize_hand_vec(vec, scale_landmark=9):
    """Normalize one hand relative to wrist and approximate hand size."""
    pts = vec.reshape(MP_LANDMARKS, XYZ).astype(np.float32)
    wrist = pts[0].copy()
    pts = pts - wrist
    scale = np.linalg.norm(pts[scale_landmark])
    if scale > 1e-6:
        pts = pts / scale
    return pts.reshape(-1)


def make_frame_feature(result, handedness=None, include_velocity=True, previous_feature=None, normalize=True):
    """
    Create a fixed feature vector:
      right hand 63 + left hand 63 + optional velocity 126 = 252
    If your Stage 2 model used a different feature dimension, update this function/config.
    """
    right = np.zeros(HAND_VEC, dtype=np.float32)
    left = np.zeros(HAND_VEC, dtype=np.float32)
    right_present = 0
    left_present = 0

    for i, hand in enumerate(result.hand_landmarks):
        vec = landmarks_to_vec(hand)
        if normalize:
            vec = normalize_hand_vec(vec)

        label = None
        if result.handedness and i < len(result.handedness) and len(result.handedness[i]) > 0:
            label = result.handedness[i][0].category_name.lower()

        # MediaPipe label is from the image perspective; keep it literal here.
        if label == "right" and right_present == 0:
            right = vec
            right_present = 1
        elif label == "left" and left_present == 0:
            left = vec
            left_present = 1
        elif right_present == 0:
            right = vec
            right_present = 1
        elif left_present == 0:
            left = vec
            left_present = 1

    base = np.concatenate([right, left]).astype(np.float32)

    if include_velocity:
        if previous_feature is None:
            vel = np.zeros_like(base)
        else:
            vel = base - previous_feature[:base.shape[0]]
        feat = np.concatenate([base, vel]).astype(np.float32)
    else:
        feat = base

    meta = {
        "right_present": right_present,
        "left_present": left_present,
        "num_hands": len(result.hand_landmarks),
        "right": right,
        "left": left,
    }
    return feat, meta


class RollingWindow:
    def __init__(self, window_size, feature_dim):
        self.window_size = int(window_size)
        self.feature_dim = int(feature_dim)
        self.buf = deque(maxlen=self.window_size)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.shape[0] != self.feature_dim:
            raise ValueError(f"Feature dim mismatch: got {x.shape[0]}, expected {self.feature_dim}")
        self.buf.append(x)

    @property
    def ready(self):
        return len(self.buf) == self.window_size

    def array(self):
        if not self.ready:
            return None
        return np.stack(list(self.buf), axis=0)

    def motion_energy(self):
        if len(self.buf) < 2:
            return 0.0
        arr = np.stack(list(self.buf), axis=0)
        diff = np.diff(arr, axis=0)
        return float(np.mean(np.linalg.norm(diff, axis=1)))
