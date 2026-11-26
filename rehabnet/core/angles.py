"""Joint-angle computation on a common joint set shared by every skeleton source.

All sources (MediaPipe camera pose, OptiTrack mocap in REHAB24-6, Kinect in KIMORE / UI-PRMD)
are first mapped to the same 13 named joints, then the same angle definitions are applied, so
camera angles can be compared directly with mocap ground truth.
"""
from __future__ import annotations

import numpy as np

COMMON = ["neck", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
          "pelvis", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]

# source joint index for each common joint (neck/pelvis may be a mean of two joints)
MAPS = {
    # MediaPipe BlazePose 33 landmarks
    "mediapipe": {"l_shoulder": 11, "r_shoulder": 12, "l_elbow": 13, "r_elbow": 14, "l_wrist": 15,
                  "r_wrist": 16, "l_hip": 23, "r_hip": 24, "l_knee": 25, "r_knee": 26, "l_ankle": 27,
                  "r_ankle": 28, "neck": (11, 12), "pelvis": (23, 24)},
    # REHAB24-6 OptiTrack 26-joint skeleton (joints_names.txt)
    "optitrack": {"neck": 3, "l_shoulder": 7, "r_shoulder": 12, "l_elbow": 8, "r_elbow": 13,
                  "l_wrist": 9, "r_wrist": 14, "pelvis": 0, "l_hip": 16, "r_hip": 21, "l_knee": 17,
                  "r_knee": 22, "l_ankle": 18, "r_ankle": 23},
}
# Rehab-Pile skeleton layouts (KIMORE 18-joint Kinect, UI-PRMD 22-joint Vicon) given in the order
# pelvis, neck, head, l_sh, l_el, l_wr, r_sh, r_el, r_wr, l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle
_RP_ORDER = ["pelvis", "neck", None, "l_shoulder", "l_elbow", "l_wrist", "r_shoulder", "r_elbow", "r_wrist",
             "l_hip", "l_knee", "l_ankle", "r_hip", "r_knee", "r_ankle"]
for _name, _idx in {"kimore": [0, 17, 2, 3, 4, 5, 7, 8, 9, 11, 12, 13, 14, 15, 16],
                    "uiprmd": [0, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 16, 18, 19, 20]}.items():
    MAPS[_name] = {n: i for n, i in zip(_RP_ORDER, _idx) if n}

# angle name -> (a, b, c): angle at b between b->a and b->c, in degrees
ANGLES = {
    "l_elbow": ("l_shoulder", "l_elbow", "l_wrist"),
    "r_elbow": ("r_shoulder", "r_elbow", "r_wrist"),
    "l_shoulder": ("l_hip", "l_shoulder", "l_elbow"),   # arm elevation relative to trunk side
    "r_shoulder": ("r_hip", "r_shoulder", "r_elbow"),
    "l_hip": ("l_shoulder", "l_hip", "l_knee"),          # trunk-thigh angle
    "r_hip": ("r_shoulder", "r_hip", "r_knee"),
    "l_knee": ("l_hip", "l_knee", "l_ankle"),
    "r_knee": ("r_hip", "r_knee", "r_ankle"),
}
ANGLE_NAMES = list(ANGLES) + ["trunk_lean", "hip_abd_l", "hip_abd_r"]


def to_common(x: np.ndarray, source: str) -> np.ndarray:
    """(T, J_src, D) -> (T, 14, D) on the COMMON joint list."""
    m = MAPS[source]
    out = []
    for name in COMMON:
        i = m[name]
        out.append(x[:, list(i)].mean(1) if isinstance(i, tuple) else x[:, i])
    return np.stack(out, 1)


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    cos = np.sum(v1 * v2, -1) / np.where(n > 1e-9, n, np.nan)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def _vec_angle(v1, v2):
    n = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    cos = np.sum(v1 * v2, -1) / np.where(n > 1e-9, n, np.nan)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def joint_angles(c: np.ndarray, up: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """c: (T,14,D) common joints (D=2 or 3). up: 'up' direction of the coordinate system
    (e.g. [0,1,0] for mocap y-up, [0,-1,0] for MediaPipe y-down). Returns name -> (T,) degrees."""
    J = {n: c[:, i] for i, n in enumerate(COMMON)}
    out = {k: _angle(J[a], J[b], J[cc]) for k, (a, b, cc) in ANGLES.items()}
    trunk = J["neck"] - J["pelvis"]
    if up is None:
        up = np.zeros(c.shape[-1]); up[1] = -1.0  # image / MediaPipe convention: y grows downward
    out["trunk_lean"] = _vec_angle(trunk, np.broadcast_to(up, trunk.shape))
    # hip abduction: angle between thigh and the downward trunk axis, measured in the frontal plane
    pel_lr = J["l_hip"] - J["r_hip"]
    for s in ("l", "r"):
        thigh = J[f"{s}_knee"] - J[f"{s}_hip"]
        down = -trunk
        if c.shape[-1] == 3:
            # project onto the frontal plane spanned by trunk axis and pelvis left-right axis
            nrm = np.cross(trunk, pel_lr)
            nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True) + 1e-9
            thigh = thigh - np.sum(thigh * nrm, -1, keepdims=True) * nrm
        out[f"hip_abd_{s}"] = _vec_angle(thigh, down)
    return out


def angle_matrix(c, up=None) -> np.ndarray:
    a = joint_angles(c, up)
    return np.stack([a[k] for k in ANGLE_NAMES], -1)  # (T, n_angles)
