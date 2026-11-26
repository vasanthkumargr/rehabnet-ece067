"""Keypoint normalisation and temporal smoothing.

Bounding-box normalisation: every frame, keypoints are expressed relative to a person
bounding box whose centre and size are themselves smoothed over time. This removes the effect
of where the person stands in the image and how far they are from the camera, so the same
movement gives the same numbers on any device.

One-Euro filter (Casiez et al., CHI 2012): adaptive low-pass filter widely used for live
pose tracking. It removes jitter when the joint is still and keeps lag low when it moves fast.
"""
from __future__ import annotations

import numpy as np

# BlazePose indices of shoulders, elbows, wrists, hips, knees, ankles (kept here so this module does not need
# MediaPipe installed; the server only needs numpy)
BODY = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def _interp_nan(x: np.ndarray) -> np.ndarray:
    """Linear interpolation of NaN runs along time (axis 0) for every other axis."""
    x = x.copy()
    flat = x.reshape(len(x), -1)
    t = np.arange(len(x))
    for j in range(flat.shape[1]):
        v = flat[:, j]
        m = np.isfinite(v)
        if m.sum() == 0:
            continue
        if (~m).any():
            v[~m] = np.interp(t[~m], t[m], v[m])
    return flat.reshape(x.shape)


class OneEuro:
    def __init__(self, fps=30.0, min_cutoff=1.0, beta=0.05, d_cutoff=1.0):
        self.fps, self.min_cutoff, self.beta, self.d_cutoff = fps, min_cutoff, beta, d_cutoff
        self.x_prev = None
        self.dx_prev = None

    @staticmethod
    def _alpha(cutoff, fps):
        tau = 1.0 / (2 * np.pi * cutoff)
        return 1.0 / (1.0 + tau * fps)

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.x_prev is None or not np.all(np.isfinite(self.x_prev)):
            self.x_prev, self.dx_prev = x, np.zeros_like(x)
            return x
        dx = (x - self.x_prev) * self.fps
        a_d = self._alpha(self.d_cutoff, self.fps)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, self.fps)
        x_hat = a * x + (1 - a) * self.x_prev
        x_hat = np.where(np.isfinite(x), x_hat, self.x_prev)
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat


def one_euro_sequence(seq: np.ndarray, fps=30.0, min_cutoff=1.0, beta=0.05) -> np.ndarray:
    f = OneEuro(fps, min_cutoff, beta)
    return np.stack([f(s) for s in seq])


def bbox_normalize(xy: np.ndarray, vis: np.ndarray | None = None, vis_thr=0.5, smooth=0.8,
                   pad=0.1, body=None) -> tuple[np.ndarray, np.ndarray]:
    """Normalise 2-D keypoints (T,J,2) by a temporally smoothed person bounding box.

    Returns normalised keypoints (T,J,2) in roughly [-0.5,0.5] and the boxes (T,4) = cx, cy, w, h.
    The box uses the larger side (square) so the aspect ratio of the body is preserved.
    """
    body = BODY if body is None else body
    T = len(xy)
    boxes = np.full((T, 4), np.nan)
    state = None
    for t in range(T):
        pts = xy[t, body]
        m = np.all(np.isfinite(pts), axis=1)
        if vis is not None:
            m &= vis[t, body] > vis_thr
        if m.sum() >= 4:
            lo, hi = pts[m].min(0), pts[m].max(0)
            c = (lo + hi) / 2
            s = max(hi - lo) * (1 + 2 * pad)
            cur = np.array([c[0], c[1], s, s])
            state = cur if state is None else smooth * state + (1 - smooth) * cur
        boxes[t] = state if state is not None else np.nan
    boxes = _interp_nan(boxes)
    norm = (xy - boxes[:, None, :2]) / boxes[:, None, 2:3]
    return norm, boxes


def torso_normalize_3d(world: np.ndarray, l_sh, r_sh, l_hip, r_hip) -> np.ndarray:
    """Centre 3-D joints on the hip midpoint and scale by torso length (hip-mid to shoulder-mid)."""
    hip = (world[:, l_hip] + world[:, r_hip]) / 2
    sh = (world[:, l_sh] + world[:, r_sh]) / 2
    L = np.linalg.norm(sh - hip, axis=-1)
    L = np.where(L > 1e-6, L, np.nan)
    return (world - hip[:, None]) / L[:, None, None]


def jitter(x: np.ndarray) -> float:
    """Mean absolute second difference (acceleration) - a standard jitter measure."""
    a = np.diff(x, n=2, axis=0)
    return float(np.nanmean(np.linalg.norm(a, axis=-1)))


def torso_normalize_2d(xy: np.ndarray, l_sh, r_sh, l_hip, r_hip, window=45) -> tuple[np.ndarray, np.ndarray]:
    """Hip-centred, torso-scaled 2-D normalisation (causal, usable live).

    Centre = hip midpoint of the current frame. Scale = running median of the torso length
    (shoulder midpoint to hip midpoint) over the last `window` frames, so a raised arm or a bent
    leg does not change the scale the way a bounding box does.
    Returns (T,J,2) normalised keypoints and the (T,) scale used."""
    hip = (xy[:, l_hip] + xy[:, r_hip]) / 2
    sh = (xy[:, l_sh] + xy[:, r_sh]) / 2
    L = np.linalg.norm(sh - hip, axis=-1)
    scale = np.full(len(xy), np.nan)
    for t in range(len(xy)):
        w = L[max(0, t - window + 1): t + 1]
        w = w[np.isfinite(w)]
        if len(w):
            scale[t] = np.median(w)
    scale = _interp_nan(scale[:, None])[:, 0]
    return (xy - hip[:, None]) / scale[:, None, None], scale
