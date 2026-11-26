"""MediaPipe BlazePose wrapper (Tasks API) used by every part of RehabNet.

Returns, for each frame:
  img  : (33, 4) image landmarks  -> x, y in pixels, z (relative), visibility
  world: (33, 3) world landmarks  -> metres, origin at hip centre (MediaPipe y points DOWN)
Missing detections are returned as NaN arrays so sequences keep their length.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(ROOT, "models")

# BlazePose landmark indices we use
MP = dict(
    nose=0, l_shoulder=11, r_shoulder=12, l_elbow=13, r_elbow=14, l_wrist=15, r_wrist=16,
    l_hip=23, r_hip=24, l_knee=25, r_knee=26, l_ankle=27, r_ankle=28, l_foot=31, r_foot=32,
)


@dataclass
class PoseFrame:
    img: np.ndarray    # (33,4)
    world: np.ndarray  # (33,3)
    ok: bool


class PoseExtractor:
    """Video-mode pose landmarker (uses temporal tracking between frames)."""

    def __init__(self, variant: str = "full", min_conf: float = 0.5):
        path = os.path.join(MODEL_DIR, f"pose_landmarker_{variant}.task")
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=path,
                delegate=(mp_python.BaseOptions.Delegate.GPU if os.environ.get("REHABNET_DELEGATE") == "gpu"
                          else mp_python.BaseOptions.Delegate.CPU)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_conf,
            min_pose_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
        )
        self.lm = vision.PoseLandmarker.create_from_options(opts)
        self._last_ts = -1

    def process(self, rgb: np.ndarray, ts_ms: int) -> PoseFrame:
        h, w = rgb.shape[:2]
        ts_ms = max(int(ts_ms), self._last_ts + 1)  # timestamps must increase
        self._last_ts = ts_ms
        res = self.lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), ts_ms)
        if not res.pose_landmarks:
            return PoseFrame(np.full((33, 4), np.nan), np.full((33, 3), np.nan), False)
        img = np.array([[p.x * w, p.y * h, p.z * w, p.visibility] for p in res.pose_landmarks[0]], dtype=np.float32)
        world = np.array([[p.x, p.y, p.z] for p in res.pose_world_landmarks[0]], dtype=np.float32)
        return PoseFrame(img, world, True)

    def close(self):
        self.lm.close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


def extract_video(path: str, variant: str = "full", every: int = 1, max_side: int | None = 960,
                  frame_range: tuple[int, int] | None = None):
    """Run the extractor over a video file. Returns dict of arrays (T,33,4), (T,33,3), frame idx, fps."""
    import cv2

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    start, stop = frame_range if frame_range else (0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    if start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    imgs, worlds, idx = [], [], []
    scale = 1.0
    with PoseExtractor(variant) as pe:
        f = start
        while f < stop:
            ok, frame = cap.read()
            if not ok:
                break
            if (f - start) % every == 0:
                if max_side and max(frame.shape[:2]) > max_side:
                    scale = max_side / max(frame.shape[:2])
                    frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                pf = pe.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), f * 1000.0 / fps)
                img = pf.img.copy()
                img[:, :3] /= scale  # back to original pixel units
                imgs.append(img)
                worlds.append(pf.world)
                idx.append(f)
            f += 1
    cap.release()
    return dict(img=np.stack(imgs), world=np.stack(worlds), frame=np.array(idx), fps=fps)
