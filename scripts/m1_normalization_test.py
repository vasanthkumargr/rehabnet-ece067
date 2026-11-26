"""Milestone 1 - does keypoint normalisation make the pipeline independent of camera distance and position?

The same video frames are shown to MediaPipe three ways:
  original   : the camera frame as recorded
  far_left   : person shrunk to 55 % (as if standing ~1.8x further away), shifted to the left
  far_right  : shrunk to 70 %, shifted right and down
For each normalisation (none = raw pixels, bounding box, torso) we measure how far the normalised keypoints of the
altered frame are from those of the original frame, as a percentage of the torso length. 0 % = perfectly invariant.

Output: results/m1/normalisation_invariance.csv, results/m1/normalisation_invariance_by_cond.csv
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rehabnet.core.angles import COMMON, angle_matrix, to_common  # noqa: E402
from rehabnet.core.normalize import bbox_normalize, torso_normalize_2d  # noqa: E402
from rehabnet.core.pose import PoseExtractor  # noqa: E402

D = os.path.join(ROOT, "data", "rehab24_6")
RES = os.path.join(ROOT, "results", "m1")
LIMB = [COMMON.index(j) for j in ("l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
                                    "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle")]
CONDS = {"original": (1.0, 0.5, 0.5), "far_left": (0.55, 0.22, 0.45), "far_right": (0.70, 0.75, 0.62)}
I = {k: COMMON.index(k) for k in ("l_shoulder", "r_shoulder", "l_hip", "r_hip", "neck", "pelvis")}


def warp(frame, s, cx, cy):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    canvas = np.full_like(frame, int(frame.mean()))
    x0 = int(np.clip(cx * w - small.shape[1] / 2, 0, w - small.shape[1]))
    y0 = int(np.clip(cy * h - small.shape[0] / 2, 0, h - small.shape[0]))
    canvas[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    return canvas


def main(n_per_ex=2, max_frames=240, step=3):
    seg = pd.read_csv(os.path.join(D, "Segmentation.csv"), sep=";")
    rows = []
    for ex in range(1, 7):
        vids = seg[seg.exercise_id == ex].video_id.unique()[:n_per_ex]
        for vid in vids:
            g = seg[seg.video_id == vid]
            a, b = g.first_frame.min(), g.last_frame.max()
            path = os.path.join(D, "videos", f"Ex{ex}", f"{vid}-Camera17-30fps.mp4")
            cap = cv2.VideoCapture(path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, a)
            frames = []
            f = a
            while f <= b and len(frames) < max_frames:
                if not cap.grab():
                    break
                if (f - a) % step == 0:
                    ok, fr = cap.retrieve()
                    frames.append(cv2.resize(fr, (960, 540), interpolation=cv2.INTER_AREA))
                f += 1
            cap.release()
            kp = {}
            for c, (s, cx, cy) in CONDS.items():
                out = []
                with PoseExtractor("full") as pe:
                    for i, fr in enumerate(frames):
                        pf = pe.process(cv2.cvtColor(warp(fr, s, cx, cy), cv2.COLOR_BGR2RGB), i * 100)
                        out.append(pf.img[:, :2])
                kp[c] = to_common(np.stack(out), "mediapipe")
            ref = kp["original"]
            ok_ref = np.isfinite(ref[:, LIMB]).all((1, 2))
            for c in ("far_left", "far_right"):
                x = kp[c]
                ok = ok_ref & np.isfinite(x[:, LIMB]).all((1, 2))
                det = float(np.isfinite(x[:, 0, 0]).mean())
                for meth in ("none", "bbox", "torso"):
                    if meth == "none":
                        A, B = ref, x
                    elif meth == "bbox":
                        A, _ = bbox_normalize(ref, body=LIMB); B, _ = bbox_normalize(x, body=LIMB)
                    else:
                        A, _ = torso_normalize_2d(ref, I["l_shoulder"], I["r_shoulder"], I["l_hip"], I["r_hip"])
                        B, _ = torso_normalize_2d(x, I["l_shoulder"], I["r_shoulder"], I["l_hip"], I["r_hip"])
                    torso = np.linalg.norm(A[:, I["neck"]] - A[:, I["pelvis"]], axis=-1)
                    d = np.linalg.norm(A[:, LIMB] - B[:, LIMB], axis=-1) / torso[:, None]
                    rows.append(dict(ex=ex, video=vid, cond=c, method=meth, frames=int(ok.sum()), detected=det,
                                     diff_pct=float(np.nanmean(d[ok]) * 100),
                                     diff_pct_median=float(np.nanmedian(d[ok]) * 100)))
                aa = angle_matrix(ref)[ok]; bb = angle_matrix(x)[ok]
                rows.append(dict(ex=ex, video=vid, cond=c, method="angles_deg", frames=int(ok.sum()), detected=det,
                                 diff_pct=float(np.nanmean(np.abs(aa - bb))), diff_pct_median=float(np.nanmedian(np.abs(aa - bb)))))
            print(ex, vid, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "normalisation_invariance.csv"), index=False)
    s = df.groupby(["method", "cond"])[["diff_pct", "diff_pct_median", "detected"]].mean().round(2)
    s.to_csv(os.path.join(RES, "normalisation_invariance_by_cond.csv"))
    print(s)


if __name__ == "__main__":
    main()
