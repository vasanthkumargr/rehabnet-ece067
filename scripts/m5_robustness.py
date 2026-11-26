"""Milestone 5 - robustness and speed of the full camera pipeline (run on Kaggle; see scripts/kaggle_m5_job.sh).

12 REHAB24-6 front-camera recordings (2 per exercise) are processed under image degradations and with the three
BlazePose model sizes. For each condition: person detection rate, key-angle error vs OptiTrack, repetition-count
error (live counter, thresholds from OTHER people), and pose time per frame (CPU).
Conditions: original | dark (evening light simulation) | low_res (320 px + blur) | occluded (a moving block hides a
quarter of the body) | small (person at ~45 % size, far away) ; models: lite | full | heavy (original frames).
Output: results/m5/robustness.csv
"""
import os
import pickle
import sys
import time

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rehabnet.core.angles import ANGLE_NAMES, angle_matrix, to_common  # noqa: E402
from rehabnet.core.exercises import EXERCISES, REHAB24, angle_index, resolve, side_from_subtype  # noqa: E402
from rehabnet.core.normalize import one_euro_sequence  # noqa: E402
from rehabnet.core.pipeline import segment_stream  # noqa: E402
from rehabnet.core.pose import PoseExtractor  # noqa: E402
from rehabnet.core.reps import thresholds_from_reps  # noqa: E402
from rehabnet.core.scoring import fill_nan  # noqa: E402

D = os.path.join(ROOT, "data", "rehab24_6")
RES = os.path.join(ROOT, "results", "m5")
STEP = 3  # 10 fps


def degrade(frame, cond, k, rng):
    h, w = frame.shape[:2]
    if cond == "dark":
        f = (255 * (frame / 255.0) ** 2.2 * 0.6).astype(np.float32)
        f += rng.normal(0, 6, f.shape)
        return np.clip(f, 0, 255).astype(np.uint8)
    if cond == "low_res":
        s = cv2.resize(frame, (320, int(320 * h / w)), interpolation=cv2.INTER_AREA)
        s = cv2.GaussianBlur(s, (3, 3), 0)
        return cv2.resize(s, (w, h), interpolation=cv2.INTER_LINEAR)
    if cond == "occluded":
        f = frame.copy()
        bw, bh = w // 5, h // 3
        x = int((w - bw) * (0.5 + 0.5 * np.sin(k / 40)))
        y = int((h - bh) * (0.5 + 0.5 * np.cos(k / 55)))
        f[y:y + bh, x:x + bw] = 40
        return f
    if cond == "small":
        s = cv2.resize(frame, None, fx=0.45, fy=0.45, interpolation=cv2.INTER_AREA)
        c = np.full_like(frame, int(frame.mean()))
        y0, x0 = (h - s.shape[0]) // 2, (w - s.shape[1]) // 2
        c[y0:y0 + s.shape[0], x0:x0 + s.shape[1]] = s
        return c
    return frame


def run(path, cond, variant, gt3, frames_idx):
    cap = cv2.VideoCapture(path)
    rng = np.random.default_rng(0)
    world, ms, k = [], [], 0
    with PoseExtractor(variant) as pe:
        f = 0
        while True:
            if not cap.grab():
                break
            if f % STEP == 0:
                ok, fr = cap.retrieve()
                fr = cv2.resize(fr, (960, 540), interpolation=cv2.INTER_AREA)
                fr = degrade(fr, cond, k, rng)
                t = time.perf_counter()
                p = pe.process(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB), f * 1000 / 30)
                ms.append((time.perf_counter() - t) * 1000)
                world.append(p.world)
                k += 1
            f += 1
    cap.release()
    return np.array(world), np.array(ms)


def main():
    os.makedirs(RES, exist_ok=True)
    seg = pd.read_csv(os.path.join(D, "Segmentation.csv"), sep=";")
    import json
    from rehabnet.core.reps import RepThresholds
    TH = json.load(open(os.path.join(ROOT, "data", "processed", "m5_thresholds.json")))  # leave-one-person-out
    picks = []
    for ex_id in range(1, 7):
        vids = seg[(seg.exercise_id == ex_id) & (seg.cam17_orientation == "front")].video_id.unique()[:2]
        picks += [(ex_id, v) for v in vids]
    conds = [(c, "full") for c in ("original", "dark", "low_res", "occluded", "small")] + [("original", "lite"), ("original", "heavy")]
    rows = []
    for ex_id, vid in picks:
        g = seg[seg.video_id == vid].sort_values("repetition_number")
        exn, side = REHAB24[ex_id], side_from_subtype(g.exercise_subtype.iloc[0])
        ex = EXERCISES[exn]
        gt3 = np.load(os.path.join(D, "joints3d", f"Ex{ex_id}", f"{vid}-30fps.npy"))[..., :3]
        F = np.arange(0, len(gt3), STEP)
        gt_ang = angle_matrix(to_common(gt3, "optitrack"), up=np.array([0, 1.0, 0]))[F]
        key = [ANGLE_NAMES.index(resolve(a, side)) for a in ex.key_angles if not resolve(a, side).startswith("hip_abd") or exn == "leg_abduction"]
        idx = angle_index(ex.count_angle, side)
        person = int(g.person_id.iloc[0])
        rest, peak = TH[f"{exn}|{side}|{person}"]
        th = RepThresholds(rest=rest, peak=peak)
        inrep = np.zeros(len(gt3), bool)
        for _, r in g.iterrows():
            inrep[r.first_frame:r.last_frame + 1] = True
        mask = inrep[F]
        a0 = max(0, g.first_frame.min() // STEP - 10)
        a1 = g.last_frame.max() // STEP + 11
        path = os.path.join(D, "videos", f"Ex{ex_id}", f"{vid}-Camera17-30fps.mp4")
        for cond, variant in conds:
            w, ms = run(path, cond, variant, gt3, F)
            n = min(len(w), len(F))
            w, m = w[:n], mask[:n]
            det = np.isfinite(w[:, 0, 0])
            wf = fill_nan(to_common(w, "mediapipe")) if det.any() else None
            if wf is None or not np.isfinite(wf).all():
                rows.append(dict(exercise=exn, video=vid, cond=cond, model=variant, detected=float(det[m].mean()),
                                 ms=float(np.median(ms)), angle_mae=np.nan, count_err=np.nan, gt_reps=len(g)))
                continue
            A = angle_matrix(one_euro_sequence(wf, 10.0, 1.5, 0.01), up=np.array([0, -1.0, 0]))
            mae = float(np.nanmean(np.abs(A[:n][m][:, key] - gt_ang[:n][m][:, key])))
            counted = len(segment_stream(A[a0:a1], exn, side, th))
            rows.append(dict(exercise=exn, video=vid, cond=cond, model=variant, detected=float(det[m].mean()),
                             ms=float(np.median(ms)), angle_mae=mae, counted=counted, gt_reps=len(g),
                             count_err=abs(counted - len(g))))
            print(vid, exn, cond, variant, rows[-1], flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(RES, "robustness.csv"), index=False)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "robustness.csv"), index=False)
    print(df.groupby(["cond", "model"])[["detected", "angle_mae", "count_err", "ms"]].mean().round(3))


if __name__ == "__main__":
    main()
