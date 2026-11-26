"""Milestone 1 - preprocess KIMORE and UI-PRMD skeletons (Rehab-Pile release) into RehabNet's common format.

For every recording: map to the 14 common joints, put the body in a body-centred frame (pelvis origin, scaled
by torso length), compute joint angles, and store with exercise / subject / score metadata.

Output: data/processed/{kimore,uiprmd}.pkl  (list of dicts)  +  results/m1/rehabpile_summary.csv
Run:   .venv\\Scripts\\python.exe -P scripts/m1_preprocess_rehabpile.py
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rehabnet.core.angles import ANGLE_NAMES, COMMON, angle_matrix, to_common  # noqa: E402

SRC = os.environ.get("REHABPILE_DIR", r"C:\Users\vasan\Downloads\KIMORE_final_package\KIMORE_RESEARCH\external_data\rehabpile_all")
OUT = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "results", "m1")
EX_NAMES = {
    "KIMORE": {"LA": "Lifting arms (frontal raise)", "LT": "Lateral trunk tilt", "TR": "Trunk rotation",
               "PR": "Pelvis rotation", "Sq": "Squat"},
    "UIPRMD": {"DS": "Deep squat", "HS": "Hurdle step", "IL": "Inline lunge", "SL": "Side lunge",
               "STS": "Sit to stand", "SASLR": "Standing active straight leg raise",
               "SSA": "Standing shoulder abduction", "SSE": "Standing shoulder extension",
               "SSIER": "Standing shoulder internal-external rotation", "SSS": "Standing shoulder scaption"},
}


def body_centre(c):
    """c (T,14,3): pelvis at origin, divide by mean torso length."""
    pel = c[:, COMMON.index("pelvis")]
    neck = c[:, COMMON.index("neck")]
    torso = np.nanmean(np.linalg.norm(neck - pel, axis=-1))
    return (c - pel[:, None]) / (torso + 1e-9), torso


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(RES, exist_ok=True)
    rows = []
    for origin, key, src_map in [("KIMORE_reg", "kimore", "kimore"), ("UIPRMD_reg", "uiprmd", "uiprmd")]:
        ds = origin.split("_")[0]
        recs = []
        for ex in sorted(os.listdir(os.path.join(SRC, origin))):
            d = os.path.join(SRC, origin, ex)
            if not os.path.isdir(d):
                continue
            info = json.load(open(os.path.join(d, "info.json")))
            J, D = info["n_joints"], info["n_dim"]
            for part in ("train", "test"):  # fold-0 train + test = every recording exactly once
                x = np.load(f"{d}/fold0/x_{part}_fold0.npy")
                y = np.load(f"{d}/fold0/y_{part}_fold0.npy", allow_pickle=True).reshape(len(x), -1)[:, 0]
                s = np.load(f"{d}/fold0/s_{part}_fold0.npy", allow_pickle=True)
                P = x.reshape(len(x), J, D, -1).transpose(0, 3, 1, 2)[..., :3]  # N,T,J,3
                for i in range(len(x)):
                    raw = P[i]
                    bad = ~np.isfinite(raw).all((1, 2))
                    c = to_common(raw, src_map)
                    cb, torso = body_centre(c)
                    ang = angle_matrix(c, up=np.array([0.0, 1.0, 0.0]))  # both sources are y-up
                    recs.append(dict(dataset=ds, exercise=ex, exercise_name=EX_NAMES[ds].get(ex, ex),
                                     subject=f"{ds}:{s[i]}", score=float(y[i]), joints=cb.astype(np.float32),
                                     angles=ang.astype(np.float32), n_frames=len(raw), n_bad=int(bad.sum()),
                                     torso_len_raw=float(torso)))
        pickle.dump(dict(common_joints=COMMON, angle_names=ANGLE_NAMES, records=recs),
                    open(os.path.join(OUT, f"{key}.pkl"), "wb"))
        df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("joints", "angles")} for r in recs])
        for ex, g in df.groupby("exercise"):
            A = np.concatenate([r["angles"] for r in recs if r["exercise"] == ex])
            rom = {f"rom_{n}": float(np.nanpercentile(A[:, j], 95) - np.nanpercentile(A[:, j], 5))
                   for j, n in enumerate(ANGLE_NAMES)}
            rows.append(dict(dataset=ds, exercise=ex, name=EX_NAMES[ds].get(ex, ex), recordings=len(g),
                             subjects=g.subject.nunique(), frames=int(g.n_frames.iloc[0]),
                             score_min=g.score.min(), score_max=g.score.max(), score_mean=g.score.mean(),
                             bad_frames=int(g.n_bad.sum()), **rom))
        print(f"{ds}: {len(recs)} recordings, {df.subject.nunique()} subjects, {df.exercise.nunique()} exercises")
    pd.DataFrame(rows).round(3).to_csv(os.path.join(RES, "rehabpile_summary.csv"), index=False)


if __name__ == "__main__":
    main()
