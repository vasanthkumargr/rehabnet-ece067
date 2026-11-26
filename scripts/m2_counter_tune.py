"""Milestone 2 - choose counter settings (smoothing, hysteresis, minimum rep length) without testing on the same people.

2-fold split by person (odd / even person id): settings are chosen on one half (lowest mean |count error| on
MediaPipe angles) and evaluated on the other half, then the halves are swapped. Output: results/m2/counter_tuning.csv
"""
import itertools
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rehabnet.core.exercises import EXERCISES, angle_index  # noqa: E402
from rehabnet.core.reps import count_reps, match_reps, thresholds_from_reps  # noqa: E402

d = pickle.load(open(os.path.join(ROOT, "data", "processed", "rehab24_reps.pkl"), "rb"))
REPS, STREAMS = d["reps"], d["streams"]


def ema(x, tau, fs=10.0):
    if tau <= 0:
        return x
    a = 1 - np.exp(-1 / (tau * fs))
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + a * (x[i] - y[i - 1]) if np.isfinite(x[i]) else y[i - 1]
    return y


def run(streams, tau, frac, min_s, src="mp"):
    rows = []
    for s in streams:
        ex = EXERCISES[s["exercise"]]
        idx = angle_index(ex.count_angle, s["side"])
        train = [ema(np.nanmean(r[src][:, idx], 1), tau) for r in REPS if r["exercise"] == s["exercise"]
                 and r["side"] == s["side"] and r["person"] != s["person"] and r["correct"] == 1 and r["cam"] == s["cam"]]
        th = thresholds_from_reps(train, frac=frac, min_samples=min_s, tau=0)
        a0 = max(0, s["gt_reps"][0][0] - 10)
        a1 = min(len(s[src]), s["gt_reps"][-1][1] + 11)
        sig = ema(np.nanmean(s[src][a0:a1][:, idx], 1), tau)
        m = match_reps(count_reps(sig, th), [(a - a0, b - a0) for a, b in s["gt_reps"]])
        rows.append(dict(video=s["video"], cam=s["cam"], view=s["view"], exercise=s["exercise"], person=s["person"],
                         **m, err=abs(m["pred"] - m["gt"])))
    return pd.DataFrame(rows)


def main():
    grid = list(itertools.product([0.3, 0.5, 0.8, 1.2], [0.25, 0.3, 0.4], [6, 8, 12]))
    out = []
    for fold in (0, 1):
        tr = [s for s in STREAMS if s["person"] % 2 == fold]
        te = [s for s in STREAMS if s["person"] % 2 != fold]
        scores = [(run(tr, *g).err.mean(), g) for g in grid]
        scores.sort()
        best = scores[0][1]
        res = run(te, *best)
        res["fold"], res["tau"], res["frac"], res["min_s"] = fold, *best
        out.append(res)
        print("fold", fold, "best", best, "train MAE", round(scores[0][0], 3), "test MAE", round(res.err.mean(), 3), flush=True)
    df = pd.concat(out)
    df.to_csv(os.path.join(ROOT, "results", "m2", "counter_tuning.csv"), index=False)
    # final settings chosen on all people (used in the app), reported separately
    scores = sorted((run(STREAMS, *g).err.mean(), g) for g in grid)
    print("chosen on all:", scores[0][1], "MAE", round(scores[0][0], 3))
    pd.DataFrame([dict(tau=scores[0][1][0], frac=scores[0][1][1], min_samples=scores[0][1][2])]).to_csv(
        os.path.join(ROOT, "results", "m2", "counter_settings.csv"), index=False)


if __name__ == "__main__":
    main()
