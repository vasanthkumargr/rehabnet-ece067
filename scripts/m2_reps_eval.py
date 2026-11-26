"""Milestone 2 - validate the angle-threshold repetition counter.

REHAB24-6: every stream is counted with thresholds learned from OTHER people's correct repetitions (leave-one-
person-out), on the annotated span +- 1 s, and compared with the annotated repetitions (count error, precision,
recall). Done for MediaPipe angles (camera) and OptiTrack angles (upper bound).
UI-PRMD: every sequence is one repetition -> the counter should report exactly 1.
KIMORE: no repetition labels exist; the count distribution is reported only.
Outputs: results/m2/reps_*.csv
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rehabnet.core.angles import ANGLE_NAMES  # noqa: E402
from rehabnet.core.exercises import EXERCISES, angle_index  # noqa: E402
from rehabnet.core.reps import count_reps, match_reps, thresholds_from_reps  # noqa: E402

RES = os.path.join(ROOT, "results", "m2")


def signal(a, idx):
    return np.nanmean(a[:, idx], 1)


def rehab24(src, frac=0.25):
    d = pickle.load(open(os.path.join(ROOT, "data", "processed", "rehab24_reps.pkl"), "rb"))
    reps, streams = d["reps"], d["streams"]
    rows = []
    for s in streams:
        ex = EXERCISES[s["exercise"]]
        idx = angle_index(ex.count_angle, s["side"])
        train = [signal(r[src], idx) for r in reps if r["exercise"] == s["exercise"] and r["side"] == s["side"]
                 and r["person"] != s["person"] and r["correct"] == 1 and r["cam"] == s["cam"]]
        th = thresholds_from_reps(train, frac=frac)
        a0 = max(0, s["gt_reps"][0][0] - int(s["fps"]))
        a1 = min(len(s[src]), s["gt_reps"][-1][1] + int(s["fps"]) + 1)
        sig = signal(s[src][a0:a1], idx)
        pred = count_reps(sig, th)
        gt = [(a - a0, b - a0) for a, b in s["gt_reps"]]
        m = match_reps(pred, gt)
        rows.append(dict(src=src, video=s["video"], cam=s["cam"], view=s["view"], exercise=s["exercise"],
                         person=s["person"], **m, count_err=m["pred"] - m["gt"]))
    return pd.DataFrame(rows)


def rehabpile(name):
    d = pickle.load(open(os.path.join(ROOT, "data", "processed", f"{name}.pkl"), "rb"))["records"]
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ("joints", "angles")} for r in d])
    rows = []
    for ex, g in df.groupby("exercise"):
        recs = [r for r in d if r["exercise"] == ex]
        good = [r for r in recs if r["score"] >= np.median([q["score"] for q in recs])]
        # counting angle = the angle that moves most in the better half of recordings
        rom = np.nanmedian([np.nanpercentile(r["angles"], 95, 0) - np.nanpercentile(r["angles"], 5, 0) for r in good], 0)
        rom[[ANGLE_NAMES.index(a) for a in ("hip_abd_l", "hip_abd_r")]] = 0  # unstable definition when hips flex
        j = int(np.argmax(rom))
        for r in recs:
            train = [q["angles"][:, j] for q in good if q["subject"] != r["subject"]]
            if name == "uiprmd":
                th = thresholds_from_reps(train)
            else:  # KIMORE recordings hold several reps; learn rest/peak from the recording-level extremes
                th = thresholds_from_reps(train)
            n = len(count_reps(r["angles"][:, j], th))
            rows.append(dict(dataset=name, exercise=ex, count_angle=ANGLE_NAMES[j], subject=r["subject"], score=r["score"], count=n))
    return pd.DataFrame(rows)


def main():
    os.makedirs(RES, exist_ok=True)
    df = pd.concat([rehab24("mp"), rehab24("gt")])
    df.to_csv(os.path.join(RES, "reps_rehab24.csv"), index=False)
    agg = lambda g: pd.Series(dict(streams=len(g), gt_reps=g["gt"].sum(), pred_reps=g["pred"].sum(),  # noqa: E731
                                   precision=g["tp"].sum() / max(1, g["pred"].sum()), recall=g["tp"].sum() / max(1, g["gt"].sum()),
                                   exact_count=(g.count_err == 0).mean(), within1=(g.count_err.abs() <= 1).mean(),
                                   mae_count=g.count_err.abs().mean()))
    s1 = df.groupby(["src", "exercise"]).apply(agg).round(3)
    s2 = df.groupby(["src", "view"]).apply(agg).round(3)
    s0 = df.groupby("src").apply(agg).round(3)
    s1.to_csv(os.path.join(RES, "reps_rehab24_by_exercise.csv"))
    s2.to_csv(os.path.join(RES, "reps_rehab24_by_view.csv"))
    s0.to_csv(os.path.join(RES, "reps_rehab24_overall.csv"))
    print(s0, "\n", s1, "\n", s2)
    # sensitivity to the thresholds (MediaPipe)
    sens = []
    for fr in (0.2, 0.25, 0.3, 0.4, 0.5):
        g = rehab24("mp", fr)
        sens.append(dict(hysteresis=fr, **agg(g).to_dict()))
    pd.DataFrame(sens).round(3).to_csv(os.path.join(RES, "reps_threshold_sensitivity.csv"), index=False)
    print(pd.DataFrame(sens).round(3))
    up = rehabpile("uiprmd")
    up.to_csv(os.path.join(RES, "reps_uiprmd.csv"), index=False)
    u = up.groupby("exercise").agg(count_angle=("count_angle", "first"), sequences=("count", "size"),
                                   exactly_one=("count", lambda c: (c == 1).mean())).round(3)
    u.to_csv(os.path.join(RES, "reps_uiprmd_summary.csv"))
    print(u, "\nUI-PRMD exactly one:", (up["count"] == 1).mean())
    km = rehabpile("kimore")
    km.to_csv(os.path.join(RES, "reps_kimore.csv"), index=False)
    k = km.groupby("exercise")["count"].describe()
    k.to_csv(os.path.join(RES, "reps_kimore_summary.csv"))
    print(k)


if __name__ == "__main__":
    main()
