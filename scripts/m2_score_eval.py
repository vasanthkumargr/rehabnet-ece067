"""Milestone 2 - validate DTW / NCC / multi-factor scoring.

REHAB24-6 (camera = MediaPipe angles, and mocap = OptiTrack angles for comparison):
  every repetition is scored against correct repetitions of the same exercise (and side) from OTHER people
  (leave-one-person-out). AUC = how well a score separates the physiotherapist's correct vs incorrect labels
  (0.5 = chance, 1.0 = perfect). Reported for each factor alone, the fixed-weight RehabNet score, and a logistic
  regression on the factors (also leave-one-person-out) as an upper reference.
UI-PRMD and KIMORE (skeleton datasets): each recording is scored against the better-scored half of other
  people's recordings; Spearman correlation with the dataset's quality score / clinician score.
Outputs: results/m2/score_*.csv
"""
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rehabnet.core.angles import ANGLE_NAMES  # noqa: E402
from rehabnet.core.exercises import Exercise  # noqa: E402
from rehabnet.core.scoring import HANDSET_WEIGHTS, WEIGHTS, ReferenceModel, resample_time  # noqa: E402

warnings.filterwarnings("ignore")
RES = os.path.join(ROOT, "results", "m2")
FACTORS = list(WEIGHTS)
SIGN = {"posture": -1, "similarity": 1, "rom": -1, "stability": -1, "tempo": -1, "symmetry": -1}


def auc(y, s):
    ok = np.isfinite(s)
    y, s = np.asarray(y)[ok], np.asarray(s)[ok]
    return roc_auc_score(y, s) if len(set(y)) == 2 else np.nan


N_CAL = 5  # personal mode: the patient's first N correct repetitions are their reference ("calibration")


def score_rehab24(src, mode="generic"):
    """mode generic: references = correct reps of OTHER people. mode personal: references = the same person's
    first N_CAL correct reps of that exercise/side (both cameras). In both modes those calibration reps are left
    out of the evaluation, so the two modes are tested on exactly the same repetitions."""
    d = pickle.load(open(os.path.join(ROOT, "data", "processed", "rehab24_reps.pkl"), "rb"))
    reps = [r for r in d["reps"] if not (src == "gt" and r["mocap_err"])]
    rows = []
    groups = {}
    for r in reps:
        groups.setdefault((r["exercise"], r["side"], r["person"]), []).append(r)
    for (exn, side, person), mine in groups.items():
        corr = sorted({(r["video"], r["rep"]) for r in mine if r["correct"]}, key=lambda k: (k[0], k[1]))
        cal = set(corr[:N_CAL])
        test = [r for r in mine if (r["video"], r["rep"]) not in cal]
        if mode == "personal":
            refs = {None: [q[src] for q in mine if (q["video"], q["rep"]) in cal]}
        elif mode == "generic":
            refs = {None: [q[src] for q in reps if q["exercise"] == exn and q["side"] == side and q["person"] != person and q["correct"]]}
        else:  # generic_view: other people, same camera view only
            refs = {v: [q[src] for q in reps if q["exercise"] == exn and q["side"] == side and q["person"] != person
                        and q["correct"] and q["view"] == v] for v in {r["view"] for r in test}}
        if min(len(x) for x in refs.values()) < 3 or not test:
            continue
        models = {v: ReferenceModel.build(exn, side, x) for v, x in refs.items()}
        for r in test:
            model = models[None] if None in models else models[r["view"]]
            out = model.score(r[src])
            row = dict(src=src, mode=mode, exercise=exn, side=side, person=person, video=r["video"], cam=r["cam"],
                       view=r["view"], rep=r["rep"], correct=r["correct"], lights=r["lights"], score=out["score"])
            row.update({f"f_{k}": out["factors"].get(k) for k in FACTORS})
            row.update({f"s_{k}": out["sub"].get(k) for k in FACTORS})
            rows.append(row)
        print(src, mode, exn, side, person, len(test), flush=True)
    return pd.DataFrame(rows)


def global_weights(df):
    """Leave-one-person-out logistic regression on the 0-100 sub-scores, pooled over all exercises
    (one weight set for the whole app). Returns per-rep predictions and the weights fitted on everyone."""
    cols = [f"s_{k}" for k in FACTORS]
    X = df[cols].fillna(100.0).values / 100
    y = df.correct.values
    pred = np.full(len(df), np.nan)
    for p in df.person.unique():
        tr, te = df.person.values != p, df.person.values == p
        m = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    m = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
    w = np.clip(m.coef_[0], 0, None)
    return pred, dict(zip(FACTORS, (w / w.sum()).round(3) if w.sum() > 0 else w))


def learned_auc(df):
    """Leave-one-person-out logistic regression on the factor values, per exercise."""
    out = {}
    for exn, g in df.groupby("exercise"):
        cols = [f"f_{k}" for k in FACTORS if g[f"f_{k}"].notna().mean() > 0.9]
        X = g[cols].fillna(g[cols].median()).values
        y = g.correct.values
        pred = np.full(len(g), np.nan)
        for p in g.person.unique():
            tr, te = g.person.values != p, g.person.values == p
            if len(set(y[tr])) < 2:
                continue
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            m = LogisticRegression(C=1.0, max_iter=1000).fit((X[tr] - mu) / sd, y[tr])
            pred[te] = m.predict_proba((X[te] - mu) / sd)[:, 1]
        out[exn] = auc(y, pred)
    return out


def summarise(df):
    rows = []
    for (src, mode, exn), g in df.groupby(["src", "mode", "exercise"]):
        row = dict(src=src, mode=mode, exercise=exn, reps=len(g), correct_share=g.correct.mean(),
                   rehabnet_score=auc(g.correct, g.score), global_weights=auc(g.correct, g.p_global))
        for k in FACTORS:
            row[k] = auc(g.correct, SIGN[k] * g[f"f_{k}"]) if g[f"f_{k}"].notna().any() else np.nan
        rows.append(row)
    s = pd.DataFrame(rows)
    for (src, mode) in s[["src", "mode"]].drop_duplicates().itertuples(index=False):
        la = learned_auc(df[(df.src == src) & (df["mode"] == mode)])
        sel = (s.src == src) & (s["mode"] == mode)
        s.loc[sel, "per_exercise_LR"] = s.loc[sel, "exercise"].map(la)
    return s


# ---------------------------------------------------------------- skeleton datasets
def score_rehabpile(name, downsample=1):
    recs = pickle.load(open(os.path.join(ROOT, "data", "processed", f"{name}.pkl"), "rb"))["records"]
    rows = []
    for exn in sorted(set(r["exercise"] for r in recs)):
        R = [r for r in recs if r["exercise"] == exn]
        med = np.median([r["score"] for r in R])
        good = [r for r in R if r["score"] >= med]
        A = lambda r: resample_time(r["angles"][::downsample].astype(float), 1, 1)  # noqa: E731
        rom = np.nanmedian([np.nanpercentile(A(r), 95, 0) - np.nanpercentile(A(r), 5, 0) for r in good], 0)
        rom[[ANGLE_NAMES.index(a) for a in ("hip_abd_l", "hip_abd_r")]] = 0
        key = [ANGLE_NAMES[j] for j in np.argsort(-rom)[:4] if rom[j] > 15] or [ANGLE_NAMES[int(np.argmax(rom))]]
        ex = Exercise(f"{name}_{exn}", exn, key_angles=key, count_angle=key[:1],
                      stability_angles=[a for a in ("trunk_lean",) if a not in key])
        for subj in sorted(set(r["subject"] for r in R)):
            refs = [A(r) for r in good if r["subject"] != subj]
            model = ReferenceModel.build(ex, None, refs, max_refs=40)
            for r in R:
                if r["subject"] != subj:
                    continue
                out = model.score(A(r))
                hand = model.score(A(r), weights={k: v for k, v in HANDSET_WEIGHTS.items() if k != "tempo"})
                row = dict(dataset=name, exercise=exn, key=",".join(key), subject=subj, target=r["score"], score=out["score"],
                           score_handset=hand["score"])
                row.update({f"f_{k}": out["factors"].get(k) for k in FACTORS})
                rows.append(row)
        print(name, exn, flush=True)
    df = pd.DataFrame(rows)
    summ = []
    for exn, g in df.groupby("exercise"):
        row = dict(dataset=name, exercise=exn, key_angles=g.key.iloc[0], recordings=len(g),
                   rehabnet_score=spearmanr(g.score, g.target)[0], handset_weights=spearmanr(g.score_handset, g.target)[0])
        for k in ("posture", "similarity", "rom", "stability", "symmetry"):
            v = SIGN[k] * g[f"f_{k}"]
            row[k] = spearmanr(v, g.target, nan_policy="omit")[0] if v.notna().sum() > 5 else np.nan
        summ.append(row)
    return df, pd.DataFrame(summ)


def main():
    os.makedirs(RES, exist_ok=True)
    which = sys.argv[1:] or ["rehab24", "uiprmd", "kimore"]
    if "rehab24" in which:
        parts, weights = [], []
        for src in ("mp", "gt"):
            for mode in (("generic", "generic_view", "personal") if src == "mp" else ("generic", "personal")):
                g = score_rehab24(src, mode)
                g["p_global"], w = global_weights(g)
                weights.append(dict(src=src, mode=mode, **w))
                parts.append(g)
        df = pd.concat(parts)
        df.to_csv(os.path.join(RES, "score_rehab24_reps.csv"), index=False)
        pd.DataFrame(weights).to_csv(os.path.join(RES, "score_global_weights.csv"), index=False)
        s = summarise(df)
        s.round(3).to_csv(os.path.join(RES, "score_rehab24_auc.csv"), index=False)
        pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
        print(s.round(3).to_string())
        ov = []
        for (src, mode), g in df.groupby(["src", "mode"]):
            ss = s[(s.src == src) & (s["mode"] == mode)]
            row = dict(src=src, mode=mode, reps=len(g), score_auc=ss.rehabnet_score.mean(),
                       global_weights_auc=ss.global_weights.mean(), posture_only=ss.posture.mean(),
                       per_exercise_LR=ss.per_exercise_LR.mean())
            for v, gv in g.groupby("view"):
                row[f"score_{v}"] = np.nanmean([auc(x.correct, x.score) for _, x in gv.groupby("exercise")])
            ov.append(row)
        pd.DataFrame(ov).round(3).to_csv(os.path.join(RES, "score_rehab24_overall.csv"), index=False)
        print(pd.DataFrame(ov).round(3).to_string())
        print(pd.DataFrame(weights).to_string())
    for name, ds in (("uiprmd", 1), ("kimore", 3)):
        if name in which:
            df, s = score_rehabpile(name, ds)
            df.to_csv(os.path.join(RES, f"score_{name}_recs.csv"), index=False)
            s.round(3).to_csv(os.path.join(RES, f"score_{name}_spearman.csv"), index=False)
            print(s.round(3))


if __name__ == "__main__":
    main()
