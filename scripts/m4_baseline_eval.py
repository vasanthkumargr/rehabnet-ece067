"""Milestone 4 - does a personal baseline help? (live pipeline, camera angles, leave-one-person-out)

For each person and exercise the first N_CAL counted repetitions that the physiotherapist labelled correct are the
'supervised calibration session'. The person's remaining repetitions are scored with
  generic  : correct repetitions of other people (the default)
  personal : only the person's calibration repetitions
  combined : generic + personal (each personal rep counted PERSONAL_WEIGHT times)  <- what the app does
and, for counting, the counter thresholds from other people vs from the person's calibration repetitions.
Outputs: results/m4/baseline_eval.csv, baseline_eval_summary.csv
"""
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rehabnet.core.exercises import EXERCISES, angle_index  # noqa: E402
from rehabnet.core.pipeline import label_segments, segment_stream  # noqa: E402
from rehabnet.core.reps import count_reps, match_reps, thresholds_from_reps  # noqa: E402
from rehabnet.core.scoring import ReferenceModel  # noqa: E402

warnings.filterwarnings("ignore")
RES = os.path.join(ROOT, "results", "m4")
N_CAL = int(os.environ.get("N_CAL", 5))
PERSONAL_WEIGHT = 3


def main():
    os.makedirs(RES, exist_ok=True)
    d = pickle.load(open(os.path.join(ROOT, "data", "processed", "rehab24_reps.pkl"), "rb"))
    reps, streams = d["reps"], d["streams"]
    for st in streams:
        lab = {r["rep"]: r["correct"] for r in reps if r["video"] == st["video"] and r["cam"] == st["cam"]}
        st["gt_labels"] = [lab[k] for k in sorted(lab)][: len(st["gt_reps"])]
    rows, count_rows = [], []
    for (exn, side, person) in sorted({(s["exercise"], s["side"], s["person"]) for s in streams}, key=str):
        ex = EXERCISES[exn]
        idx = angle_index(ex.count_angle, side)
        th = thresholds_from_reps([np.nanmean(r["mp"][:, idx], 1) for r in reps if r["exercise"] == exn
                                   and r["side"] == side and r["person"] != person and r["correct"]])
        gen = []
        for o in streams:
            if o["exercise"] == exn and o["side"] == side and o["person"] != person:
                segs = segment_stream(o["mp"], exn, side, th)
                gen += [o["mp"][a:b + 1] for (a, b, _), l in zip(segs, label_segments(segs, o["gt_reps"], o["gt_labels"])) if l == 1]
        mine = [s for s in streams if s["exercise"] == exn and s["side"] == side and s["person"] == person]
        items = []   # (stream, a, b, label, peak)
        for st in mine:
            segs = segment_stream(st["mp"], exn, side, th)
            for (a, b, r), l in zip(segs, label_segments(segs, st["gt_reps"], st["gt_labels"])):
                items.append((st, a, b, l))
        # calibration = first N_CAL correct reps of the first stream (camera c17), both cameras of those reps excluded
        first = sorted({(st["video"], st["cam"]) for st, *_ in items})[0]
        cal_items = [it for it in items if (it[0]["video"], it[0]["cam"]) == first and it[3] == 1][:N_CAL]
        if len(cal_items) < N_CAL:
            continue
        cal_end = {first[0]: max(it[2] for it in cal_items)}
        test = [it for it in items if it[3] is not None and not (it[0]["video"] in cal_end and it[1] <= cal_end[it[0]["video"]])]
        personal = [it[0]["mp"][it[1]:it[2] + 1] for it in cal_items]
        models = {"generic": ReferenceModel.build(exn, side, gen, max_refs=80),
                  "personal": ReferenceModel.build(exn, side, personal, max_refs=80),
                  "combined": ReferenceModel.build(exn, side, gen + personal * PERSONAL_WEIGHT, max_refs=10_000)}
        for st, a, b, l in test:
            row = dict(exercise=exn, side=side, person=person, video=st["video"], cam=st["cam"], label=l)
            for k, m in models.items():
                row[k] = m.score(st["mp"][a:b + 1])["score"]
            rows.append(row)
        # counting with personal thresholds on the rest of each stream
        pth = thresholds_from_reps([np.nanmean(p[:, idx], 1) for p in personal])
        for st in mine:
            start = cal_end.get(st["video"], -1) + 1
            sig = np.nanmean(st["mp"][start:, idx], 1)
            gt = [(ga - start, gb - start) for ga, gb in st["gt_reps"] if ga >= start]
            for name, t in (("generic", th), ("personal", pth)):
                m = match_reps(count_reps(sig, t), gt)
                count_rows.append(dict(exercise=exn, person=person, video=st["video"], cam=st["cam"], thresholds=name,
                                       **m, err=abs(m["pred"] - m["gt"])))
        print(exn, side, person, "test reps", len(test), flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "baseline_eval.csv"), index=False)
    out = []
    for exn, g in df.groupby("exercise"):
        out.append(dict(exercise=exn, reps=len(g), **{k: roc_auc_score(g.label, g[k]) for k in ("generic", "personal", "combined")}))
    s = pd.DataFrame(out)
    s.loc[len(s)] = dict(exercise="MEAN", reps=len(df), **{k: s[k].mean() for k in ("generic", "personal", "combined")})
    s.round(3).to_csv(os.path.join(RES, "baseline_eval_summary.csv"), index=False)
    c = pd.DataFrame(count_rows)
    c.to_csv(os.path.join(RES, "baseline_count_eval.csv"), index=False)
    cs = c.groupby("thresholds").apply(lambda g: pd.Series(dict(recall=g["tp"].sum() / g["gt"].sum(),
                                                                 precision=g["tp"].sum() / max(1, g["pred"].sum()),
                                                                 mae=g.err.mean()))).round(3)
    cs.to_csv(os.path.join(RES, "baseline_count_summary.csv"))
    pd.set_option("display.width", 200)
    print(s.round(3).to_string()); print(cs)


if __name__ == "__main__":
    main()
