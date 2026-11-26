"""Validate the full live pipeline as the app runs it (leave-one-person-out).

For each REHAB24-6 camera stream: counter thresholds and reference repetitions come only from OTHER people; the
stream is cut into repetitions by the live counter (+ trimming), every counted repetition is scored and gets
feedback, and is matched to the annotated repetition (label) that contains its peak.
Outputs: results/m3/live_eval_reps.csv, live_eval_summary.csv
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
from rehabnet.core.feedback import feedback  # noqa: E402
from rehabnet.core.pipeline import label_segments, segment_stream  # noqa: E402
from rehabnet.core.reps import thresholds_from_reps  # noqa: E402
from rehabnet.core.scoring import ReferenceModel  # noqa: E402

warnings.filterwarnings("ignore")
RES = os.path.join(ROOT, "results", "m3")


def main():
    os.makedirs(RES, exist_ok=True)
    d = pickle.load(open(os.path.join(ROOT, "data", "processed", "rehab24_reps.pkl"), "rb"))
    reps, streams = d["reps"], d["streams"]
    for st in streams:
        lab = {r["rep"]: r["correct"] for r in reps if r["video"] == st["video"] and r["cam"] == st["cam"]}
        st["gt_labels"] = [lab[k] for k in sorted(lab)][: len(st["gt_reps"])]
    rows = []
    cache = {}
    for st in streams:
        exn, side, person = st["exercise"], st["side"], st["person"]
        key = (exn, side, person)
        if key not in cache:
            ex = EXERCISES[exn]
            idx = angle_index(ex.count_angle, side)
            th = thresholds_from_reps([np.nanmean(r["mp"][:, idx], 1) for r in reps if r["exercise"] == exn
                                       and r["side"] == side and r["person"] != person and r["correct"]])
            src = []
            for o in streams:
                if o["exercise"] == exn and o["side"] == side and o["person"] != person:
                    segs = segment_stream(o["mp"], exn, side, th)
                    labs = label_segments(segs, o["gt_reps"], o["gt_labels"])
                    src += [o["mp"][a:b + 1] for (a, b, _), lab in zip(segs, labs) if lab == 1]
            cache[key] = (th, ReferenceModel.build(exn, side, src, max_refs=80))
            print(exn, side, person, "refs", len(src), flush=True)
        th, model = cache[key]
        segs = segment_stream(st["mp"], exn, side, th)
        labs = label_segments(segs, st["gt_reps"], st["gt_labels"])
        for (a, b, r), lab in zip(segs, labs):
            out = model.score(st["mp"][a:b + 1])
            fb = feedback(out, exn, side)
            rows.append(dict(exercise=exn, person=person, video=st["video"], cam=st["cam"], view=st["view"],
                             label=lab, score=out["score"], good=fb[0]["code"] == "good",
                             codes="+".join(m["code"] for m in fb), duration=(b - a + 1) / 10))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RES, "live_eval_reps.csv"), index=False)
    lab = df[df.label.notna()]
    summ = []
    for exn, g in lab.groupby("exercise"):
        summ.append(dict(exercise=exn, counted=int((df.exercise == exn).sum()), matched=len(g),
                         auc=roc_auc_score(g.label, g.score), correct_told_good=g[g.label == 1].good.mean(),
                         incorrect_corrected=1 - g[g.label == 0].good.mean(),
                         mean_score_correct=g[g.label == 1].score.mean(), mean_score_incorrect=g[g.label == 0].score.mean()))
    s = pd.DataFrame(summ)
    s.loc[len(s)] = dict(exercise="ALL", counted=len(df), matched=len(lab), auc=s.auc.mean(),
                         correct_told_good=lab[lab.label == 1].good.mean(), incorrect_corrected=1 - lab[lab.label == 0].good.mean(),
                         mean_score_correct=lab[lab.label == 1].score.mean(), mean_score_incorrect=lab[lab.label == 0].score.mean())
    s.round(3).to_csv(os.path.join(RES, "live_eval_summary.csv"), index=False)
    pd.set_option("display.width", 200)
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
