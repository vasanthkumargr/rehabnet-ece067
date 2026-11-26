"""Milestone 2 figures and worked examples (results/m2/fig_*.png, examples_*.csv)."""
import os
import pickle
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402
from dtaidistance import dtw  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rehabnet.core.angles import ANGLE_NAMES  # noqa: E402
from rehabnet.core.exercises import EXERCISES, angle_index  # noqa: E402
from rehabnet.core.feedback import feedback, session_summary  # noqa: E402
from rehabnet.core.reps import count_reps, smooth, thresholds_from_reps  # noqa: E402
from rehabnet.core.scoring import WEIGHTS, ReferenceModel  # noqa: E402

R = os.path.join(ROOT, "results", "m2")
NAVY, TEAL, ORANGE, GREY, RED, LIGHT = "#122B4A", "#0F766E", "#D97706", "#6B7280", "#B91C1C", "#E5E7EB"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})
NICE = {k: v.name for k, v in EXERCISES.items()}
NICE["pushup_table"] = "Table push-up"
D = pickle.load(open(os.path.join(ROOT, "data", "processed", "rehab24_reps.pkl"), "rb"))
REPS, STREAMS = D["reps"], D["streams"]


def save(fig, name):
    fig.savefig(os.path.join(R, name), dpi=170, bbox_inches="tight")
    plt.close(fig)


def pipeline():
    fig, ax = plt.subplots(figsize=(13, 3.3))
    ax.axis("off")
    steps = [("Joint angles\n(Milestone 1)", "3-D MediaPipe,\n10 samples/s"),
             ("Repetition\ncounter", "cycle detection with\nhysteresis + smoothing"),
             ("DTW\nalignment", "vs correct reference\nrepetitions"),
             ("NCC\nsimilarity", "shape correlation\nafter alignment"),
             ("Factors", "posture, ROM, stability,\ntempo, symmetry"),
             ("Score +\nfeedback", "0-100 per rep,\nsession summary")]
    w, gap = 1.85, 0.35
    for i, (t, sub) in enumerate(steps):
        x = i * (w + gap)
        col = TEAL if i in (1, 2, 3, 5) else NAVY
        ax.add_patch(FancyBboxPatch((x, 0.35), w, 1.1, boxstyle="round,pad=0.04,rounding_size=0.12", fc=col, ec="none"))
        ax.text(x + w / 2, 0.9, t, ha="center", va="center", color="white", fontsize=12, fontweight="bold")
        ax.text(x + w / 2, 0.02, sub, ha="center", va="center", color=GREY, fontsize=9.5)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.03, 0.9), xytext=(x + w + 0.03, 0.9),
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.8))
    ax.set_xlim(-0.1, len(steps) * (w + gap)); ax.set_ylim(-0.4, 1.6)
    save(fig, "fig_m2_pipeline.png")


def counter_example():
    cases = [("squat", "c17"), ("lunge", "c17")]
    fig, axs = plt.subplots(2, 1, figsize=(13, 6.4))
    for ax, (exn, cam) in zip(axs, cases):
        s = [x for x in STREAMS if x["exercise"] == exn and x["cam"] == cam and x["view"] == "front"][0]
        ex = EXERCISES[exn]
        idx = angle_index(ex.count_angle, s["side"])
        train = [np.nanmean(r["mp"][:, idx], 1) for r in REPS if r["exercise"] == exn and r["side"] == s["side"]
                 and r["person"] != s["person"] and r["correct"] and r["cam"] == cam]
        th = thresholds_from_reps(train)
        sig = np.nanmean(s["mp"][:, idx], 1)
        t = np.arange(len(sig)) / s["fps"]
        for k, (a, b) in enumerate(s["gt_reps"]):
            ax.axvspan(a / s["fps"], b / s["fps"], color=(TEAL if k % 2 else "#99d5cf"), alpha=0.18, lw=0)
        ax.plot(t, sig, color=GREY, lw=1, label="knee angle (camera)")
        ax.plot(t, smooth(sig, th.tau, th.rate), color=NAVY, lw=2, label="smoothed (what the counter sees)")
        reps = count_reps(sig, th)
        for r in reps:
            ax.plot(r.peak / s["fps"], smooth(sig, th.tau, th.rate)[r.peak], "v", color=ORANGE, ms=9)
        ax.set_title(f"{NICE[exn]} ({s['video']}, front camera): {len(s['gt_reps'])} annotated reps (shaded), "
                     f"{len(reps)} counted (orange)")
        ax.set_ylabel("degrees")
    axs[1].set_xlabel("time (s)"); axs[0].legend(frameon=False, loc="lower left", fontsize=9)
    save(fig, "fig_counter_example.png")


def counter_results():
    d = pd.read_csv(os.path.join(R, "counter_tuning.csv"))
    f = lambda g: pd.Series(dict(precision=g["tp"].sum() / g["pred"].sum(), recall=g["tp"].sum() / g["gt"].sum(),  # noqa: E731
                                 mae=g.err.mean()))
    e = d.groupby("exercise").apply(f)
    v = d.groupby("view").apply(f).loc[["front", "half-profile", "profile"]]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4), gridspec_kw=dict(width_ratios=[2, 1]))
    x = np.arange(len(e))
    axs[0].bar(x - 0.2, e.precision * 100, 0.4, color=NAVY, label="precision (counted reps that are real)")
    axs[0].bar(x + 0.2, e.recall * 100, 0.4, color=TEAL, label="recall (real reps that were counted)")
    axs[0].set_xticks(x, [NICE[i] for i in e.index], rotation=15); axs[0].set_ylim(50, 102)
    axs[0].set_ylabel("%"); axs[0].legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2); axs[0].set_title("By exercise (camera, unseen people)")
    axs[1].bar(range(3), v.mae, color=[TEAL, NAVY, ORANGE])
    for i, val in enumerate(v.mae):
        axs[1].text(i, val + 0.05, f"{val:.1f}", ha="center")
    axs[1].set_xticks(range(3), v.index); axs[1].set_ylabel("count error per recording (reps)")
    axs[1].set_title("By camera view")
    save(fig, "fig_counter_results.png")


def dtw_example():
    exn = "squat"
    # take the correct and the incorrect front-camera squat whose (leave-one-person-out) score is closest to the
    # median of its group, so the example is representative of the whole dataset
    sc = pd.read_csv(os.path.join(R, "score_rehab24_reps.csv"))
    g0 = sc[(sc.src == "mp") & (sc["mode"] == "generic") & (sc.exercise == exn) & (sc.view == "front")]
    pick = {}
    for lab in (1, 0):
        gg = g0[g0.correct == lab]
        row = gg.iloc[(gg.score - gg.score.median()).abs().argmin()]
        pick[lab] = [r for r in REPS if r["video"] == row.video and r["cam"] == row.cam and r["rep"] == row.rep][0]
    rep_c, rep_w = pick[1], pick[0]
    models = {}
    for lab, rep in pick.items():
        refs = [r["mp"] for r in REPS if r["exercise"] == exn and r["person"] != rep["person"] and r["correct"]]
        models[lab] = ReferenceModel.build(exn, None, refs)
    m = models[1]
    k = ANGLE_NAMES.index("l_knee")
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.9))
    for ax, rep, lab in ((axs[0], rep_c, "correct"), (axs[1], rep_w, "incorrect")):
        m = models[1 if lab == "correct" else 0]
        near = m.nearest(rep["mp"])[0][1]
        ref = m.reps[near]
        a, b = rep["mp"][:, k], ref[:, k]
        path = dtw.warping_path(a, b)
        ax.plot(np.arange(len(b)), b + 60, color=NAVY, lw=2, label="nearest correct reference (+60 deg)")
        ax.plot(np.arange(len(a)), a, color=(TEAL if lab == "correct" else ORANGE), lw=2, label=f"patient rep ({lab})")
        for i, j in path[::2]:
            ax.plot([i, j], [a[i], b[j] + 60], color=GREY, lw=0.4, alpha=0.6)
        out = m.score(rep["mp"])
        ax.set_title(f"{lab.capitalize()} squat - score {out['score']:.0f}/100")
        ax.set_xlabel("sample (10 per second)"); ax.set_ylabel("left knee angle (deg)")
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    sc = pd.read_csv(os.path.join(R, "score_rehab24_reps.csv"))
    g = sc[(sc.src == "mp") & (sc["mode"] == "generic")].copy()
    cols = [f"s_{k}" for k in WEIGHTS]
    w = np.array([WEIGHTS[k] for k in WEIGHTS])
    S = g[cols].fillna(100).values
    g["learned"] = (S * w).sum(1) / w.sum()
    axs[2].hist(g[g.correct == 1].learned, bins=30, alpha=0.6, color=TEAL, label="correct reps", density=True)
    axs[2].hist(g[g.correct == 0].learned, bins=30, alpha=0.6, color=ORANGE, label="incorrect reps", density=True)
    axs[2].set_title("RehabNet score, all exercises (camera)"); axs[2].set_xlabel("score (0-100)")
    axs[2].legend(frameon=False, fontsize=9)
    save(fig, "fig_dtw_example.png")


def score_auc():
    s = pd.read_csv(os.path.join(R, "score_rehab24_auc.csv"))
    g = s[(s.src == "mp") & (s["mode"] == "generic")].set_index("exercise")
    m = s[(s.src == "gt") & (s["mode"] == "generic")].set_index("exercise")
    fig, ax = plt.subplots(figsize=(12, 4.2))
    x = np.arange(len(g))
    ax.bar(x - 0.3, g.posture, 0.2, color=GREY, label="DTW posture only (camera)")
    ax.bar(x - 0.1, g.similarity, 0.2, color="#9CA3AF", label="NCC only (camera)")
    ax.bar(x + 0.1, g.global_weights, 0.2, color=TEAL, label="RehabNet score (camera)")
    ax.bar(x + 0.3, m.global_weights, 0.2, color=NAVY, label="RehabNet score (motion capture)")
    ax.axhline(0.5, color=RED, lw=1, ls="--"); ax.text(len(g) - 0.45, 0.47, "chance", color=RED, fontsize=9, ha="right", va="top")
    ax.set_xticks(x, [NICE[i] for i in g.index], rotation=12); ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("AUC (correct vs incorrect)"); ax.legend(frameon=False, ncol=2, fontsize=9, loc="upper left")
    ax.set_title("How well scores separate the physiotherapist's correct / incorrect labels (unseen people)")
    save(fig, "fig_score_auc.png")


def skeleton_corr():
    u = pd.read_csv(os.path.join(R, "score_uiprmd_spearman.csv"))
    k = pd.read_csv(os.path.join(R, "score_kimore_spearman.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw=dict(width_ratios=[2, 1]))
    for ax, d, t in ((axs[0], u, "UI-PRMD: vs quality score"), (axs[1], k, "KIMORE: vs clinician score")):
        x = np.arange(len(d))
        ax.bar(x, d.rehabnet_score, color=TEAL)
        for i, v in enumerate(d.rehabnet_score):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_xticks(x, d.exercise); ax.set_ylim(0, 1); ax.set_title(t); ax.set_ylabel("Spearman correlation")
    save(fig, "fig_skeleton_corr.png")


def examples():
    """Score a full session (one recording) and show the feedback per repetition."""
    rows = []
    for exn in ("squat", "arm_abduction", "leg_abduction"):
        s = [x for x in STREAMS if x["exercise"] == exn and x["view"] == "front"][0]
        refs = [r["mp"] for r in REPS if r["exercise"] == exn and r["side"] == s["side"] and r["person"] != s["person"] and r["correct"]]
        m = ReferenceModel.build(exn, s["side"], refs)
        mine = [r for r in REPS if r["video"] == s["video"] and r["cam"] == s["cam"]]
        res = []
        for r in mine:
            out = m.score(r["mp"])
            res.append(out)
            fb = feedback(out, exn, s["side"])
            rows.append(dict(exercise=NICE[exn], video=s["video"], rep=r["rep"], label="correct" if r["correct"] else "incorrect",
                             score=round(out["score"]), feedback=" ".join(x["text"] for x in fb)))
        summ = session_summary(res, [len(r["mp"]) / 10 for r in mine])
        rows.append(dict(exercise=NICE[exn], video=s["video"], rep="session", label="",
                         score=round(summ["mean_score"]),
                         feedback=f"{summ['reps']} reps, consistency {summ['consistency']:.0f}/100, weakest factor: {summ['weakest']}"))
    pd.DataFrame(rows).to_csv(os.path.join(R, "examples_feedback.csv"), index=False)


if __name__ == "__main__":
    pipeline(); counter_example(); counter_results(); dtw_example(); score_auc(); skeleton_corr(); examples()
    print("ok")
