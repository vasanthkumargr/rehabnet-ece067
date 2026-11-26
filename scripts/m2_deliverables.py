"""Build the Milestone 2 report (.docx) and slides (.pptx) from results/m2."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rehabnet.core.exercises import EXERCISES  # noqa: E402
from rehabnet.core.feedback import THRESH  # noqa: E402
from rehabnet.core.scoring import WEIGHTS  # noqa: E402
from tools.deliver import Deck, Report  # noqa: E402
from scripts.common_content import deck_intro, introduction, refs  # noqa: E402

R = os.path.join(ROOT, "results", "m2")
OUT = os.path.join(ROOT, "deliverables", "M2")
F = lambda n: os.path.join(R, n)  # noqa: E731
NICE = {k: v.name for k, v in EXERCISES.items()}
TITLE = "Milestone 2: Exercise Accuracy Engine"
SUB = "Repetition counting, DTW alignment, NCC similarity, multi-factor quality score and feedback"
PERIOD = "Planned: August - September 2026"


def build():
    tune = pd.read_csv(F("counter_tuning.csv"))
    rep_ov = pd.read_csv(F("reps_rehab24_overall.csv")).set_index("src")
    up = pd.read_csv(F("reps_uiprmd.csv"))
    km = pd.read_csv(F("reps_kimore_summary.csv"))
    auc = pd.read_csv(F("score_rehab24_auc.csv"))
    ov = pd.read_csv(F("score_rehab24_overall.csv")).set_index(["src", "mode"])
    ui = pd.read_csv(F("score_uiprmd_spearman.csv"))
    ki = pd.read_csv(F("score_kimore_spearman.csv"))
    fr = pd.read_csv(F("feedback_rates.csv")).set_index("exercise")
    ex = pd.read_csv(F("examples_feedback.csv"))

    # --------------------------------------------------------------- numbers
    agg = lambda g: pd.Series(dict(  # noqa: E731
        recordings=len(g), reps=int(g["gt"].sum()), precision=100 * g["tp"].sum() / g["pred"].sum(),
        recall=100 * g["tp"].sum() / g["gt"].sum(), exact=100 * (g.err == 0).mean(), within1=100 * (g.err <= 1).mean(),
        mae=g.err.mean()))
    c_all = agg(tune)
    c_fh = agg(tune[tune.view != "profile"])
    c_ex = tune.groupby("exercise").apply(agg).reset_index()
    c_ex["exercise"] = c_ex.exercise.map(NICE)
    c_vw = tune.groupby("view").apply(agg).loc[["front", "half-profile", "profile"]].reset_index()
    up_one = 100 * (up["count"] == 1).mean()
    up_ex = up.groupby("exercise").agg(count_angle=("count_angle", "first"), sequences=("count", "size"),
                                       exactly_one=("count", lambda c: 100 * (c == 1).mean())).reset_index()
    cam = ov.loc[("mp", "generic")]
    moc = ov.loc[("gt", "generic")]
    mocp = ov.loc[("gt", "personal")]
    camp = ov.loc[("mp", "personal")]
    camv = ov.loc[("mp", "generic_view")]
    a_cam = auc[(auc.src == "mp") & (auc["mode"] == "generic")].set_index("exercise")
    a_moc = auc[(auc.src == "gt") & (auc["mode"] == "generic")].set_index("exercise")
    auc_tab = pd.DataFrame({"Exercise": [NICE[e] for e in a_cam.index],
                            "DTW posture": a_cam.posture.values, "NCC similarity": a_cam.similarity.values,
                            "ROM": a_cam.rom.values, "Stability": a_cam.stability.values,
                            "RehabNet score (camera)": a_cam.global_weights.values,
                            "RehabNet score (mocap)": a_moc.loc[a_cam.index].global_weights.values})
    mode_tab = pd.DataFrame([
        ["Camera (MediaPipe)", "other people, all views", cam.global_weights_auc],
        ["Camera (MediaPipe)", "other people, same view only", camv.global_weights_auc],
        ["Camera (MediaPipe)", "patient's own first 5 correct reps", camp.global_weights_auc],
        ["Motion capture", "other people", moc.global_weights_auc],
        ["Motion capture", "patient's own first 5 correct reps", mocp.global_weights_auc],
    ], columns=["Angles from", "Reference repetitions", "Mean AUC"])
    w_tab = pd.DataFrame([[k, f"{v:.2f}"] for k, v in WEIGHTS.items()], columns=["Factor", "Weight"])
    ui_t = ui[["exercise", "key_angles", "recordings", "rehabnet_score", "posture", "stability"]].copy()
    ui_t.columns = ["Exercise", "Key angles (auto)", "Recordings", "RehabNet score", "DTW posture", "Stability"]
    ki_t = ki[["exercise", "key_angles", "recordings", "rehabnet_score", "posture", "similarity"]].copy()
    ki_t.columns = ["Exercise", "Key angles (auto)", "Recordings", "RehabNet score", "DTW posture", "NCC"]
    fr_t = fr.reset_index()
    fr_t["exercise"] = fr_t.exercise.map(lambda e: NICE.get(e, "All exercises"))
    fr_t = fr_t.rename(columns={"exercise": "Exercise"})
    fr_t[fr_t.columns[1:]] = fr_t[fr_t.columns[1:]] * 100
    fr_t.columns = ["Exercise", "Incorrect reps with no correction (%)", "Correct reps told 'good' (%)"]
    fr_t = fr_t[["Exercise", "Correct reps told 'good' (%)", "Incorrect reps with no correction (%)"]]
    fr_t["Incorrect reps that got a correction (%)"] = 100 - fr_t["Incorrect reps with no correction (%)"]
    fr_t = fr_t.drop(columns=["Incorrect reps with no correction (%)"])
    ex_show = ex[ex.exercise == "Leg abduction"][["rep", "label", "score", "feedback"]].copy()
    ex_show.columns = ["Rep", "Physio label", "Score", "Feedback shown to the patient"]

    st = pd.DataFrame([
        ["DTW temporal alignment to reference demonstrations", "Done", "multivariate DTW on key joint angles (dtaidistance)"],
        ["NCC movement-similarity module", "Done", "max normalised cross-correlation after resampling"],
        ["Angle-threshold repetition counter, validated", "Done", "cycle detector; REHAB24-6 + UI-PRMD (+ KIMORE)"],
        ["Multi-factor score: posture, ROM, stability, tempo", "Done", "weights learned from physio labels"],
        ["Deliverable: per-session quality score", "Done", "session summary + per-rep feedback messages"],
    ], columns=["Task (from proposal)", "Status", "How"])

    # --------------------------------------------------------------- report
    rp = Report(TITLE, SUB, PERIOD)
    rp.header_footer("Milestone 2")
    rp.abstract(
        "This report describes the exercise-analysis engine of RehabNet. Joint angles from the camera pipeline of Milestone 1 "
        "are turned into a repetition count, a 0-100 quality score per repetition and short corrective messages. Repetitions "
        "are counted by a swing (cycle) detector on a smoothed counting angle. Each repetition is compared with correct "
        "reference repetitions using multivariate Dynamic Time Warping (posture) and normalised cross-correlation (shape), "
        "together with range of motion, stability, tempo and symmetry; the factor weights were learned from a "
        "physiotherapist's correct/incorrect labels. Evaluated on people never seen during tuning, the counter found "
        f"{c_all.recall:.0f} % of repetitions with {c_all.precision:.0f} % precision, and the score separated correct from "
        f"incorrect repetitions with a mean AUC of {cam.global_weights_auc:.2f}. The score correlates with UI-PRMD quality "
        f"scores (Spearman {ui.rehabnet_score.mean():.2f}) and weakly with KIMORE clinician scores ({ki.rehabnet_score.mean():.2f}).",
        ["repetition counting", "dynamic time warping", "normalised cross-correlation", "movement quality", "feedback"])
    rp.toc()
    introduction(rp, 2, [
        ("Count repetitions", "live, from the joint angles, without the patient pressing anything."),
        ("Compare with a reference", "using DTW alignment and NCC similarity, as planned in the proposal."),
        ("Score quality", "with several factors (posture, range of motion, stability, tempo) combined into one score."),
        ("Give feedback", "a short, specific correction after each repetition and a session summary."),
    ], "All tests in this milestone use the camera-based angles (not motion capture) and people who were not used for "
       "tuning (leave-one-person-out), so the numbers reflect what a new patient would experience.")
    rp.h1("Related work")
    rp.p("Comparing two movements of different speed is a classic time-series problem. Dynamic Time Warping (DTW) [5] finds "
         "the best alignment between two sequences by stretching and compressing time, so a slow and a fast squat can be "
         "compared phase by phase; efficient implementations exist for multivariate sequences [6]. Normalised "
         "cross-correlation (NCC) [7] measures how similar the shapes of two signals are, independent of their amplitude, "
         "and is a standard template-matching measure.")
    rp.p("For rehabilitation scoring, the UI-PRMD [9] and KIMORE [8] datasets made it possible to train models that predict "
         "movement quality from skeletons. Deep networks reach high correlations on these datasets [11], but they need large "
         "amounts of labelled data per exercise and give no explanation that a patient could act on. Distance-based "
         "approaches compare a repetition with correct examples and can say which joint deviated. Jánošová et al. [10] "
         "showed on REHAB24-6 that comparing a person with their own correct repetitions can work better than comparing "
         "with others; we test this for camera input in Section 6.4.")
    rp.p("We evaluate separation of correct and incorrect repetitions with the area under the ROC curve (AUC) [12]: the "
         "probability that a randomly chosen correct repetition gets a higher score than a randomly chosen incorrect one. "
         "Factor weights are fitted with logistic regression [13].")
    rp.table(pd.DataFrame([
        ["Fixed angle thresholds", "Simple", "Fails when the patient does not return fully", "First version, replaced"],
        ["Swing / cycle detection", "Robust to drift and partial returns", "Needs a typical amplitude", "RehabNet counter"],
        ["DTW distance", "Handles speed differences", "Sensitive to noise", "Posture factor"],
        ["NCC", "Shape only, amplitude-free", "Ignores size of movement", "Similarity factor"],
        ["Deep-learning regressors", "High accuracy on large data", "Data-hungry, no explanation", "Not used"],
    ], columns=["Technique", "Strength", "Weakness", "Use in RehabNet"]), "Techniques for counting and comparing movements", font=9)
    rp.h1("Summary of results")
    rp.p("Milestone 2 turns the joint angles from Milestone 1 into the numbers a patient and doctor see: how many "
         "repetitions were done, how good each one was, and what to correct. Everything was tested on people the system "
         "had never seen (leave-one-person-out), using the camera-based angles, not motion capture.")
    rp.bullets([
        ("Repetition counter:", f"finds {c_all.recall:.1f} % of repetitions and {c_all.precision:.1f} % of what it counts are real "
                                f"repetitions. With the camera in front or at 45 deg the count is off by {c_fh.mae:.1f} repetitions per "
                                f"recording of ~16 ({c_fh.within1:.0f} % of recordings within one); squats are almost exact. On UI-PRMD it "
                                f"reports exactly one repetition for {up_one:.0f} % of single-repetition recordings."),
        ("Quality score:", f"separates the physiotherapist's correct and incorrect repetitions with an average AUC of "
                           f"{cam.global_weights_auc:.2f} (0.5 = chance, 1 = perfect), between {a_cam.global_weights.min():.2f} "
                           f"(lunge) and {a_cam.global_weights.max():.2f} (push-up). It correlates with UI-PRMD quality scores "
                           f"(mean Spearman {ui.rehabnet_score.mean():.2f}) and, weakly, with KIMORE clinician scores ({ki.rehabnet_score.mean():.2f})."),
        ("Feedback:", f"{fr.loc['ALL', 'correct: told good']*100:.0f} % of correct repetitions are told 'good', and "
                      f"{100 - fr.loc['ALL', 'incorrect: no correction']*100:.0f} % of incorrect repetitions get a specific "
                      "correction (e.g. 'Keep your upper body upright')."),
        ("Honest limit:", "the score is a useful guide, not a diagnosis: about 3 in 10 correct/incorrect pairs are ranked the "
                          "wrong way round. Lunges, whose typical mistakes (knee alignment) are not captured by joint angles, are the weakest."),
    ])
    rp.h2("Milestone tasks and status")
    rp.table(st, "Milestone 2 tasks")

    rp.h1("2. Engine overview")
    rp.figure(F("fig_m2_pipeline.png"), "Milestone 2 processing chain.", 16)
    rp.p("All scoring works on the 3-D joint angles from Milestone 1 at 10 samples per second. Each exercise in the "
         "library defines its key angles (the joints that should move), the counting angle, the joints that should stay "
         "still (stability), and left/right pairs (symmetry). Six exercises from REHAB24-6 make up the first library: "
         + ", ".join(v.name.lower() for v in EXERCISES.values()) + ".")

    rp.h1("3. Repetition counter")
    rp.h2("3.1 Method")
    rp.p("The counting signal is the mean of the exercise's counting angle(s), e.g. both knee angles for a squat. It is "
         "smoothed with a causal exponential filter (time constant 0.8 s) so that it can run live. A repetition is a full "
         "cycle: the signal must move away from its last low point by at least H degrees, reach a peak, and come back by H. "
         "H is 25 % of the typical movement amplitude learned from correct reference repetitions. Because the counter looks "
         "at swings, not fixed angle levels, it still works when a patient does not return fully to the start position "
         "between repetitions (as in lunges) or drifts slowly.")
    rp.p("The first version used fixed angle levels (the proposal's 'angular threshold') and counted lunges badly, because "
         "patients stayed low between repetitions. The smoothing, H and the minimum repetition length (0.8 s) were then "
         "chosen on half of the people and tested on the other half, and vice versa, so the results below are on unseen people.")
    rp.figure(F("fig_counter_example.png"), "Counter on a squat and a lunge recording (front camera). Shaded: annotated "
              "repetitions; orange: counted repetitions.", 16)
    rp.h2("3.2 Results")
    rp.table(c_ex.round(1).rename(columns={"exercise": "Exercise", "recordings": "Recordings", "reps": "Reps",
                                           "precision": "Precision %", "recall": "Recall %", "exact": "Exact count %",
                                           "within1": "Within 1 %", "mae": "Count error"}),
             "Counter on REHAB24-6 camera data, unseen people (count error = mean |counted - annotated| per recording)", font=8.5)
    rp.table(c_vw.round(1).rename(columns={"view": "Camera view", "recordings": "Recordings", "reps": "Reps",
                                           "precision": "Precision %", "recall": "Recall %", "exact": "Exact count %",
                                           "within1": "Within 1 %", "mae": "Count error"}), "Counter by camera view", font=8.5)
    rp.figure(F("fig_counter_results.png"), "Counter precision/recall by exercise and count error by camera view.", 16)
    rp.p(f"With motion-capture angles the same counter is off by {rep_ov.loc['gt', 'mae_count']:.1f} repetitions per recording "
         f"(camera: {rep_ov.loc['mp', 'mae_count']:.1f} with the same settings), so most remaining counting errors come from "
         "camera noise. Leg abduction seen from the side is the worst case: the sideways leg lift points towards the camera and "
         "is barely visible. The app will therefore ask for a front (or 45 deg) view for each exercise.")
    rp.table(up_ex.round(1).rename(columns={"exercise": "UI-PRMD exercise", "count_angle": "Counting angle (auto)",
                                            "sequences": "Recordings", "exactly_one": "Counted exactly 1 (%)"}),
             "UI-PRMD: each recording holds one repetition", font=8.5)
    rp.p("The two UI-PRMD exercises below 80 % are shoulder extension and internal-external shoulder rotation: rotation "
         "of the arm around its own axis barely changes any joint angle we measure, so it cannot be counted this way. "
         f"KIMORE has no repetition labels; the counter found a median of {km.set_index('exercise').loc['LA', '50%']:.0f} repetitions "
         f"for lifting arms and {km.set_index('exercise').loc['Sq', '50%']:.0f} for squats, but implausible counts for the pelvis and trunk "
         "rotation exercises, again because rotations are not captured by joint angles.")

    rp.h1("4. Scoring engine")
    rp.h2("4.1 Factors")
    rp.bullets([
        ("Posture (DTW):", "multivariate Dynamic Time Warping distance between the repetition and its 3 nearest correct "
                           "reference repetitions, on the key angles (each divided by its spread), normalised by length. DTW "
                           "stretches time so a slower repetition is compared with the matching phase of the reference."),
        ("Similarity (NCC):", "normalised cross-correlation between the repetition and those references after resampling "
                              "to the same length, maximised over small time shifts; it measures shape, independent of size."),
        ("Range of motion:", "how far each moving joint's range differs from the references (too small or too large)."),
        ("Stability:", "extra movement of joints that should stay still (trunk, standing knee, hips in a push-up)."),
        ("Tempo:", "repetition duration compared with the references."),
        ("Symmetry:", "left-right angle difference for two-sided exercises."),
    ])
    rp.p("Each factor becomes a 0-100 sub-score by comparing it with how much the same factor varies among the correct "
         "references themselves: typical of a correct repetition = 100, as bad as the worst 5 % of correct repetitions = 50, "
         "twice that = 0. The repetition score is a weighted mean of the sub-scores.")
    rp.h2("4.2 Weights learned from physiotherapist labels")
    rp.p("Instead of guessing the weights, a logistic regression was fitted on the sub-scores of all REHAB24-6 repetitions to "
         "predict the physiotherapist's correct/incorrect label, with one weight set for all exercises. Evaluated on unseen "
         f"people, the learned weights reached a mean AUC of {cam.global_weights_auc:.3f} against 0.674 "
         "for hand-set weights. Range of motion, tempo and symmetry received zero weight: in this dataset they do not tell "
         "correct from incorrect, because the physiotherapist's 'incorrect' repetitions are mostly about posture and "
         "compensation, not speed or size. They are still shown to the doctor as separate sub-scores and used for feedback.")
    rp.table(w_tab, "Final factor weights in the repetition score")
    rp.h2("4.3 Results on REHAB24-6")
    rp.table(auc_tab.round(3), "AUC for correct vs incorrect repetitions, unseen people (camera angles unless stated)", font=8.5)
    rp.figure(F("fig_score_auc.png"), "How well each score separates correct from incorrect repetitions.", 16)
    rp.figure(F("fig_dtw_example.png"), "Left/middle: DTW alignment of a correct and an incorrect squat to their nearest "
              "correct reference (grey lines show matched time points). Right: score distribution for correct and incorrect "
              "repetitions over all exercises.", 16)
    rp.h2("4.4 Which references?")
    rp.p("Three ways of choosing reference repetitions were compared on exactly the same test repetitions (each patient's "
         "first 5 correct repetitions are held back as a possible personal calibration).")
    rp.table(mode_tab.round(3), "Choice of reference repetitions (mean AUC over exercises, unseen people)")
    rp.p("With accurate motion-capture angles, comparing a patient with their own earlier correct repetitions is clearly better "
         f"({mocp.global_weights_auc:.2f} vs {moc.global_weights_auc:.2f}), matching what Jánošová et al. [10] reported. "
         "With camera angles it was not: five noisy camera repetitions are not a stable enough reference. Restricting "
         "references to the same camera view also hurt, because it leaves too few of them. RehabNet therefore uses other "
         "people's correct repetitions from all views by default; personal references are revisited in Milestone 4 with more "
         "calibration repetitions.")
    rp.h2("4.5 Results on UI-PRMD and KIMORE")
    rp.p("These skeleton datasets give a continuous score per recording (UI-PRMD quality score, KIMORE clinician score). "
         "Each recording was scored against the better-scored half of the other people's recordings, with key angles chosen "
         "automatically (the angles that move most) and the weights learned on REHAB24-6 unchanged.")
    rp.table(ui_t.round(3), "UI-PRMD: Spearman correlation with the quality score", font=8.5)
    rp.table(ki_t.round(3), "KIMORE: Spearman correlation with the clinician score", font=8.5)
    rp.figure(F("fig_skeleton_corr.png"), "Correlation of the RehabNet score with dataset scores.", 16)
    rp.p("UI-PRMD correlates well for most exercises. KIMORE is weak: its clinician score largely describes the person "
         "(our earlier analysis found that subject identity dominates KIMORE scores), and two of its five exercises are "
         "rotations that joint angles do not capture.")

    rp.h1("5. Feedback and session summary")
    rp.p("For each repetition the factor values are turned into at most two plain-language messages. A message is only "
         "given when a sub-score is below a threshold chosen so that it would fire on at most ~10 % of correct repetitions "
         "of unseen people (5 % for range of motion, tempo and symmetry). Range-of-motion feedback is limited to each "
         "exercise's main joint, which a single camera measures most reliably (Milestone 1 showed hip flexion seen from the "
         "front is unreliable).")
    rp.table(pd.DataFrame([[k, v] for k, v in THRESH.items()], columns=["Sub-score", "Message if below"]), "Feedback thresholds")
    rp.table(fr_t.round(1), "Feedback on unseen people")
    rp.table(ex_show, "Example: one leg-abduction recording (front camera) scored repetition by repetition", font=8)
    rp.p("Per session RehabNet stores the repetition count, the mean, best and worst repetition score, a consistency value "
         "(100 minus the coefficient of variation of the scores) and the weakest factor. These are the numbers the doctor "
         "dashboard (Milestone 3) and progress tracking (Milestone 4) will show.")

    rp.h1("6. Findings")
    rp.bullets([
        ("Counting works.", "Swing-based counting with smoothing is reliable for front/45 deg views; squats are almost exact."),
        ("DTW is the strongest single factor;", "NCC adds a little; stability catches compensation (leaning, bending the standing knee)."),
        ("Learned beats guessed.", "Learning the weights from the physiotherapist's labels improved separation and showed which factors matter."),
        ("Camera noise is the main limit.", "The same engine on motion-capture angles is better, especially with personal references."),
        ("Lunge mistakes are not visible in joint angles.", "Knee-over-toe or knee-inward errors need extra measures (Milestone 4/5)."),
    ])
    rp.h2("Limitations")
    rp.bullets([
        "REHAB24-6: 10 healthy adults with simulated mistakes; patients may make different mistakes.",
        "Thresholds and weights were chosen on REHAB24-6 (with person-wise checks); new exercises need reference recordings.",
        "UI-PRMD/KIMORE recordings are resampled to a fixed length in the Rehab-Pile release, so tempo cannot be tested on them.",
    ])
    rp.h1("Conclusion")
    rp.p(f"Milestone 2 produced a complete exercise-analysis engine. The repetition counter is reliable for front and 45° "
         f"camera views (about {c_fh.mae:.1f} repetitions off per set), and the quality score ranks correct above incorrect "
         f"repetitions in about {cam.global_weights_auc*100:.0f} % of pairs on people it has never seen. Learning the weights from the "
         "physiotherapist's labels showed that posture (DTW), shape (NCC) and stability carry the information, while range "
         "of motion, tempo and symmetry are better used as feedback topics than as score components. The engine is fast "
         "enough to run live, which is what Milestone 3 builds on.")
    rp.callout(["Score = 0.40 posture + 0.29 similarity + 0.31 stability (learned).",
                "References = other people's correct repetitions, all camera views.",
                "Feedback only when a sub-score is clearly worse than correct repetitions (≤ 10 % false alarms).",
                "The app asks for a front or 45° camera view, where counting and scoring are most reliable."],
               title="Decisions carried forward")
    rp.h1("Next: Milestone 3 (October - November 2026)")
    rp.bullets(["FastAPI backend serving the counter, scoring and feedback.",
                "Doctor portal: exercise library, assigning exercises to patients, reports.",
                "Patient portal: exercise guide, live camera pose with on-screen counter and feedback, session summary.",
                "Encrypted storage and two-factor login."])
    rp.references(refs("cieza", "argent", "blazepose", "rehab24", "sakoe", "dtaidistance", "lewis", "kimore", "uiprmd",
                       "janosova", "liao", "hanley", "sklearn"))
    rp.h1("Appendix: code")
    rp.code("rehabnet/core/exercises.py   exercise library (key, counting, stability angles, cues)\n"
            "rehabnet/core/reps.py        live repetition counter\n"
            "rehabnet/core/scoring.py     DTW / NCC / factors / weighted score\n"
            "rehabnet/core/feedback.py    feedback messages, session summary\n"
            "scripts/m2_build_reps.py     per-repetition dataset from Milestone 1 poses\n"
            "scripts/m2_counter_tune.py   person-split choice of counter settings\n"
            "scripts/m2_reps_eval.py      counter validation (REHAB24-6, UI-PRMD, KIMORE)\n"
            "scripts/m2_score_eval.py     scoring validation and weight learning")
    rp.save(os.path.join(OUT, "M2_Report.docx"))

    # --------------------------------------------------------------- slides
    dk = Deck(TITLE, SUB, PERIOD)
    dk.agenda([("Problem and plan", "where Milestone 2 fits"), ("Engine overview", "angles in, count + score + feedback out"),
               ("Repetition counter", "method and results on unseen people"), ("Scoring engine", "DTW, NCC, factors, learned weights"),
               ("Feedback", "messages and session summary"), ("Findings and next steps", "what works, what does not")],
              say="Milestone 2 is the brain of RehabNet: it decides how many repetitions were done and how good they were.")
    deck_intro(dk, 2, [
        ("Count repetitions", "live, from joint angles"),
        ("Compare with a reference", "DTW alignment + NCC similarity"),
        ("Score quality", "posture, range of motion, stability, tempo -> one score"),
        ("Give feedback", "one specific correction after each repetition"),
    ])
    dk.stats("Milestone 2 in numbers", [
        (f"{c_all.recall:.0f} %", "repetitions found", "recall on unseen people"),
        (f"{c_all.precision:.0f} %", "counts are real", "precision"),
        (f"±{c_fh.mae:.1f}", "reps per set", "front / 45° camera, ~16-rep sets"),
        (f"{cam.global_weights_auc:.2f}", "AUC correct vs incorrect", "camera angles, unseen people"),
        (f"{fr.loc['ALL', 'correct: told good']*100:.0f} %", "correct reps told 'good'", "few false alarms"),
    ], kicker="Summary", takeaway="Counting is reliable; the score is a useful guide that agrees with the physiotherapist about 7 times in 10.",
        say="All numbers are on people the system never saw during tuning, using camera angles - the honest setting.")
    dk.table("Milestone 2 tasks", st, kicker="Status", font=14, col_w=[5.6, 1.4, 5.0],
             takeaway="All five planned tasks are done; the counter design changed from fixed thresholds to swings.")
    dk.cards("Counting and comparing movements: options", [
        ("Fixed angle thresholds", ["simple", "fails if the patient does not return fully", "-> first version, replaced"]),
        ("Swing detection", ["robust to drift and partial returns", "-> RehabNet counter"]),
        ("DTW", ["aligns different speeds phase by phase", "-> posture factor"]),
        ("NCC", ["compares shape, not size", "-> similarity factor"]),
        ("Deep learning", ["accurate on large labelled data", "no explanation for the patient", "-> not used"]),
        ("Personal references", ["compare with your own correct reps", "tested in this milestone"]),
    ], kicker="Related work", icons=False, takeaway="We chose explainable, distance-based methods so every score can come with a reason.")
    dk.image("Engine overview", F("fig_m2_pipeline.png"), kicker="Method",
             takeaway="Everything runs on 3-D joint angles at 10 samples per second, fast enough for live use.")

    dk.section(1, "Repetition counter", "Counting swings of the movement, not fixed angle levels")
    dk.flow("How a repetition is counted", [
        ("Counting signal", ["mean of the exercise's counting angle(s)", "e.g. both knees for a squat"]),
        ("Smooth", ["causal exponential filter", "time constant 0.8 s", "works live"]),
        ("Swing out", ["move away from the last low point", "by at least H degrees"]),
        ("Swing back", ["peak, then return by H", "-> 1 repetition, at least 0.8 s long"]),
    ], kicker="Counting", note="H = 25 % of the typical amplitude of correct repetitions; settings chosen on half the people, tested on the other half",
        takeaway="Swings work even when a patient stays low between lunges or drifts slowly.")
    dk.image("Counter in action", F("fig_counter_example.png"), kicker="Counting",
             takeaway="Shaded = annotated repetitions, orange = counted: the counter matches the physiotherapist's annotations.")
    dk.image("Counter results (unseen people)", F("fig_counter_results.png"), kicker="Counting",
             notes=[f"Front/45°: off by {c_fh.mae:.1f} reps per ~16-rep set", "Squat almost exact",
                    "Side view hides leg abduction", f"UI-PRMD: exactly 1 rep in {up_one:.0f} %"],
             takeaway="The app asks for a front or 45° view, where counting is most reliable.")

    dk.section(2, "Scoring engine", "Comparing each repetition with correct reference repetitions")
    dk.cards("Six quality factors", [
        ("Posture (DTW)", "time-warped distance to the 3 nearest correct reference reps"),
        ("Similarity (NCC)", "shape correlation after alignment"),
        ("Range of motion", "too small or too large vs references"),
        ("Stability", "extra movement of joints that should stay still"),
        ("Tempo", "duration vs references"),
        ("Symmetry", "left-right difference for two-sided exercises"),
    ], kicker="Scoring", takeaway="Each factor -> 0-100 sub-score: 100 = typical correct rep, 50 = as bad as the worst 5 % of correct reps.")
    dk.table("Weights learned from physiotherapist labels", w_tab, kicker="Scoring", font=16,
             notes=[f"Learned: AUC {cam.global_weights_auc:.3f}", "Hand-set: AUC 0.674",
                    "ROM / tempo / symmetry did not predict the label", "They stay as sub-scores and feedback topics"],
             takeaway="Learning the weights beat guessing them and showed which factors matter.")
    dk.image("Correct vs incorrect: AUC per exercise", F("fig_score_auc.png"), kicker="Results",
             takeaway=f"From {a_cam.global_weights.min():.2f} (lunge) to {a_cam.global_weights.max():.2f} (push-up); 0.5 would be guessing.")
    dk.image("DTW alignment and score distribution", F("fig_dtw_example.png"), kicker="Results",
             takeaway="DTW lines up matching phases; incorrect repetitions get lower scores on average.")
    dk.table("Which reference repetitions?", mode_tab.round(3), kicker="Results", font=14,
             notes=["Personal refs win with motion capture", "Not with 5 noisy camera reps",
                    "Default: other people, all views", "Revisited in Milestone 4"],
             takeaway="With a camera, other people's correct repetitions are the more stable reference.")
    dk.image("UI-PRMD and KIMORE", F("fig_skeleton_corr.png"), kicker="Results",
             caption=f"UI-PRMD mean r = {ui.rehabnet_score.mean():.2f}; KIMORE mean r = {ki.rehabnet_score.mean():.2f}",
             takeaway="Good agreement on UI-PRMD; KIMORE scores mostly describe the person, and its rotations are invisible to joint angles.")

    dk.section(3, "Feedback", "Turning numbers into one clear correction")
    dk.cards("How feedback is chosen", [
        ("Only when clearly wrong", "a message fires only if the sub-score is worse than ~90 % of correct reps"),
        ("At most two messages", "the weakest factors first, in plain language"),
        ("Main joint only for ROM", "the joint a single camera measures best"),
        ("Pairs judged together", "e.g. both knees in a squat"),
        ("Praise when good", "'Good repetition - keep it up!'; score ≥ 85 adds one tip"),
        ("Session summary", "count, mean / best / worst score, consistency, weakest factor"),
    ], kicker="Feedback")
    dk.table("Feedback on unseen people", fr_t.round(1), kicker="Feedback", font=13,
             takeaway="Most correct repetitions are praised; about half of the incorrect ones get a specific correction.")
    ex_sl = ex_show[ex_show.Rep.astype(str).isin(["1", "2", "3", "13", "15", "19", "session"])]
    dk.table("Example session: leg abduction", ex_sl, kicker="Feedback", font=12, col_w=[1.0, 1.4, 0.9, 8.7],
             takeaway="Incorrect repetitions get lower scores and a concrete instruction.")

    dk.cards("Findings and limits", [
        ("Works", "counting (front/45°), DTW + stability scoring, targeted feedback"),
        ("Guide, not diagnosis", f"AUC {cam.global_weights_auc:.2f}: about 3 in 10 pairs ranked the wrong way"),
        ("Weakest exercise", "lunge: knee-alignment mistakes are not visible in joint angles"),
        ("Camera noise", "the main limit - motion-capture angles score better"),
        ("Data", "10 healthy adults with simulated mistakes"),
        ("Rotations", "shoulder / trunk rotations cannot be measured with these angles"),
    ], kicker="Conclusions", takeaway="Milestone 3 puts this engine behind a live web app for patients and doctors.")
    dk.flow("Next: Milestone 3 (Oct - Nov 2026)", [
        ("FastAPI backend", "serves counter, scoring and feedback over a WebSocket"),
        ("Doctor portal", "patients, prescriptions, session history"),
        ("Patient portal", "live camera skeleton, counter and spoken feedback"),
        ("Security", "encrypted storage, two-factor login"),
    ], kicker="Plan")
    dk.closing(lines=["Questions and discussion", "Code: rehabnet/core/{reps, scoring, feedback}.py  ·  scripts/m2_*.py"])
    dk.save(os.path.join(OUT, "M2_Presentation.pptx"))
    print("saved", OUT)


if __name__ == "__main__":
    build()
