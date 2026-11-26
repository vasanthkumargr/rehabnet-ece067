"""Build the Milestone 1 report (.docx) and slide deck (.pptx) from results/m1."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.deliver import Deck, Report  # noqa: E402
from scripts.common_content import deck_intro, introduction, refs  # noqa: E402

R = os.path.join(ROOT, "results", "m1")
OUT = os.path.join(ROOT, "deliverables", "M1")
TAG = os.environ.get("TAG", "")
F = lambda n: os.path.join(R, n)  # noqa: E731
EXN = {1: "Arm abduction", 2: "Arm V-W", 3: "Push-ups", 4: "Leg abduction", 5: "Leg lunge", 6: "Squat"}
PRETTY = {"l_elbow": "L elbow", "r_elbow": "R elbow", "l_shoulder": "L shoulder", "r_shoulder": "R shoulder",
          "l_hip": "L hip flexion", "r_hip": "R hip flexion", "l_knee": "L knee", "r_knee": "R knee",
          "trunk_lean": "Trunk lean", "hip_abd_l": "L hip abduction", "hip_abd_r": "R hip abduction"}
TITLE = "Milestone 1: Data and Pose-Estimation Foundation"
SUB = "Datasets, MediaPipe pipeline, keypoint normalisation, joint-angle validation"
PERIOD = "Planned: June - July 2026"


def load():
    d = {k: pd.read_csv(F(f"raw{TAG}_{k}.csv")) for k in ("detection", "keypoints", "angles", "rom", "normalisation")}
    d["rom"]["rom_err"] = (d["rom"].mp_rom - d["rom"].gt_rom).abs()
    d["inv"] = pd.read_csv(F("normalisation_invariance_by_cond.csv")) if os.path.exists(F("normalisation_invariance_by_cond.csv")) else None
    d["j30"] = pd.read_csv(F("raw_30fps_normalisation.csv")) if os.path.exists(F("raw_30fps_normalisation.csv")) else None
    d["speed"] = pd.read_csv(F("pose_model_speed.csv"))
    d["rp"] = pd.read_csv(F("rehabpile_summary.csv"))
    d["seg"] = pd.read_csv(os.path.join(ROOT, "data", "rehab24_6", "Segmentation.csv"), sep=";")
    return d


def numbers(d):
    det, kp, ang, rom = d["detection"], d["keypoints"], d["angles"], d["rom"]
    key = ang[ang.key]
    n = {}
    n["det"] = det.detected.mean() * 100
    n["det_txt"] = "more than 99.9 %" if n["det"] > 99.9 else f"{n['det']:.1f} %"
    n["det_view"] = det.groupby("view").detected.mean() * 100
    n["det_light"] = det.groupby("lights").detected.mean() * 100
    n["pck20"] = kp.pck20.mean() * 100
    n["pck_view"] = kp.groupby("view").pck20.mean() * 100
    n["nme_view"] = kp.groupby("view").nme.mean()
    n["mae"] = key.groupby("src").mae.mean()
    n["r"] = key.groupby("src").r.mean()
    n["mae_view"] = key.groupby(["src", "view"]).mae.mean().unstack(0)
    n["mae_ex"] = key[key.src == "mp3d_smooth"].groupby("ex").mae.mean()
    n["mae_angle"] = key[key.src == "mp3d_smooth"].groupby("angle")[["mae", "bias", "r"]].mean()
    n["rom_med"] = rom.groupby("src").rom_err.median()
    n["rom_mean"] = rom.groupby("src").rom_err.mean()
    n["rom_r"] = rom.groupby("src").apply(lambda x: x[["gt_rom", "mp_rom"]].corr().iloc[0, 1])
    n["rom_view"] = rom[rom.src == "mp3d_smooth"].groupby("view").rom_err.median()
    # best source per view (2-D vs 3-D)
    n["best_front"] = n["mae_view"].loc["front"].idxmin()
    return n


def build():
    d = load()
    n = numbers(d)
    seg = d["seg"]
    os.makedirs(OUT, exist_ok=True)
    det, kp, ang, rom = d["detection"], d["keypoints"], d["angles"], d["rom"]

    # ------------------------------------------------------------------ tables
    t_data = pd.DataFrame([
        ["REHAB24-6 (2024)", "Camera videos (2 views) + OptiTrack 3-D/2-D joints", "65 recordings, 10 people, 6 exercises",
         "1,072 reps: start/end frame, correct/incorrect", "Main: pose validation, angle ground truth, rep counting, correctness"],
        ["KIMORE (2019)", "Kinect v2 3-D skeleton", "355 recordings, 71 people, 5 exercises",
         "Clinical score 0-100 per exercise", "Scoring validation against clinician scores (M2)"],
        ["UI-PRMD (2018)", "Vicon 3-D skeleton", "1,326 reps, 10 people, 10 exercises",
         "Quality score per rep (correct vs incorrect)", "Reference templates, correct/incorrect scoring (M2)"],
    ], columns=["Dataset", "Data used", "Size", "Labels", "Role in RehabNet"])
    t_rp = d["rp"][["dataset", "exercise", "name", "recordings", "subjects", "score_min", "score_max"]].copy()
    t_rp.columns = ["Dataset", "Code", "Exercise", "Recordings", "People", "Score min", "Score max"]
    ex_tab = seg.groupby("exercise_id").agg(reps=("correctness", "size"), correct=("correctness", "sum"),
                                            people=("person_id", "nunique"), videos=("video_id", "nunique")).reset_index()
    ex_tab["exercise_id"] = ex_tab.exercise_id.map(EXN)
    ex_tab.columns = ["Exercise", "Repetitions", "Correct", "People", "Recordings"]

    det_tab = det.groupby("view").agg(videos=("video", "size"), frames=("frames", "sum"), detected=("detected", "mean")).reset_index()
    det_tab["detected"] = (det_tab.detected * 100).round(1)
    det_tab.columns = ["Camera view", "Video streams", "Frames evaluated", "Pose detected (%)"]

    kp_view = kp.groupby("view")[["nme", "pck10", "pck20"]].mean().reset_index()
    kp_view[["pck10", "pck20"]] *= 100
    kp_view.columns = ["Camera view", "Mean error (x torso)", "PCK@0.1 (%)", "PCK@0.2 (%)"]
    kp_light = kp.groupby("lights")[["nme", "pck20"]].mean().reset_index()
    kp_light["lights"] = kp_light.lights.map({0: "Lights off (evening)", 1: "Lights on"})
    kp_light["pck20"] *= 100
    kp_light.columns = ["Lighting", "Mean error (x torso)", "PCK@0.2 (%)"]
    order = ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
    kp_joint = kp.groupby("joint")[["nme", "pck20"]].mean().loc[order].reset_index()
    kp_joint["pck20"] *= 100
    kp_joint["joint"] = kp_joint.joint.str.replace("_", " ")
    kp_joint.columns = ["Joint", "Mean error (x torso)", "PCK@0.2 (%)"]

    src_names = {"mp2d": "2-D image angles", "mp3d": "3-D world angles", "mp3d_smooth": "3-D world + One-Euro"}
    ang_src = ang[ang.key].groupby("src")[["mae", "r"]].mean().reset_index()
    ang_src["src"] = ang_src.src.map(src_names)
    ang_src.columns = ["Angle source", "MAE (deg)", "Correlation r"]
    ang_view = ang[ang.key].groupby(["view", "src"]).mae.mean().unstack().rename(columns=src_names).reset_index()
    ang_view.columns.name = None
    ang_view = ang_view.rename(columns={"view": "Camera view"})
    ang_ex = ang[ang.key].groupby(["ex", "src"]).mae.mean().unstack().rename(columns=src_names).reset_index()
    ang_ex.columns.name = None
    ang_ex["ex"] = ang_ex.ex.map(EXN)
    ang_ex = ang_ex.rename(columns={"ex": "Exercise"})
    ang_angle = n["mae_angle"].reset_index()
    ang_angle["angle"] = ang_angle.angle.map(PRETTY)
    ang_angle.columns = ["Joint angle", "MAE (deg)", "Bias (deg)", "Correlation r"]
    rom_tab = rom.groupby(["src", "view"]).rom_err.median().unstack().reset_index()
    rom_tab.columns.name = None
    rom_tab["src"] = rom_tab.src.map(src_names)
    rom_tab = rom_tab.rename(columns={"src": "Angle source"})
    rom_tab["All views r"] = rom_tab["Angle source"].map({src_names[k]: v for k, v in n["rom_r"].items()})

    speed = d["speed"][d["speed"].input_px == 960][["model", "median_ms_per_frame", "fps"]]
    speed.columns = ["BlazePose model", "Time per frame (ms)", "Frames per second"]
    kg = pd.read_csv(F("pose_model_speed_kaggle_t4.csv"))
    kg = kg[kg.delegate == "cpu"].set_index("model")
    speed_both = pd.DataFrame({
        "BlazePose model": speed["BlazePose model"].values,
        "Laptop ms (Ryzen 3, 2 cores)": speed["Time per frame (ms)"].values,
        "Laptop fps": speed["Frames per second"].values,
        "Cloud CPU ms (Kaggle, 4 cores)": kg.loc[speed["BlazePose model"], "median_ms"].values,
        "Cloud fps": kg.loc[speed["BlazePose model"], "fps"].values})
    for c in speed_both.columns[1:]:
        speed_both[c] = speed_both[c].map(lambda v: f"{v:.1f}")

    inv = d["inv"]
    if inv is not None:
        inv_t = inv[inv.method.isin(["none", "bbox", "torso"])].pivot(index="method", columns="cond", values="diff_pct").loc[["none", "bbox", "torso"]].reset_index()
        inv_t["method"] = inv_t.method.map({"none": "None (raw pixels)", "bbox": "Bounding box", "torso": "Torso (hip-centred)"})
        inv_t.columns.name = None
        inv_t = inv_t.rename(columns={"method": "Normalisation", "far_left": "Shrunk 55 %, left (%)", "far_right": "Shrunk 70 %, right (%)"})
        ang_inv = inv[inv.method == "angles_deg"].set_index("cond").diff_pct
        det_far = inv[inv.method == "none"].set_index("cond").detected * 100
    j30 = d["j30"] if d["j30"] is not None else d["normalisation"]
    jit = dict(raw=j30.jitter_raw.mean(), sm=j30.jitter_smooth.mean(), e_raw=j30.nme_raw.mean(), e_sm=j30.nme_smooth.mean())
    jit["red"] = (1 - jit["sm"] / jit["raw"]) * 100

    best3 = n["mae"]["mp3d_smooth"]
    front2, front3 = n["mae_view"].loc["front", "mp2d"], n["mae_view"].loc["front", "mp3d_smooth"]
    prof2, prof3 = n["mae_view"].loc["profile", "mp2d"], n["mae_view"].loc["profile", "mp3d_smooth"]

    # ------------------------------------------------------------------ report
    rp = Report(TITLE, SUB, PERIOD)
    rp.header_footer("Milestone 1")
    rp.abstract(
        "This report describes the first milestone of RehabNet, a camera-based system for monitoring home rehabilitation "
        "exercises. We selected REHAB24-6 as the main dataset because it combines ordinary RGB video with synchronised "
        "OptiTrack motion capture and physiotherapist labels, and prepared KIMORE and UI-PRMD skeletons for later scoring "
        f"work. A MediaPipe BlazePose pipeline was built and validated on {int(det.frames.sum()):,} frames from 130 video "
        f"streams: a person was detected in {n['det_txt']} of frames, {n['pck20']:.1f} % of keypoints were within 0.2 torso "
        f"lengths of the motion-capture joints, and joint angles on the moving joints had a mean error of {best3:.1f}° with "
        f"per-repetition range of motion correlating r = {n['rom_r']['mp3d_smooth']:.2f} with ground truth. Torso-based "
        "normalisation and One-Euro smoothing were adopted. Because much of the angle error is a constant per-joint bias, the "
        "scoring engine of Milestone 2 compares patients with references measured by the same pipeline.",
        ["pose estimation", "MediaPipe BlazePose", "rehabilitation", "motion capture", "joint angles", "REHAB24-6"])
    rp.toc()
    introduction(rp, 1, [
        ("Choose and prepare datasets", "that contain camera video, accurate ground truth and clinical labels."),
        ("Build the pose pipeline", "from camera frame to body landmarks and joint angles."),
        ("Normalise and smooth", "so position, distance and jitter do not change the measurements."),
        ("Validate against motion capture", "and decide how far single-camera measurements can be trusted."),
    ], "Everything in this milestone runs on an ordinary laptop (AMD Ryzen 3 3250U, 2 cores, no GPU); large batch jobs ran "
       "on Kaggle's cloud CPUs. The results here decide how the scoring engine of Milestone 2 is designed.")
    rp.h1("Related work")
    rp.p("Clinical movement analysis traditionally uses marker-based motion capture (e.g. OptiTrack, Vicon): reflective "
         "markers on the body are tracked by many infrared cameras with millimetre accuracy, but the system needs a dedicated "
         "lab. Depth cameras such as the Microsoft Kinect brought skeleton tracking into homes and were used to record the "
         "KIMORE dataset [5], but the device is no longer manufactured. Wearable inertial sensors are accurate for single "
         "joints but must be strapped on correctly every time.")
    rp.p("Deep-learning pose estimation removed the need for special hardware. OpenPose [8] detects the skeletons of several "
         "people in an image but needs a desktop GPU for real-time use. BlazePose [3], part of Google's MediaPipe framework [9], "
         "was designed for fitness and on-device use: it tracks one person, returns 33 landmarks (including hands and feet) in "
         "both image and 3-D 'world' coordinates, and runs in real time on phones and in web browsers. Because RehabNet must "
         "run on the patient's own device, BlazePose was chosen.")
    rp.p("For rehabilitation assessment, published datasets mostly contain skeletons recorded with Kinect or Vicon (UI-PRMD "
         "[6], KIMORE [5]) and deep-learning scoring models have been trained on them [10, 11]. Few datasets contain both "
         "ordinary video and motion capture at the same time; REHAB24-6 [4] is one of them, which allows the camera pipeline "
         "itself to be validated - the main question of this milestone.")
    rp.table(pd.DataFrame([
        ["Marker motion capture", "mm-level", "Lab only, costly, markers on body", "Ground truth (REHAB24-6, UI-PRMD)"],
        ["Depth camera (Kinect)", "cm-level", "Discontinued device", "KIMORE dataset"],
        ["Wearable sensors", "Good per joint", "Must be worn and placed correctly", "Not used"],
        ["OpenPose (2-D)", "Good", "Needs desktop GPU for real time", "Not used"],
        ["MediaPipe BlazePose", "Good, 2-D + 3-D", "One person; on-device, in the browser", "RehabNet pose engine"],
    ], columns=["Approach", "Accuracy", "Practical limits", "Use in RehabNet"]), "Ways of measuring body movement", font=9)
    rp.h1("Summary of results")
    rp.p("Milestone 1 builds the foundation of RehabNet: the data that every later module is tested on, and the "
         "pose-estimation pipeline that turns a camera image into body joints and joint angles. Everything in this "
         "milestone runs on an ordinary laptop (AMD Ryzen 3 3250U, 2 cores / 4 threads, 6 GB RAM, no GPU).")
    rp.bullets([
        ("Data:", f"three public datasets prepared in one common format: REHAB24-6 videos with OptiTrack motion-capture "
                  f"ground truth ({len(seg):,} labelled repetitions), KIMORE (355 recordings with clinician scores) and UI-PRMD "
                  f"(1,326 repetitions)."),
        ("Pose pipeline:", f"MediaPipe BlazePose found a person in {n['det_txt']} of {int(det.frames.sum()):,} "
                           f"evaluated frames across 130 video streams (front, half-profile and side views; lights on and off)."),
        ("Keypoints:", f"{n['pck20']:.1f} % of keypoints lie within 0.2 torso-lengths of the motion-capture joint "
                       f"(front view {n['pck_view'].get('front', np.nan):.1f} %, side view {n['pck_view'].get('profile', np.nan):.1f} %)."),
        ("Joint angles:", f"on the joints that move in each exercise, the mean angle error against OptiTrack is "
                          f"{front2:.1f} deg (2-D) / {front3:.1f} deg (3-D) in front view and {prof2:.1f} / {prof3:.1f} deg in side view. "
                          f"Range of motion per repetition correlates r = {n['rom_r']['mp3d_smooth']:.2f} with ground truth."),
        ("Normalisation:", "both a bounding-box and a hip-centred torso normalisation remove about 99 % of the effect of where "
                           "the person stands and how far away they are. Torso normalisation is adopted because the box also "
                           "changes size whenever an arm or leg is lifted."),
        ("Honest limit:", f"single-camera angles are about {best3:.0f} deg off motion capture on average, so RehabNet will score "
                          "movements relative to a reference measured by the same pipeline (Milestone 2), not as absolute "
                          "clinical angles."),
    ])
    rp.h2("Milestone tasks and status")
    st = pd.DataFrame([
        ["Acquire and preprocess datasets (KIMORE, UI-PRMD, REHAB24-6)", "Done", "All three in one common joint format"],
        ["MediaPipe pose pipeline, validated on benchmark videos", "Done", "130 REHAB24-6 video streams vs OptiTrack"],
        ["Bounding-box normalisation for camera angle / distance", "Done", "Box and torso methods compared; torso adopted"],
        ["Joint-angle module validated against OptiTrack", "Done", "REHAB24-6 OptiTrack used instead of UCO (see 4.1)"],
    ], columns=["Task (from proposal)", "Status", "Note"])
    rp.table(st, "Milestone 1 tasks")

    rp.h1("2. Datasets")
    rp.h2("2.1 Choice of datasets")
    rp.p("RehabNet runs on a phone or laptop camera, so the most important dataset is one with real camera video and an "
         "accurate measurement of the true body position at the same moment. REHAB24-6 (Masaryk University, SISAP 2024) "
         "provides exactly that: two RGB cameras recorded in sync with a 16-camera OptiTrack motion-capture system, plus "
         "start/end frames and a correct/incorrect label for every repetition, set by a physiotherapist. It therefore also "
         "replaces the UCO dataset that the proposal planned for OptiTrack validation: one dataset now covers pose accuracy, "
         "angle accuracy, repetition counting and correctness.")
    rp.p("KIMORE and UI-PRMD do not include usable video in their public releases, but they bring labels REHAB24-6 lacks: "
         "clinician scores (KIMORE) and many correct/incorrect repetitions of 10 standard exercises (UI-PRMD). They are used as "
         "skeleton datasets for the scoring engine in Milestone 2.")
    rp.table(t_data, "Datasets used and their role", font=8.5)
    rp.figure(F("fig_rehab24_overview.png"), "REHAB24-6 repetitions per exercise, split by the physiotherapist's correct/incorrect label.", 13)
    rp.table(ex_tab, "REHAB24-6 exercises")
    rp.h2("2.2 Preprocessing")
    rp.bullets([
        ("Common joint set:", "every source (MediaPipe 33 landmarks, OptiTrack 26 joints, Kinect 18/25 joints, Vicon 22 joints) "
                              "is mapped to the same 14 joints: neck, pelvis, both shoulders, elbows, wrists, hips, knees and ankles."),
        ("Body-centred frame:", "skeleton datasets are moved so the pelvis is at the origin and scaled by torso length, so body size "
                                "and position in the room do not affect the numbers."),
        ("Joint angles:", "the same 11 angle definitions are computed for every source."),
        ("Quality checks:", "no missing frames were found in KIMORE or UI-PRMD; REHAB24-6 repetitions flagged as having "
                            f"mocap errors ({int(seg.mocap_erroneous.sum())} of {len(seg)}) are excluded from validation."),
        ("Storage:", "data/processed/{kimore,uiprmd}.pkl and data/mp_rehab24/ (MediaPipe output for every REHAB24-6 video)."),
    ])
    rp.table(t_rp, "KIMORE and UI-PRMD after preprocessing (score = clinician score for KIMORE, quality score for UI-PRMD)", font=8)

    rp.h1("3. Pose-estimation pipeline")
    rp.figure(F("fig_pipeline.png"), "RehabNet Milestone 1 pipeline.", 16)
    rp.p("Each frame goes through MediaPipe BlazePose (Tasks API, video mode). BlazePose returns 33 landmarks twice: in image "
         "pixels (2-D) and in metres around the hips (3-D 'world' landmarks). Video mode reuses the previous frame's body "
         "position, so tracking is stable and faster than detecting from scratch.")
    rp.table(speed_both, "BlazePose model speed (single process, 960-px frames, median time per frame)")
    rp.p("The 'full' model is used for validation. On the project laptop the 'lite' model already meets the proposal's "
         "50 ms per frame target, and on a 4-core cloud CPU (Kaggle) even the 'full' model runs at over 40 frames per second. "
         "MediaPipe's GPU mode could not be used on the Kaggle server (no display for its OpenGL context), so the heavy "
         "processing for this milestone ran on Kaggle's CPUs. In the patient app, pose estimation runs in the browser "
         "(MediaPipe for JavaScript on the device's GPU), so the laptop CPU timings are a worst case; the accuracy/speed "
         "trade-off between models is measured in Milestone 5.")
    rp.figure(F(f"fig_overlay_PM_000.png"), "Same moment seen by both cameras. In the side view the raised arm is partly hidden "
              "by the head and MediaPipe places the wrist too low, a typical failure of single-camera pose estimation.", 16)

    rp.h1("4. Normalisation and smoothing")
    rp.h2("4.1 Methods")
    rp.bullets([
        ("Bounding-box normalisation (proposal):", "keypoints are expressed relative to a person box whose centre and size are "
                                                    "smoothed over time."),
        ("Torso normalisation (adopted):", "keypoints are centred on the hip midpoint and divided by the torso length "
                                           "(shoulder midpoint to hip midpoint), using the median over the last 1.5 s so it "
                                           "works live, frame by frame."),
        ("One-Euro filter:", "an adaptive low-pass filter (Casiez et al., 2012) that removes jitter when a joint is still and "
                             "follows quickly when it moves."),
    ])
    rp.h2("4.2 Results")
    if inv is not None:
        rp.p("To test distance and position, the same frames from 12 videos (2 per exercise) were shown to MediaPipe three times: "
             "as recorded, shrunk to 55 % and moved left (as if the person stood about twice as far away), and shrunk to 70 % and "
             "moved right. A perfect normalisation gives identical keypoints in all three cases.")
        rp.table(inv_t, "Change in normalised keypoints when the person is further away / off-centre (% of torso length; lower is better)")
        rp.figure(F("fig_norm_invariance.png"), "Without normalisation the keypoints move by more than one torso length; "
                  "both normalisations bring this down to a few percent (log scale).", 12.5)
        bb = inv_t.set_index("Normalisation").loc["Bounding box"]
        to = inv_t.set_index("Normalisation").loc["Torso (hip-centred)"]
        cl, cr = "Shrunk 55 %, left (%)", "Shrunk 70 %, right (%)"
        rp.p(f"Both normalisations remove almost all of the effect. At 70 % size they are equally good ({bb[cr]:.1f} % vs "
             f"{to[cr]:.1f} % of torso length). At 55 % size both leave more ({bb[cl]:.1f} % box, {to[cl]:.1f} % torso), but this "
             f"remainder is mostly MediaPipe itself becoming less precise on a small person: the scale-free joint angles also "
             f"changed by {ang_inv['far_left']:.1f} deg there (vs {ang_inv['far_right']:.1f} deg at 70 %), and the person was "
             f"detected in only {det_far['far_left']:.0f} % of those frames. In practice the patient app should ask the patient "
             "to fill a reasonable part of the frame.")
        rp.p("The two methods differ in one important way. Within a single recording the bounding box changed size by "
             f"{d['j30'].box_size_cv.mean()*100 if d['j30'] is not None else d['normalisation'].box_size_cv.mean()*100:.0f} % "
             "(coefficient of variation) because the box grows each time an arm or leg is lifted, so the same body would be "
             "scaled differently at different points of the movement. The torso length does not depend on limb position, so "
             "RehabNet uses torso normalisation and keeps the box only for cropping. Its weakness is the opposite case: when the "
             "trunk bends towards the camera its 2-D length shrinks; the 1.5 s running median limits this, and 3-D world "
             "coordinates are used for angles anyway.")
    rp.p(f"One-Euro smoothing reduced frame-to-frame jitter by {jit['red']:.0f} % (normalised units, measured at 30 fps on 10 "
         f"recordings) while leaving keypoint accuracy unchanged (mean error {jit['e_raw']:.3f} vs {jit['e_sm']:.3f} torso lengths).")

    rp.h1("5. Validation against OptiTrack")
    rp.p("Ground truth is REHAB24-6's OptiTrack skeleton, projected into each camera for 2-D comparison. Only frames inside "
         "annotated repetitions with valid motion capture are used. Videos were processed at 10 frames per second (every 3rd "
         "frame) to keep processing time on the laptop reasonable; the jitter test used full 30 fps.")
    rp.h2("5.1 Detection and keypoint accuracy")
    rp.table(det_tab, "Pose detection rate")
    rp.p(f"Detection rate by lighting: lights on {n['det_light'].get(1, np.nan):.1f} %, evening light {n['det_light'].get(0, np.nan):.1f} %.")
    rp.table(kp_view, "2-D keypoint error by camera view (error divided by torso length; PCK = share of keypoints within the threshold)")
    rp.figure(F("fig_kp_accuracy.png"), "Keypoint accuracy by joint and camera view.", 16)
    rp.table(kp_joint, "2-D keypoint error by joint (all views)")
    rp.table(kp_light, "Keypoint error by lighting")
    rp.p("Part of the remaining error is a definition difference rather than a mistake: OptiTrack places the hip joint inside "
         "the pelvis from body markers, while MediaPipe places it where the hip appears in the image. Hips therefore show a "
         "constant offset; wrists show the largest true errors because they move fastest and are often hidden.")
    rp.h2("5.2 Joint-angle accuracy")
    rp.p("For each exercise only the joints that actually move are scored (for example shoulder and elbow for arm abduction, "
         "knees and hips for squats). Three ways of computing angles are compared.")
    rp.table(ang_src, "Angle accuracy on the moving joints (all views)")
    rp.table(ang_view.round(2), "Angle MAE (deg) by camera view")
    rp.figure(F("fig_angle_mae.png"), "Angle error per exercise and camera view (3-D world angles with smoothing).", 14)
    rp.table(ang_ex.round(2), "Angle MAE (deg) by exercise")
    rp.table(ang_angle.round(2), "Accuracy per joint angle (3-D world + One-Euro)")
    rp.figure(F("fig_angle_trajectories.png"), "Angle over time: OptiTrack (navy) and MediaPipe from both cameras. "
              "Shaded bands are annotated repetitions (green correct, orange incorrect).", 16)
    rp.h2("5.3 Range of motion per repetition")
    rp.p("Clinicians mostly look at how far a joint moves in each repetition (range of motion, ROM), not at every single frame.")
    rp.table(rom_tab.round(2), "Median ROM error (deg) per repetition by view, and correlation with ground truth over all repetitions")
    rp.figure(F("fig_rom_scatter.png"), "Measured vs ground-truth range of motion, one point per repetition and joint.", 15)

    rp.h1("6. Findings")
    half3 = n["mae_view"].loc["half-profile", "mp3d_smooth"]
    worst = n["mae_angle"].mae.idxmax()
    rp.bullets([
        ("Pose detection is not the problem.", f"A person was found in {n['det_txt']} of frames, in every view and lighting condition."),
        ("Angle accuracy is moderate.", f"Average error on the moving joints is {best3:.1f} deg (3-D), best at 45 deg to the camera "
                                        f"({half3:.1f} deg) and similar in front ({front3:.1f}) and side view ({prof3:.1f}). The "
                                        f"elbow is the weakest joint ({n['mae_angle'].loc[worst, 'mae']:.0f} deg, with a constant bias of "
                                        f"{n['mae_angle'].loc[worst, 'bias']:+.0f} deg), push-ups and arm V-W the hardest exercises."),
        ("3-D beats 2-D.", f"MediaPipe's 3-D world angles were more accurate than 2-D image angles in every view "
                           f"({n['mae']['mp3d']:.1f} vs {n['mae']['mp2d']:.1f} deg overall), so the scoring engine uses 3-D angles."),
        ("Range of motion is usable for feedback.", f"Per-repetition ROM correlates r = {n['rom_r']['mp3d_smooth']:.2f} with OptiTrack "
                                                    f"(median error {n['rom_med']['mp3d_smooth']:.0f} deg), enough for 'raise your arm higher' "
                                                    "feedback but not for clinical goniometry."),
        ("Consequence for Milestone 2.", "Much of the error is a constant bias per joint. Comparing a patient with a reference "
                                         "demonstration measured by the same pipeline cancels a constant bias, which is why the "
                                         "scoring engine will be relative (DTW/NCC against a reference) rather than absolute."),
        ("Torso normalisation, not bounding box.", "Equal on distance/position, but the box changes size when limbs move."),
    ])
    rp.h2("Limitations")
    rp.bullets([
        "REHAB24-6 has 10 healthy adults in one room; elderly or impaired patients and home environments are not represented.",
        "Videos were evaluated at 10 fps to fit processing time (jitter at 30 fps on 6 recordings); live use runs at camera rate.",
        f"Single-camera angles are about {best3:.0f} deg off motion capture on average, so absolute angles are not clinical-grade.",
        "OptiTrack's skeleton is itself a model fitted to markers, so a few degrees of disagreement are definition differences.",
    ])
    rp.h1("Conclusion")
    rp.p(f"Milestone 1 delivered the measurement foundation of RehabNet. The datasets are prepared in one format, the camera "
         f"pipeline finds the patient in practically every frame, and its joint angles follow motion capture closely enough "
         f"(about {best3:.0f}° mean error, ROM r = {n['rom_r']['mp3d_smooth']:.2f}) to judge whether a movement was big enough, "
         "smooth and in the right shape. They are not accurate enough to replace a goniometer, which shaped the main design "
         "decision for the rest of the project: score each patient relative to correct repetitions measured by the same camera "
         "pipeline, so the constant per-joint errors cancel out.")
    rp.callout(["3-D world angles are used everywhere (more accurate than 2-D in every view).",
                "Torso normalisation and One-Euro smoothing are part of the live pipeline.",
                "The patient app must check framing (whole body visible, not too far away).",
                "Scoring is relative to reference repetitions, not to absolute angle targets."],
               title="Decisions carried forward")
    rp.h1("7. Next: Milestone 2 (August - September 2026)")
    rp.bullets([
        "DTW alignment of a patient repetition to a reference demonstration (dtaidistance).",
        "NCC similarity score between joint-angle curves.",
        "Angle-threshold repetition counter, validated against REHAB24-6 start/end frames.",
        "Multi-factor quality score (posture accuracy, range of motion, stability, tempo consistency), checked against "
        "REHAB24-6 correct/incorrect labels, UI-PRMD labels and KIMORE clinician scores.",
    ])
    rp.references(refs("cieza", "argent", "blazepose", "rehab24", "kimore", "uiprmd", "casiez", "openpose", "mediapipe",
                       "liao", "rehabpile"))
    rp.h1("Appendix: code")
    rp.code("rehabnet/core/pose.py         MediaPipe wrapper (video mode, 2-D + 3-D landmarks)\n"
            "rehabnet/core/normalize.py    bounding-box and torso normalisation, One-Euro filter\n"
            "rehabnet/core/angles.py       common joint set and joint-angle definitions\n"
            "scripts/m1_preprocess_rehabpile.py   KIMORE / UI-PRMD preprocessing\n"
            "scripts/m1_extract_rehab24.py        MediaPipe over REHAB24-6 videos\n"
            "scripts/m1_validate.py               validation vs OptiTrack\n"
            "scripts/m1_normalization_test.py     distance / position invariance test")
    rp.save(os.path.join(OUT, "M1_Report.docx"))

    # ------------------------------------------------------------------ slides
    dk = Deck(TITLE, SUB, PERIOD)
    dk.agenda([("Problem and plan", "why RehabNet, five milestones"), ("Datasets", "REHAB24-6, KIMORE, UI-PRMD"),
               ("Pose pipeline", "MediaPipe BlazePose, speed"), ("Normalisation and smoothing", "distance, position, jitter"),
               ("Validation vs motion capture", "keypoints, angles, range of motion"), ("Findings and next steps", "decisions for Milestone 2")],
              say="Six parts: motivation, data, the pipeline, normalisation, validation, and what we decided from it.")
    deck_intro(dk, 1, [
        ("Choose and prepare datasets", "camera video + accurate ground truth + clinical labels"),
        ("Build the pose pipeline", "camera frame -> 33 landmarks -> joint angles"),
        ("Normalise and smooth", "remove the effect of position, distance and jitter"),
        ("Validate against motion capture", "how far can single-camera measurements be trusted?"),
    ])
    dk.stats("Milestone 1 in numbers", [
        (f"{int(det.frames.sum()/1000)}k", "frames validated", "130 video streams, 3 camera views, lights on/off"),
        (">99.9 %" if n["det"] > 99.9 else f"{n['det']:.1f} %", "person detected", "in every view and lighting"),
        (f"{best3:.0f}°", "mean angle error", "moving joints, 3-D angles vs OptiTrack"),
        (f"r = {n['rom_r']['mp3d_smooth']:.2f}", "range of motion", "per repetition vs ground truth"),
        (f"-{jit['red']:.0f} %", "jitter", "One-Euro filter, accuracy unchanged"),
    ], kicker="Summary", takeaway="The camera pipeline is reliable enough for exercise feedback; absolute angles are not clinical-grade.",
        say="Headline results. Detection is essentially perfect; angle error is about 15 degrees, so we will score relative to references.")
    dk.table("Milestone 1 tasks", st, kicker="Status", font=15, col_w=[6.2, 1.9, 3.9],
             takeaway="All four tasks from the proposal are done; UCO was replaced by the richer REHAB24-6 dataset.")

    dk.section(1, "Datasets", "Video with motion-capture ground truth, plus two skeleton datasets with clinical labels")
    dk.cards("Ways to measure body movement", [
        ("Motion capture", ["mm accuracy", "lab only, markers on the body", "-> our ground truth"]),
        ("Depth camera (Kinect)", ["cm accuracy", "device discontinued", "-> KIMORE dataset"]),
        ("OpenPose", ["good 2-D accuracy", "needs a desktop GPU"]),
        ("MediaPipe BlazePose", ["33 landmarks, 2-D + 3-D", "real time on phones and in browsers", "-> RehabNet's pose engine"]),
    ], kicker="Related work", cols=4, icons=False,
        takeaway="BlazePose is the only option that runs in real time on the patient's own phone or laptop.")
    dk.cards("Why REHAB24-6 as the main dataset", [
        ("Camera video", "two RGB cameras: the same input RehabNet gets from a phone"),
        ("True body position", "16-camera OptiTrack recorded at the same moment"),
        ("Labels", f"{len(seg):,} repetitions with start/end frame and physiotherapist's correct/incorrect label"),
        ("Real-world variation", "front, half-profile and side views; lights on and off"),
        ("Replaces UCO", "one dataset covers pose, angle, counting and correctness validation"),
        ("KIMORE + UI-PRMD", "skeleton-only; add clinician scores and 10 more exercises (Milestone 2)"),
    ], kicker="Datasets", takeaway="One dataset lets us test every step from camera image to correctness label.")
    dk.table("Datasets and their roles", t_data[["Dataset", "Size", "Labels", "Role in RehabNet"]], kicker="Datasets", font=13,
             col_w=[2.2, 3.2, 3.3, 3.4])
    dk.image("REHAB24-6 at a glance", F("fig_rehab24_overview.png"), kicker="Datasets",
             notes=["6 exercises, 10 people", "65 recordings × 2 cameras", "about half correct, half incorrect", "30 fps, 1080p video",
                    f"{int(seg.mocap_erroneous.sum())} reps with mocap errors excluded"],
             say="Balanced correct and incorrect repetitions make it suitable for testing the scoring later.")
    dk.flow("Preprocessing into one common format", [
        ("Map joints", ["MediaPipe 33, OptiTrack 26,", "Kinect 25, Vicon 22", "-> 14 common joints"]),
        ("Body-centre", ["pelvis at origin", "scale by torso length"]),
        ("Joint angles", ["same 11 angle definitions", "for every source"]),
        ("Quality checks", ["missing frames", "mocap-error flags"]),
        ("Store", ["data/processed/*.pkl", "data/mp_rehab24/*.npz"]),
    ], kicker="Datasets", takeaway="Every later module reads the same joints and angles, whatever sensor recorded them.")

    dk.section(2, "Pose-estimation pipeline", "From a camera frame to joint angles, frame by frame")
    dk.image("Pose-estimation pipeline", F("fig_pipeline.png"), kicker="Pipeline",
             takeaway="Each frame gives 33 landmarks in pixels (2-D) and in metres around the hips (3-D).")
    dk.table("Pose model speed", speed_both, kicker="Pipeline", font=14,
             notes=["Laptop: Ryzen 3 3250U, no GPU", "'lite' meets < 50 ms even on the laptop",
                    "MediaPipe GPU mode failed on Kaggle (no display)", "In the app, pose runs in the browser on the device GPU"],
             takeaway="Real time is possible even on a low-end laptop CPU with the lite model.")
    dk.image("Same moment, two cameras", F("fig_overlay_PM_000.png"), kicker="Pipeline",
             caption="Side view: the raised arm is hidden behind the head and MediaPipe places the wrist too low.",
             takeaway="Hidden limbs are the typical failure of a single camera - camera placement matters.")

    dk.section(3, "Normalisation and smoothing", "Making measurements independent of where the patient stands")
    dk.cards("Three techniques", [
        ("Bounding box (proposal)", ["keypoints relative to a smoothed person box", "box grows when a limb is lifted"]),
        ("Torso (adopted)", ["centre on the hips, divide by torso length", "1.5 s running median, works live"]),
        ("One-Euro filter", ["adaptive low-pass filter (Casiez 2012)", "removes jitter, follows fast movement"]),
    ], kicker="Normalisation")
    if inv is not None:
        dk.image("Normalisation: distance and position", F("fig_norm_invariance.png"), kicker="Normalisation",
                 notes=["Same frames, person shrunk and moved", "Raw pixels move > 1 torso length",
                        "Box and torso both remove ~99 %", "At 55 % size MediaPipe itself gets less precise",
                        "Torso chosen: box grows when a limb is lifted", f"One-Euro: jitter -{jit['red']:.0f} %, accuracy unchanged"],
                 takeaway="Torso normalisation: as good as the box on distance, and stable while limbs move.")

    dk.section(4, "Validation against motion capture", "How close is the camera to OptiTrack?")
    dk.table("Detection and keypoint accuracy", kp_view, kicker="Validation", font=15,
             notes=[f"Detected in {n['det_txt']} of frames", "PCK@0.2 = keypoint within 0.2 torso of mocap",
                    "Side view is hardest: one side hidden"],
             takeaway="Keypoints are found reliably; the side view loses accuracy because one side of the body is hidden.")
    dk.image("Keypoint accuracy by joint", F("fig_kp_accuracy.png"), kicker="Validation",
             takeaway="Wrists are least accurate (fast and often hidden); hips show a constant definition offset.")
    dk.table("Joint-angle error vs OptiTrack (degrees)", ang_view.round(1), kicker="Validation", font=15,
             notes=["Moving joints only", "Best at 45° to the camera", "3-D beats 2-D in every view", "Elbow is the weakest joint"],
             takeaway=f"3-D world angles: {best3:.1f}° mean error - used by the scoring engine.")
    dk.image("Angle error by exercise", F("fig_angle_mae.png"), kicker="Validation",
             takeaway="Push-ups and arm V-W are hardest; squats and leg exercises are measured best.")
    dk.image("Angles over time", F("fig_angle_trajectories.png"), kicker="Validation",
             takeaway="The camera follows the shape of every repetition, with a mostly constant offset.")
    dk.image("Range of motion per repetition", F("fig_rom_scatter.png"), kicker="Validation",
             notes=[f"r = {n['rom_r']['mp3d_smooth']:.2f} (3-D) vs mocap", f"median error {n['rom_med']['mp3d_smooth']:.1f}°",
                    "Supports 'raise higher' feedback", "Not a goniometer replacement"],
             takeaway="Range of motion per repetition is reliable enough for feedback.")

    dk.cards("Findings and decisions", [
        ("Detection is not the problem", f"person found in {n['det_txt']} of frames, all views and lighting"),
        ("Angles: ~15° error", "fine for feedback, not for clinical goniometry"),
        ("3-D angles", "more accurate than 2-D in every view -> used by the scoring engine"),
        ("Relative scoring", "compare with references measured by the same pipeline: constant bias cancels"),
        ("Torso normalisation", "box changes size when limbs move"),
        ("Limitations", "healthy adults, one lab room, 10 fps evaluation"),
    ], kicker="Conclusions", takeaway="Milestone 2 scores patients against correct repetitions measured by the same camera pipeline.")
    dk.flow("Next: Milestone 2 (Aug - Sep 2026)", [
        ("Repetition counter", "angle-based, validated on REHAB24-6 start/end frames"),
        ("DTW alignment", "match each repetition to reference demonstrations"),
        ("NCC similarity", "compare the shape of joint-angle curves"),
        ("Quality score", "posture, ROM, stability, tempo vs physiotherapist labels"),
    ], kicker="Plan")
    dk.closing(lines=["Questions and discussion", "Code: rehabnet/core/{pose, normalize, angles}.py  ·  scripts/m1_*.py"])
    dk.save(os.path.join(OUT, "M1_Presentation.pptx"))
    print("saved", OUT)


if __name__ == "__main__":
    build()
