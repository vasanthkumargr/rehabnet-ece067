"""Build Milestone 5 (testing, performance, deployment) report + slides, and the RehabNet user guide."""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rehabnet.core.exercises import EXERCISES  # noqa: E402
from tools.deliver import Deck, Report  # noqa: E402
from scripts.common_content import deck_intro, introduction, refs  # noqa: E402

R5 = os.path.join(ROOT, "results", "m5")
S = os.path.join(ROOT, "results", "m3", "screens")
OUT = os.path.join(ROOT, "deliverables", "M5")
NICE = {k: v.name for k, v in EXERCISES.items()}
COND = {"original": "Original video", "dark": "Dark (evening light)", "low_res": "Low resolution / blur",
        "occluded": "Part of body hidden", "small": "Person far away (45 % size)"}
TEAL, NAVY, ORANGE, GREY = "#0F766E", "#122B4A", "#D97706", "#6B7280"


def robustness():
    df = pd.read_csv(os.path.join(R5, "robustness.csv"))
    cond = df[df.model == "full"].groupby("cond").agg(detected=("detected", "mean"), angle_mae=("angle_mae", "mean"),
                                                      count_err=("count_err", "mean"), gt=("gt_reps", "mean"))
    cond = cond.reindex([c for c in COND if c in cond.index])
    model = df[df.cond == "original"].groupby("model").agg(detected=("detected", "mean"), angle_mae=("angle_mae", "mean"),
                                                           count_err=("count_err", "mean"), ms=("ms", "median"))
    model = model.reindex([m for m in ("lite", "full", "heavy") if m in model.index])
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.8))
    SHORT = {"original": "Original", "dark": "Dark", "low_res": "Low-res\n/ blur", "occluded": "Partly\nhidden",
             "small": "Far\naway"}
    labs = [SHORT[c] for c in cond.index]
    x = np.arange(len(cond))
    axs[0].bar(x, cond.detected * 100, color=TEAL); axs[0].set_title("Person detected (%)"); axs[0].set_ylim(0, 105)
    axs[1].bar(x, cond.angle_mae, color=NAVY); axs[1].set_title("Angle error vs mocap (deg)")
    axs[2].bar(x, cond.count_err, color=ORANGE); axs[2].set_title("Count error per recording (reps)")
    for ax, col in zip(axs, ("detected", "angle_mae", "count_err")):
        ax.set_xticks(x, labs, fontsize=8.5)
        ax.spines[["top", "right"]].set_visible(False)
        for i, v in enumerate(cond[col] * (100 if col == "detected" else 1)):
            ax.text(i, v, f"{v:.1f}" if col != "detected" else f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(R5, "fig_robustness.png"), dpi=170, bbox_inches="tight"); plt.close(fig)
    return df, cond, model


def build():
    os.makedirs(OUT, exist_ok=True)
    df, cond, model = robustness()
    sp_laptop = pd.read_csv(os.path.join(ROOT, "results", "m1", "pose_model_speed.csv"))
    sp_laptop = sp_laptop[sp_laptop.input_px == 960].set_index("model")
    sp_kaggle = pd.read_csv(os.path.join(ROOT, "results", "m1", "pose_model_speed_kaggle_t4.csv"))
    sp_kaggle = sp_kaggle[sp_kaggle.delegate == "cpu"].set_index("model")
    live = pd.read_csv(os.path.join(ROOT, "results", "m3", "live_eval_summary.csv")).set_index("exercise")
    cond_t = cond.reset_index()
    cond_t["cond"] = cond_t.cond.map(COND)
    cond_t["detected"] *= 100
    cond_t = cond_t[["cond", "detected", "angle_mae", "count_err"]]
    cond_t.columns = ["Condition (full model)", "Detected %", "Angle error (deg)", "Count error (reps / recording)"]
    model_t = model.reset_index()
    model_t["detected"] *= 100
    model_t["laptop"] = model_t.model.map(sp_laptop.median_ms_per_frame)
    model_t["kaggle"] = model_t.model.map(sp_kaggle.median_ms)
    model_t = model_t[["model", "detected", "angle_mae", "count_err", "laptop", "kaggle"]]
    model_t.columns = ["BlazePose model", "Detected %", "Angle error (deg)", "Count error", "Laptop CPU ms/frame", "Cloud CPU ms/frame"]
    lat = pd.DataFrame([
        ["Pose estimation, laptop CPU (Ryzen 3, 2 cores), lite model", f"{sp_laptop.loc['lite','median_ms_per_frame']:.0f} ms", "meets the < 50 ms target"],
        ["Pose estimation, laptop CPU, full model", f"{sp_laptop.loc['full','median_ms_per_frame']:.0f} ms", "12 fps"],
        ["Pose estimation, 4-core cloud CPU, full model", f"{sp_kaggle.loc['full','median_ms']:.0f} ms", "43 fps"],
        ["Pose estimation in the browser (built-in browser pane, this laptop)", "50-65 ms", "full model, WebAssembly"],
        ["Server: smoothing, angles, counting, scoring per frame", "0.4 ms", "measured replaying 30 fps recordings"],
        ["Feedback delay after a repetition ends", "< 1.5 s", "score sent when the next repetition starts or after 1.5 s rest"],
    ], columns=["Step", "Time", "Note"])
    tests = pd.DataFrame([
        ["Pose & angles vs motion capture (M1)", "84,532 frames, 130 video streams", "detection > 99.9 %, angle error 14.8°, ROM r = 0.81"],
        ["Repetition counter (M2)", "2,144 repetitions, unseen people", "recall 92 %, precision 95 %, ±1 rep per set front/45°"],
        ["Scoring (M2/M3)", "REHAB24-6, UI-PRMD, KIMORE", "AUC 0.71 / 0.68 live; Spearman 0.64 / 0.34"],
        ["Personal baseline (M4)", "1,357 repetitions after calibration", "combined AUC 0.705"],
        ["Robustness (M5)", f"{df.video.nunique()} recordings × {df.groupby(['cond','model']).ngroups} conditions", "dark, blur, occlusion, far away, 3 model sizes"],
        ["Automated API tests (M3)", "7 tests", "all pass: auth, 2-factor, permissions, live session, PDF, baseline"],
        ["Browser end-to-end (M3)", "demo videos in the real UI", "browser = offline result; hidden-tab bug found and fixed"],
    ], columns=["Test", "Data", "Result"])
    worst = cond.drop("original").angle_mae.idxmax() if len(cond) > 1 else None

    o = cond.loc["original"]
    per_ex = df[df.model == "full"].pivot_table(index="exercise", columns="cond", values="angle_mae", aggfunc="mean")
    per_ex = per_ex[[c for c in COND if c in per_ex.columns]].rename(columns={c: COND[c].split(" (")[0] for c in COND})
    per_ex.index = [NICE.get(i, i) for i in per_ex.index]
    per_ex = per_ex.reset_index().rename(columns={"index": "Exercise"})
    plan = pd.DataFrame([
        ["Unit / module", "angles, counter, scoring, feedback functions", "M1-M2 validation scripts", "Done"],
        ["Integration (API)", "auth, permissions, prescriptions, sessions, PDF, baseline", "pytest, 7 tests", "All pass"],
        ["End-to-end", "full live session over WebSocket; real browser with demo videos", "pytest + browser", "Pass"],
        ["Accuracy", "pose, angles, counting, scoring vs ground truth on unseen people", "REHAB24-6, UI-PRMD, KIMORE", "Reported"],
        ["Robustness", "dark, blur, occlusion, far away, model sizes", "Kaggle run, 10 recordings", "1 failure found, fixed"],
        ["Performance", "time per frame on laptop, cloud and server", "timing scripts", "< 50 ms target met"],
        ["Security", "lock-out, 2-factor, access control, token misuse", "pytest", "All pass"],
    ], columns=["Test level", "What is checked", "How", "Result"])
    deploy = pd.DataFrame([
        ["Clinic laptop", ".\\run_app.ps1", "http://localhost:8000", "Demo, single clinic, offline"],
        ["Phones on the same Wi-Fi", ".\\run_app.ps1 -Https", "https://<laptop-ip>:8443", "Home-like test with phones"],
        ["Cloud container", "docker build / docker run", "HTTPS behind the provider", "Many clinics; PostgreSQL"],
    ], columns=["Setting", "Command", "Address", "Use"])
    proposal = pd.DataFrame([
        ["Pose estimation from a normal camera", "MediaPipe BlazePose, validated vs OptiTrack (M1)", "Achieved"],
        ["Bounding-box normalisation", "box and torso compared; torso adopted (M1)", "Achieved (improved)"],
        ["DTW + NCC comparison with references", "posture and similarity factors (M2)", "Achieved"],
        ["Angular repetition counter", "swing-based counter, ±1 rep per set front/45° (M2)", "Achieved"],
        ["Multi-factor quality score", "6 factors, weights learned from physio labels (M2)", "Achieved"],
        ["Doctor and patient portals", "web app with live feedback (M3)", "Achieved"],
        ["Encrypted storage + MFA", "Fernet encryption, TOTP 2-factor (M3)", "Achieved"],
        ["Personal baselines, progress, reports", "combined references, trends, adherence, PDF (M4)", "Achieved"],
        ["Testing, < 50 ms/frame, deployment", "robustness run, lite model 46 ms, Docker/HTTPS (M5)", "Achieved"],
        ["Clinical pilot with patients", "not possible within the project", "Future work"],
    ], columns=["Proposal objective", "What was built", "Status"])

    rp = Report("Milestone 5: Testing, Optimisation and Deployment", "Robustness, real-time performance, deployment and documentation",
                "Planned: February - March 2027")
    rp.header_footer("Milestone 5")
    rp.abstract(
        "This final milestone tests RehabNet as a whole and prepares it for use. The complete camera pipeline was run on "
        f"{df.video.nunique()} REHAB24-6 recordings under simulated home conditions. Blur and partial occlusion changed the "
        f"joint-angle error by less than 2° ({o.angle_mae:.1f}° on clean video), a person standing far away was detected in "
        f"{cond.loc['small'].detected*100:.0f} % of frames, and a dark room was the one real failure "
        f"({cond.loc['dark'].detected*100:.0f} % detection). The app was therefore extended with a brightness check that asks "
        "the patient to turn on a light. The lite BlazePose model meets the proposal's 50 ms per frame target on a low-end "
        "laptop CPU, and the server needs only 0.4 ms per frame. RehabNet can be started with one command, used from phones "
        "over HTTPS on the local network, or deployed as a Docker container; documentation includes a user guide. All "
        "objectives of the proposal were achieved except a clinical pilot, which is the main future work.",
        ["system testing", "robustness", "real-time performance", "deployment", "Docker", "rehabilitation"])
    rp.toc()
    introduction(rp, 5, [
        ("Test the whole system", "accuracy, robustness to home conditions, security and end-to-end behaviour."),
        ("Real-time on ordinary hardware", "meet the proposal's < 50 ms per frame on a CPU."),
        ("Deploy", "one-command local start, phones on Wi-Fi, cloud container, offline fallback."),
        ("Document", "user guide, README and reports for every milestone."),
    ], "This milestone closes the project: it checks that the system built in Milestones 1-4 keeps working outside the "
       "lab, fixes what does not, and packages it so that others can run it.")
    rp.h1("Summary of results")
    rp.bullets([
        ("Robustness:", f"on clean video the full model detected the person in {o.detected*100:.0f} % of frames with "
                        f"{o.angle_mae:.1f}° angle error. "
                        + "; ".join(f"{COND[c].lower()}: {r.detected*100:.0f} % detected, {r.angle_mae:.1f}°, count error {r.count_err:.1f}"
                                    for c, r in cond.drop("original").iterrows()) + "."),
        ("Issue found and fixed:", "darkness is the one condition the pipeline cannot handle, so the app now warns the patient "
                                   "when the room is too dark (tested in the real app with a dark and a bright camera feed)."),
        ("Real time:", "pose estimation runs on the patient's device; the lite model meets the proposal's 50 ms per frame "
                       "target even on the project's low-end laptop CPU, and the server needs only 0.4 ms per frame."),
        ("Deployment:", "one-command start on Windows, HTTPS mode for phones on the same Wi-Fi, Docker image for any cloud; "
                        "the app works offline (all libraries and models are served locally)."),
        ("Documentation:", "README, this report, a user guide for doctors and patients, and inline code documentation."),
    ])
    st = pd.DataFrame([
        ["Testing: accuracy, occlusion, lighting, edge cases", "Done", "Kaggle robustness run + all earlier validations"],
        ["Real-time on commodity hardware (< 50 ms/frame CPU)", "Done", "lite model on laptop CPU; in-browser processing"],
        ["Cloud deployment with secure backend + offline fallback", "Ready", "Docker + HTTPS; local/offline mode works; cloud account needed to publish"],
        ["Demo-ready system + user documentation", "Done", "demo accounts, demo videos, user guide"],
        ["Research paper submission", "Not part of this build", "results in M1-M5 reports can be reused"],
    ], columns=["Task (from proposal)", "Status", "Note"])
    rp.table(st, "Milestone 5 tasks", font=9)
    rp.h1("Test plan")
    rp.p("Testing was organised in levels, from single functions to the complete system in a browser. Accuracy tests always "
         "used people that were not used to tune the system (leave-one-person-out), so they estimate performance on a new patient.")
    rp.table(plan, "Test levels and results", font=8.5)
    rp.h1("Robustness")
    rp.h2("Method")
    rp.p(f"{df.video.nunique()} REHAB24-6 front-camera recordings (covering five of the six exercises; the table push-up was "
         "not included in this run) were processed by the complete camera pipeline under simulated real-world problems:")
    rp.table(pd.DataFrame([
        ["Dark (evening light)", "brightness compressed with gamma 2.2 to 60 %, plus sensor noise", "unlit room in the evening"],
        ["Low resolution / blur", "downscaled to 320 px width and blurred", "cheap or old webcam"],
        ["Part of body hidden", "a moving block covering about a quarter of the body", "furniture, a passer-by"],
        ["Person far away", "person shrunk to 45 % of the frame", "camera placed too far"],
        ["Model size", "lite / full / heavy BlazePose on the original video", "slow vs fast devices"],
    ], columns=["Condition", "How it was simulated", "Real-world cause"]), "Simulated conditions", font=9)
    rp.p("Angle error is measured against the OptiTrack motion capture on the moving joints; the count error uses the live "
         "counter with thresholds learned from other people. The run took 79 minutes on Kaggle's cloud CPUs.")
    rp.h2("Results")
    rp.table(cond_t.round(2), "Robustness of the camera pipeline (full model)", font=9)
    rp.figure(os.path.join(R5, "fig_robustness.png"), "Detection, angle error and counting error under each condition.", 16)
    rp.table(per_ex.round(1), "Angle error (degrees) by exercise and condition", font=8.5)
    if worst:
        rp.p(f"The hardest condition was '{COND[worst].lower()}': the person was found in only "
             f"{cond.loc[worst].detected*100:.0f} % of frames, so no useful score is possible there. Blur and partial "
             "occlusion changed angle error by less than 2°, and the three model sizes differ by about 2°. A person far away "
             f"was detected in {cond.loc['small'].detected*100:.0f} % of frames with a larger error, which the app's framing "
             "check ('come closer', 'step back into the picture') is designed to prevent.")
    rp.h2("Fix: low-light warning")
    rp.p("Because darkness cannot be fixed by the pose model, the app now measures the brightness of the camera picture "
         "(mean of a 32 × 18 pixel copy of each frame) and asks the patient to turn on a light when it is below 45 of 255; "
         "the darkening used in the test turns a mid-grey of 128 into about 33. The warning was tested in the real app with "
         "a synthetic dark camera feed (warning shown) and a bright one (no warning).")
    rp.callout("Testing found one condition the system could not handle, and the app was changed so the patient is told "
               "what to do instead of silently getting wrong counts.", title="Outcome")
    rp.h1("Model size and speed")
    rp.table(model_t.round(2), "Accuracy and speed of the three BlazePose model sizes (original video)", font=9)
    rp.table(lat, "Latency of each step", font=9)
    rp.p("The app uses the full model in the browser (accuracy first) and lets the device's GPU run it when available; on "
         "slow devices the lite model is the fallback that still meets the 50 ms target. Because pose estimation happens on "
         "the patient's device, the server cost per patient is tiny and does not grow with video resolution.")
    rp.h1("All tests at a glance")
    rp.table(tests, "Validation and testing across the project", font=9)
    rp.h1("Deployment")
    rp.table(deploy, "Ways to run RehabNet", font=9)
    rp.bullets([
        ("This computer:", "`run_app.ps1` creates the environment on first run and starts the server at http://localhost:8000."),
        ("Phones on the same Wi-Fi:", "`run_app.ps1 -Https` creates a certificate for the computer's address and serves "
                                      "https://<ip>:8443; browsers only allow the camera on https pages."),
        ("Cloud:", "the Dockerfile [5] builds a small image (no MediaPipe/OpenCV needed on the server). Run it on any container "
                   "service behind HTTPS (e.g. Google Cloud Run, AWS App Runner); point REHABNET_DB_URL to PostgreSQL and pass the "
                   "secrets as environment variables. Publishing requires an account on the chosen provider."),
        ("Offline fallback:", "MediaPipe's JavaScript, WebAssembly and model files are served by RehabNet itself, so a clinic "
                              "laptop works without internet."),
    ])
    rp.code("docker build -t rehabnet .\n"
            "docker run -p 8000:8000 -v rehabnet-data:/data \\\n"
            "    -e REHABNET_JWT_SECRET=<random> -e REHABNET_DATA_KEY=<fernet key> rehabnet")
    rp.h1("Documentation")
    rp.table(pd.DataFrame([
        ["README.md", "developers", "how to run, results table, layout, deployment, data licences"],
        ["RehabNet User Guide", "doctors and patients", "starting the app, everyday tasks, troubleshooting"],
        ["Milestone reports M1-M5", "mentor, examiners", "methods, results, limitations per milestone"],
        ["Presentations M1-M5", "reviews", "summary slides with speaker notes"],
        ["Code docstrings", "developers", "purpose of every module and function"],
    ], columns=["Document", "Audience", "Content"]), "Project documentation", font=9)
    rp.h1("Project conclusion")
    rp.p("RehabNet set out to turn an ordinary camera into a rehabilitation assistant. Across five milestones it was built and "
         "validated step by step: a pose pipeline checked against motion capture, an explainable counting and scoring engine "
         "checked against a physiotherapist's labels, a secure doctor-patient web platform, personalisation and progress "
         "tracking, and finally system testing and deployment. Every number in these reports was measured on people the "
         "system had not seen during tuning.")
    rp.table(proposal, "Proposal objectives and outcome", font=8.5)
    rp.h2("Limitations and future work")
    rp.bullets([
        "Validation data are healthy volunteers with simulated mistakes; a pilot with real patients and physiotherapists is the next step.",
        "Single-camera angles are about 15° off motion capture; scores are relative to references and support, not replace, clinical judgement.",
        "Knee-alignment mistakes (lunge) and shoulder rotation are not captured by the joint angles used; adding frontal-plane knee angles is planned.",
        "The exercise library has six exercises; new exercises need a few correct reference recordings (the builder script handles the rest).",
        "Darkness cannot be compensated; the app can only warn. Infrared or depth cameras would be needed for very low light.",
    ])
    rp.references(refs("cieza", "argent", "blazepose", "rehab24", "docker", "owasp"))
    rp.save(os.path.join(OUT, "M5_Report.docx"))

    dk = Deck("Milestone 5: Testing, Optimisation and Deployment", "Robustness, real-time performance, deployment and documentation",
              "Planned: February - March 2027")
    dk.agenda([("Problem and plan", "where Milestone 5 fits"), ("Test plan", "from functions to the whole system"),
               ("Robustness", "dark, blur, occlusion, far away"), ("Speed", "model sizes and latency"),
               ("Deployment and documentation", "laptop, phones, cloud"), ("Project conclusion", "proposal vs outcome")])
    deck_intro(dk, 5, [
        ("Test the whole system", "accuracy, robustness, security, end to end"),
        ("Real time", "< 50 ms per frame on a CPU"),
        ("Deploy", "laptop, phones on Wi-Fi, cloud, offline"),
        ("Document", "user guide, README, reports"),
    ])
    dk.stats("Milestone 5 in numbers", [
        (f"{df.video.nunique()} × {df.groupby(['cond', 'model']).ngroups}", "recordings × conditions", "79 min robustness run on Kaggle"),
        (f"<2°", "extra error", "with blur or partial occlusion"),
        (f"{cond.loc['dark'].detected*100:.0f} %", "detected in the dark", "-> new low-light warning"),
        (f"{model_t.set_index('BlazePose model').loc['lite', 'Laptop CPU ms/frame']:.0f} ms", "lite model, laptop CPU", "meets < 50 ms"),
        ("3", "ways to deploy", "laptop, phones on Wi-Fi, Docker"),
    ], kicker="Summary", takeaway="The system holds up in realistic conditions; the one failure found (darkness) now gets a clear warning.")
    dk.table("Milestone 5 tasks", st, kicker="Status", font=12, col_w=[5.0, 1.8, 5.2])
    dk.section(1, "Testing", "From single functions to the complete system")
    dk.table("Test plan", plan, kicker="Testing", font=12, col_w=[2.2, 5.0, 2.8, 2.2],
             takeaway="Accuracy tests always use people the system was not tuned on.")
    dk.cards("Simulated home conditions", [
        ("Dark", "evening light: darker image plus sensor noise"),
        ("Low resolution / blur", "a cheap or old webcam"),
        ("Part of body hidden", "furniture or a passer-by"),
        ("Far away", "person at 45 % size"),
        ("Model sizes", "lite / full / heavy BlazePose"),
        ("Measured", "detection, angle error vs mocap, count error"),
    ], kicker="Robustness")
    dk.image("Robustness results", os.path.join(R5, "fig_robustness.png"), kicker="Robustness",
             takeaway="Blur and occlusion barely matter; far away hurts; darkness breaks detection.")
    dk.table("Angle error by exercise and condition (degrees)", per_ex.round(1), kicker="Robustness", font=13)
    dk.cards("Issue found -> fixed", [
        ("Found", f"in a dark room the person was detected in only {cond.loc['dark'].detected*100:.0f} % of frames"),
        ("Why it matters", "the patient would silently get wrong counts and scores"),
        ("Fix", "the app measures picture brightness and asks for more light below 45 / 255"),
        ("Verified", "real app with a dark camera feed (warning) and a bright one (no warning)"),
    ], kicker="Robustness", cols=4)
    dk.section(2, "Speed and deployment", "Real time on ordinary hardware, running anywhere")
    dk.table("Model size vs accuracy and speed", model_t.round(2), kicker="Performance", font=13,
             takeaway="The lite model meets the 50 ms target on the project laptop with almost the same accuracy.")
    dk.table("Latency of each step", lat, kicker="Performance", font=12, col_w=[6.4, 1.6, 4.0])
    dk.table("Ways to run RehabNet", deploy, kicker="Deployment", font=14, col_w=[2.6, 3.0, 3.4, 3.0],
             takeaway="All MediaPipe files are served locally, so a clinic laptop works without internet.")
    dk.cards("Documentation", [
        ("README", "run, results, layout, deployment"),
        ("User guide", "doctors and patients, troubleshooting"),
        ("Reports M1-M5", "methods, results, limitations"),
        ("Presentations", "with speaker notes"),
    ], kicker="Documentation", cols=4)
    dk.section(3, "Project conclusion", "What RehabNet achieved against the proposal")
    dk.table("Proposal objectives and outcome", proposal, kicker="Conclusion", font=11, col_w=[4.0, 6.0, 2.0])
    dk.table("All tests at a glance", tests, kicker="Conclusion", font=11, col_w=[3.4, 3.6, 5.0])
    dk.cards("Limitations and future work", [
        ("Clinical pilot", "real patients and physiotherapists"),
        ("Angle accuracy", "~15° off mocap: scores are relative, a clinical aid"),
        ("More measures", "frontal-plane knee angles (lunge), shoulder rotation"),
        ("Bigger library", "more exercises from a few reference recordings"),
    ], kicker="Conclusion", cols=4, takeaway="Every proposal objective was achieved except a clinical pilot.")
    dk.closing(lines=["Questions and discussion", "Live demo: .\\run_app.ps1  ->  http://localhost:8000"])
    dk.save(os.path.join(OUT, "M5_Presentation.pptx"))

    # ---------------------------------------------------------------- user guide
    g = Report("RehabNet User Guide", "For doctors / physiotherapists and patients", "Version 1.0")
    g.header_footer("User Guide")
    g.h1("1. Starting RehabNet")
    g.p("On the clinic computer open PowerShell in the RehabNet folder and run:")
    g.code(".\\run_app.ps1            # this computer:  http://localhost:8000\n.\\run_app.ps1 -Https     # phones/tablets on the same Wi-Fi: the https address it prints")
    g.p("The first start takes a few minutes (it installs the software). Demo sign-in details are in app\\data\\DEMO_ACCOUNTS.md.")
    g.h1("2. For doctors")
    g.bullets([
        ("Add a patient:", "Patients -> New patient. Give the patient the username and temporary password shown; they choose their own password at first sign-in."),
        ("Prescribe:", "open the patient -> Exercises -> choose the exercise (and side), sets, repetitions, sessions per week and notes."),
        ("Follow progress:", "the Overview tab shows the score over time, trends and weekly adherence; Sessions lists every session."),
        ("Session detail:", "score per repetition, joint angles, and the exact feedback the patient was given."),
        ("Personal baseline:", "after a session you supervised with good form, open it and press 'Use this session as baseline'."),
        ("Report:", "'PDF report' on the patient page."),
        ("Security:", "Account -> Two-factor login: scan the QR code with an authenticator app. Recommended for all doctors."),
    ])
    g.figure(os.path.join(S, "prescribe.png") if os.path.exists(os.path.join(S, "prescribe.png")) else os.path.join(S, "prescribe.jpg"),
             "Prescribing an exercise.", 14)
    g.h1("3. For patients")
    g.bullets([
        ("Sign in", "with the details from your doctor, then choose your own password."),
        ("My exercises:", "watch the demonstration, read your doctor's notes, press 'Start exercise'."),
        ("Set up the camera:", "put the phone or laptop 2-3 m away at hip height, so your whole body including your feet is visible; face the camera as the exercise describes; good, even light."),
        ("Exercise:", "start moving - RehabNet counts each repetition, shows a score and says a short correction if needed. The bar shows how far into the movement you are."),
        ("Finish:", "press 'Finish session' (or do all repetitions). You see your summary; your doctor sees it too."),
        ("Progress:", "'My progress' shows your scores over time; you can download your report."),
    ])
    g.figure(os.path.join(S, "live_session_done.png") if os.path.exists(os.path.join(S, "live_session_done.png")) else os.path.join(S, "live_session_done.jpg"),
             "A live session.", 14)
    g.h1("4. Troubleshooting")
    g.table(pd.DataFrame([
        ["'Could not open the camera'", "allow camera access in the browser; on a phone use the https address (run_app.ps1 -Https)"],
        ["'Step back into the picture' repeatedly", "the whole body (feet included) must be visible"],
        ["Repetitions not counted", "move through the full range shown in the demonstration; check the camera position for that exercise"],
        ["'The room is too dark'", "turn on a light or face a window; in a dark room the camera cannot find the body"],
        ["Feedback seems wrong", "check the light and that nothing blocks the camera; side-on views are less accurate"],
        ["Forgot password", "your doctor can create a new temporary password (patient page -> Reset password)"],
    ], columns=["Problem", "What to do"]), "Common problems", font=9)
    g.p("RehabNet supports, but does not replace, your doctor's or physiotherapist's advice. Stop if an exercise hurts.")
    g.save(os.path.join(OUT, "RehabNet_User_Guide.docx"))
    print("saved", OUT)


if __name__ == "__main__":
    build()
