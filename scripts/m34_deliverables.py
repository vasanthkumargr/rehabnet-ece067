"""Build Milestone 3 (platform) and Milestone 4 (personalisation, progress, reports) report + slides."""
import os
import sys

import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rehabnet.core.exercises import EXERCISES  # noqa: E402
from tools.deliver import Deck, Report  # noqa: E402
from scripts.common_content import deck_intro, introduction, refs  # noqa: E402

R3 = os.path.join(ROOT, "results", "m3")
R4 = os.path.join(ROOT, "results", "m4")
S = os.path.join(R3, "screens")
NICE = {k: v.name for k, v in EXERCISES.items()}


def shot(name):
    p = os.path.join(S, name)
    if p.endswith(".jpg"):  # convert to PNG once (python-docx/pptx handle both, PNG keeps text sharp)
        q = p[:-4] + ".png"
        if not os.path.exists(q):
            Image.open(p).save(q)
        return q
    return p


def m3():
    live = pd.read_csv(os.path.join(R3, "live_eval_summary.csv"))
    allr = live[live.exercise == "ALL"].iloc[0]
    lt = live.copy()
    lt["exercise"] = lt.exercise.map(lambda e: NICE.get(e, "All exercises"))
    lt = lt[["exercise", "counted", "auc", "correct_told_good", "incorrect_corrected", "mean_score_correct", "mean_score_incorrect"]]
    lt[["correct_told_good", "incorrect_corrected"]] *= 100
    lt.columns = ["Exercise", "Reps counted", "AUC", "Correct told 'good' %", "Incorrect corrected %", "Mean score correct", "Mean score incorrect"]
    stack = pd.DataFrame([
        ["Pose estimation", "MediaPipe BlazePose (Tasks for JavaScript, WebAssembly + GPU)", "in the browser; video never leaves the device"],
        ["Patient / doctor portal", "single-page web app (HTML, CSS, JavaScript modules, SVG charts)", "no build step; works on phones and laptops"],
        ["Backend API", "FastAPI (Python) + WebSocket for live sessions", "proposal: Flask/FastAPI"],
        ["Analysis engine", "rehabnet.core (Milestones 1-2): angles, counter, DTW/NCC scoring, feedback", "same code that was validated"],
        ["Database", "SQLAlchemy: SQLite (default) or PostgreSQL", "proposal: Firebase/PostgreSQL"],
        ["Security", "bcrypt, JWT, TOTP 2-factor, Fernet (AES-128) encryption at rest, audit log", "HIPAA-aligned design"],
        ["Reports", "ReportLab PDF + matplotlib charts", "Milestone 4"],
        ["Deployment", "Uvicorn; Dockerfile; HTTPS option for phones", "Milestone 5"],
    ], columns=["Layer", "Technology", "Note"])
    api = pd.DataFrame([
        ["POST /api/auth/login, /api/auth/mfa/verify", "sign-in (password, then 6-digit code if 2-factor is on)"],
        ["GET /api/me; POST /api/me/password, /api/me/mfa/*", "own account, password change, 2-factor set-up"],
        ["GET /api/exercises, /api/samples", "exercise library with demo clips; demo videos"],
        ["GET/POST /api/patients; GET /api/patients/{id}", "doctor: list / create patients, patient detail"],
        ["POST /api/patients/{id}/assignments; PATCH /api/assignments/{id}", "doctor: prescribe / change / stop exercises"],
        ["GET /api/patients/{id}/progress, /report.pdf", "trends, adherence, PDF report (doctor or the patient)"],
        ["GET /api/sessions/{id}", "session detail (decrypted for its patient and doctor only)"],
        ["POST/DELETE /api/patients/{id}/baseline/...", "doctor: personal baseline from a supervised session"],
        ["GET /api/my/assignments, /api/my/sessions", "patient: today's exercises and history"],
        ["WS /ws/session", "live session: landmarks in, counts / scores / feedback out"],
    ], columns=["Endpoint", "Purpose"])
    sec = pd.DataFrame([
        ["Passwords", "bcrypt (cost 12); temporary passwords must be changed at first sign-in"],
        ["Brute force", "account locked for 5 minutes after 5 wrong passwords"],
        ["Two-factor login", "TOTP (RFC 6238) with any authenticator app; QR set-up; doctors are prompted to enable it"],
        ["Sessions", "signed tokens (HS256) expiring after 12 h; half-way token after password is valid only for the code step (5 min)"],
        ["Access control", "patients see only their own data; doctors only their own patients (tested)"],
        ["Data at rest", "per-repetition results, feedback and angle traces encrypted with Fernet (AES-128-CBC + HMAC)"],
        ["Privacy", "camera video is processed on the device; only 33 body landmarks are sent"],
        ["Audit", "log of sign-ins, failed sign-ins, record views, PDF exports, baseline changes"],
        ["Transport", "HTTPS supported (self-signed for local network, real certificate in deployment); security headers"],
    ], columns=["Measure", "Implementation"])
    tests = pd.DataFrame([
        ["Health check", "server responds"], ["Wrong password + lock-out", "5 failures lock the account"],
        ["Login required", "API refuses missing / invalid tokens"],
        ["Patient flow & permissions", "create patient, prescribe, forced password change; other doctor and patient are refused"],
        ["Two-factor", "set-up, wrong code refused, half-way token is not a login, correct code signs in"],
        ["Live session end-to-end", "80 s squat recording streamed over WebSocket: reps counted, scored, saved encrypted, visible to doctor, progress, PDF, baseline then used in the next session"],
        ["WebSocket security", "bad token refused"],
    ], columns=["Automated test", "What it checks"])

    req = pd.DataFrame([
        ["F1", "Doctor creates patient accounts and prescribes exercises (side, sets, reps, sessions per week, notes)", "Doctor"],
        ["F2", "Patient sees today's exercises with a demonstration and instructions", "Patient"],
        ["F3", "Live session: camera skeleton, repetition counter, score and correction after every repetition", "Patient"],
        ["F4", "Every session stored with per-repetition scores, feedback and joint-angle traces", "System"],
        ["F5", "Doctor sees each patient's sessions, scores, feedback given and progress", "Doctor"],
        ["F6", "Secure sign-in with optional two-factor login; patients change temporary passwords", "Both"],
        ["F7", "Analyse a recorded video instead of the live camera", "Both"],
    ], columns=["ID", "Functional requirement", "User"])
    nfr = pd.DataFrame([
        ["Privacy", "camera video never leaves the patient's device; only body landmarks are sent"],
        ["Latency", "feedback within ~1.5 s after a repetition ends; server < 1 ms per frame"],
        ["Security", "encryption at rest, access control per patient, audit log, lock-out"],
        ["Portability", "any modern browser on a phone or laptop; no app installation"],
        ["Maintainability", "the analysis is the same validated Python code as Milestones 1-2"],
        ["Offline use", "all JavaScript, WebAssembly and model files served locally"],
    ], columns=["Quality attribute", "Requirement"])
    mods = pd.DataFrame([
        ["app/server/main.py", "REST API, WebSocket live sessions, security headers, start-up clean-up"],
        ["app/server/engine.py", "LiveSession: 10 Hz resampling, smoothing, counting, scoring, feedback per repetition"],
        ["app/server/security.py", "bcrypt passwords, JWT tokens, TOTP codes, Fernet encryption of stored data"],
        ["app/server/db.py", "SQLAlchemy tables: users, assignments, sessions, baselines, audit"],
        ["app/web/static/pose.js", "MediaPipe in the browser (GPU, CPU fallback), camera view, framing, skeleton drawing"],
        ["app/web/static/live.js", "live session screen: camera / video loop, WebSocket, gauge, spoken feedback, summary"],
        ["app/web/static/app.js", "router and all doctor and patient pages"],
        ["rehabnet/core/*", "pose, normalisation, angles, exercise library, counter, scoring, feedback (Milestones 1-2)"],
    ], columns=["Module", "Responsibility"])
    threats = pd.DataFrame([
        ["Password guessing", "bcrypt cost 12, lock-out after 5 failures, optional TOTP second factor"],
        ["Stolen session token", "tokens expire after 12 h; half-way token after password only valid for the code step"],
        ["Broken access control", "every patient/session request checks owner or treating doctor (automated test)"],
        ["Database theft", "repetition data, feedback and traces encrypted with Fernet; passwords only as hashes"],
        ["Video privacy", "video processed on the device; only 33 landmarks sent"],
        ["Clickjacking / sniffing", "security headers (frame-ancestors, nosniff, referrer policy); HTTPS in deployment"],
        ["Unnoticed misuse", "audit log of sign-ins, failures, record views, PDF exports and baseline changes"],
    ], columns=["Threat (OWASP-style)", "Mitigation in RehabNet"])

    rp = Report("Milestone 3: Doctor-Patient Platform", "Backend API, doctor portal, patient portal with live feedback, security",
                "Planned: October - November 2026")
    rp.header_footer("Milestone 3")
    rp.abstract(
        "This report describes the RehabNet web platform. A FastAPI backend serves a doctor portal and a patient portal as "
        "one web application. Pose estimation runs in the patient's browser with MediaPipe for JavaScript, so camera video "
        "never leaves the device; only 33 body landmarks are streamed over a WebSocket to the server, which runs the "
        "validated counting, scoring and feedback engine of Milestone 2 and answers after every repetition. Doctors create "
        "patients, prescribe exercises and review every session; patients follow guided, spoken live feedback. Security "
        "includes bcrypt passwords, signed tokens, time-based two-factor login, account lock-out, per-patient access control, "
        f"encryption of stored session data and an audit log. Evaluated live on unseen people the system reached AUC "
        f"{allr.auc:.2f} for correct versus incorrect repetitions; the browser result matched the offline analysis, and all "
        "automated tests pass.",
        ["web application", "FastAPI", "WebSocket", "MediaPipe JavaScript", "telerehabilitation", "security", "two-factor authentication"])
    rp.toc()
    introduction(rp, 3, [
        ("Backend API", "that serves the analysis engine to many patients at once."),
        ("Doctor portal", "exercise library, prescriptions, session review."),
        ("Patient portal", "exercise guide and live camera session with immediate feedback."),
        ("Security", "encrypted storage and multi-factor authentication for health data."),
    ], "The aim of this milestone is a working end-to-end application. Accuracy questions were settled in Milestones 1-2; "
       "here the question is whether the same analysis works live, in a browser, at the speed of a real exercise.")
    rp.h1("Requirements")
    rp.p("Two kinds of user drive the requirements: the physiotherapist or doctor, who prescribes and reviews, and the "
         "patient, who exercises at home with a phone or laptop.")
    rp.table(req, "Functional requirements", font=9)
    rp.table(nfr, "Non-functional requirements", font=9)
    rp.h1("Summary of results")
    rp.p("Milestone 3 turns the validated analysis of Milestones 1-2 into a working web application. A doctor creates a "
         "patient account and prescribes exercises; the patient opens the app on a phone or laptop, follows the exercise "
         "guide and exercises in front of the camera while RehabNet counts the repetitions, scores each one and speaks "
         "corrections; the doctor then sees every session, the scores and the feedback that was given.")
    rp.bullets([
        ("Working end to end:", "tested in the browser with real REHAB24-6 recordings played as the camera; the browser "
                                "result matched the offline analysis (5 of 5 repetitions, scores within a few points, same corrections)."),
        ("Live accuracy (unseen people):", f"correct vs incorrect AUC {allr.auc:.2f}; {allr.correct_told_good*100:.0f} % of "
                                            f"correct repetitions told 'good', {allr.incorrect_corrected*100:.0f} % of incorrect ones corrected; "
                                            f"mean score {allr.mean_score_correct:.0f} for correct vs {allr.mean_score_incorrect:.0f} for incorrect repetitions."),
        ("Fast:", "the server needs about 0.4 ms per camera frame; pose estimation runs on the patient's device."),
        ("Secure:", "two-factor login, encrypted session data, lock-out, access control and audit log; 7 automated tests pass."),
    ])
    st = pd.DataFrame([
        ["RESTful backend API (Flask/FastAPI)", "Done", "FastAPI + WebSocket"],
        ["Doctor portal: library, assignment, report dashboard", "Done", "plus progress charts and PDF (M4)"],
        ["Patient portal: guide, live camera feedback, summary", "Done", "spoken + on-screen feedback"],
        ["Encrypted storage and multi-factor authentication", "Done", "Fernet + TOTP"],
    ], columns=["Task (from proposal)", "Status", "Note"])
    rp.table(st, "Milestone 3 tasks")
    rp.h1("System design")
    rp.h2("Architecture")
    rp.figure(os.path.join(R3, "fig_architecture.png"), "RehabNet architecture.", 16)
    rp.p("Pose estimation runs in the patient's browser (MediaPipe for JavaScript [9], using the device GPU through WebGL "
         "when available). Only the 33 three-dimensional body landmarks are sent to the server, about 15 times a second, "
         "over a WebSocket [5]. The server runs exactly the Milestone 2 code: it resamples to 10 Hz, smooths, computes joint "
         "angles, counts repetitions and scores each repetition when it ends, and sends the result back within the same "
         "connection. When the session ends, the summary is stored and the detailed data is encrypted.")
    rp.table(stack, "Technology", font=8.5)
    rp.h2("Live session protocol")
    rp.figure(os.path.join(R3, "fig_sequence.png"), "Messages exchanged during one live session.", 15.5)
    rp.code('client -> {"type":"start","token":...,"assignment_id":...}\n'
            'server -> {"type":"ready","target":20,"personal":false}\n'
            'client -> {"type":"frame","t":ms,"w":[[x,y,z] x 33],"view":"front"}      (~15 per second)\n'
            'server -> {"type":"tick","progress":0.62} | {"type":"rep_counted","count":3}\n'
            'server -> {"type":"rep_scored","score":87,"feedback":[{"text":"Keep your upper body upright ..."}]}\n'
            'client -> {"type":"end"}   server -> {"type":"summary","reps":20,"mean_score":84,...}')
    rp.p("The server is robust to real network conditions: frames arriving more than 0.5 s apart are not filled in "
         "(so a pause is not mistaken for a slow movement), a clock going backwards resets the sampler, and the last "
         "repetition is scored and sent before the summary when the patient stops.")
    rp.h2("Data model")
    rp.figure(os.path.join(R3, "fig_datamodel.png"), "Database tables (SQLAlchemy). Grey fields marked data_enc are encrypted.", 15.5)
    rp.p("Doctors and patients share one users table with a role; each patient belongs to one doctor. Searchable "
         "summary values (date, repetitions, mean score) are stored in plain columns so lists and charts are fast, while "
         "the detailed per-repetition data, feedback texts and angle traces are encrypted as one block per session.")
    rp.h2("REST API")
    rp.table(api, "REST API", font=8.5)
    rp.h1("Implementation")
    rp.table(mods, "Main code modules", font=8.5)
    rp.p("The web front end is a single-page application written in plain JavaScript modules with hand-drawn SVG charts, "
         "so it needs no build tools and loads quickly on phones. MediaPipe's JavaScript, WebAssembly and model files are "
         "served by RehabNet itself, so the app also works without internet in a clinic.")
    rp.h1("Doctor portal")
    rp.figure(shot("doctor_home.jpg"), "Patients list with last score and weekly adherence.", 15)
    rp.figure(shot("prescribe.jpg"), "Prescribing an exercise (side, sets, repetitions, sessions per week, notes) with the demonstration.", 15)
    rp.figure(shot("library.jpg"), "Exercise library: demonstration clip, instructions and recommended camera position.", 15)
    rp.figure(shot("session_detail.jpg"), "Session detail: score per repetition and score breakdown.", 15)
    rp.figure(shot("session_feedback.jpg"), "Every repetition with its range of motion and the feedback the patient received.", 15)
    rp.h1("Patient portal and live feedback")
    rp.figure(shot("patient_home.jpg"), "Patient home: today's exercises with demonstration, prescription and weekly progress.", 15)
    rp.figure(shot("live_session_done.jpg"), "Live session (here on a REHAB24-6 recording): skeleton overlay, counter, last "
              "repetition score and correction, list of repetitions, session summary.", 15)
    rp.bullets([
        ("Guidance before starting:", "instructions, recommended camera position, whole-body framing check."),
        ("During the exercise:", "skeleton overlay, repetition counter, movement gauge (0 = rest, full = reference depth), "
                                 "score after each repetition, the correction read aloud, the body part concerned highlighted."),
        ("Checks:", "'step back into the picture', 'come closer', 'turn so the camera sees you from the front' when the "
                    "body orientation measured from the shoulders does not suit the exercise, and 'the room is too dark' (added in Milestone 5)."),
        ("Video mode:", "any exercise can be analysed from a video file instead of the camera; frames are analysed one by one, "
                        "so the result is the same on any device."),
    ])
    rp.h1("Security")
    rp.p("Health data needs protection at every layer. Passwords are hashed with bcrypt [6]; sessions use signed JSON Web "
         "Tokens [7]; the second factor uses time-based one-time passwords (TOTP) [8] compatible with any authenticator app. "
         "The threats considered follow the OWASP Top 10 [10].")
    rp.table(sec, "Security measures", font=8.5)
    rp.table(threats, "Threats and mitigations", font=8.5)
    rp.figure(shot("mfa.jpg"), "Two-factor login set-up with an authenticator app.", 14)
    rp.h1("Validation")
    rp.h2("Live pipeline on unseen people")
    rp.p("The live pipeline cuts repetitions with the counter (not with the dataset's annotation) and scores them against "
         "references cut the same way. It was evaluated leave-one-person-out on all REHAB24-6 camera recordings: each "
         "counted repetition was matched to the physiotherapist's label of the annotated repetition containing its peak.")
    rp.table(lt.round(2), "Live pipeline, REHAB24-6 camera data, unseen people", font=8.5)
    rp.h2("Browser vs offline analysis")
    rp.p("The demo recording of arm abduction (5 repetitions) analysed live in the browser gave 5 counted repetitions with "
         "scores 71, 89, 73, 74 and 87 and the same corrections as the offline Python analysis (69, 88, 68, 68, 87). The "
         "first browser test also revealed a real problem: when the browser tab is not visible, browsers slow down "
         "animation, frames were skipped and repetitions were measured too shallow. This was fixed (video files are now "
         "analysed frame by frame; the server no longer fills long gaps with repeated frames).")
    rp.h2("Automated tests")
    rp.table(tests, "Automated API and end-to-end tests (pytest): all 7 pass", font=8.5)
    rp.h1("Deviations from the proposal")
    rp.bullets([
        "Plain JavaScript single-page app instead of React/Flutter: no build tools are needed, it runs on any phone or laptop browser and is easy to maintain.",
        "SQLite by default (zero set-up), PostgreSQL through one setting, instead of Firebase: keeps patient data under the hospital's control.",
        "Pose estimation in the browser instead of on the server: lower latency, works with weak servers, and the video never leaves the patient's device.",
    ])
    rp.h1("Conclusion")
    rp.p("Milestone 3 delivered a complete, secure doctor-patient platform around the analysis engine. The key design "
         "choice - pose estimation on the patient's device, analysis on the server - keeps video private, keeps the server "
         "light and lets the exact validated code run live. Live accuracy on unseen people is close to the offline result "
         f"(AUC {allr.auc:.2f}), and the browser and offline analyses agree. Milestone 4 adds personal baselines, progress "
         "tracking and printable reports on top of this platform.")
    rp.references(refs("cieza", "argent", "blazepose", "fastapi", "websocket", "bcrypt", "jwt", "totp", "mediapipe", "owasp"))
    rp.save(os.path.join(ROOT, "deliverables", "M3", "M3_Report.docx"))

    dk = Deck("Milestone 3: Doctor-Patient Platform", "Backend API, doctor portal, patient portal with live feedback, security",
              "Planned: October - November 2026")
    dk.agenda([("Problem and plan", "where Milestone 3 fits"), ("Requirements", "doctor, patient, quality"),
               ("System design", "architecture, live protocol, data model"), ("Doctor and patient portals", "screens"),
               ("Security", "threats and mitigations"), ("Validation", "live accuracy, browser test, automated tests")])
    deck_intro(dk, 3, [
        ("Backend API", "serves the analysis engine to many patients"),
        ("Doctor portal", "library, prescriptions, session review"),
        ("Patient portal", "guide + live camera session with feedback"),
        ("Security", "encrypted storage, two-factor login"),
    ])
    dk.stats("Milestone 3 in numbers", [
        ("2", "portals", "doctor and patient, one web app"),
        ("0.4 ms", "server time per frame", "analysis engine, live"),
        (f"{allr.auc:.2f}", "live AUC", "correct vs incorrect, unseen people"),
        ("5 / 5", "reps in browser test", "same as offline analysis"),
        ("7 / 7", "automated tests pass", "incl. full live session"),
    ], kicker="Summary", takeaway="A working, secure web app: the validated engine now runs live in any browser.")
    dk.table("Milestone 3 tasks", st, kicker="Status", font=15, col_w=[6.0, 1.6, 4.4])
    dk.section(1, "Requirements", "What doctors and patients need, and how well it must work")
    dk.table("Functional requirements", req, kicker="Requirements", font=13, col_w=[0.7, 9.6, 1.6])
    dk.cards("Quality requirements", [(a, b) for a, b in nfr.values], kicker="Requirements",
             takeaway="Privacy drives the design: the video stays on the patient's device.")
    dk.section(2, "System design", "Pose on the device, analysis on the server")
    dk.image("Architecture", os.path.join(R3, "fig_architecture.png"), kicker="Design",
             takeaway="Only 33 landmarks per frame travel to the server - never the video.")
    dk.image("Live session protocol", os.path.join(R3, "fig_sequence.png"), kicker="Design",
             takeaway="Count, score and correction come back over the same WebSocket during the exercise.")
    dk.image("Data model", os.path.join(R3, "fig_datamodel.png"), kicker="Design",
             takeaway="Summaries in plain columns for fast charts; detailed repetition data encrypted.")
    dk.table("Technology", stack, kicker="Design", font=11, col_w=[2.4, 5.6, 4.0])
    dk.table("Code modules", mods, kicker="Implementation", font=12, col_w=[3.4, 8.6])
    dk.section(3, "Doctor and patient portals", "What each user sees")
    dk.images("Doctor portal", [shot("doctor_home.jpg"), shot("prescribe.jpg")], ["Patients list", "Prescribe an exercise"],
              kicker="Doctor portal", takeaway="Prescriptions carry side, sets, repetitions, sessions per week and notes.")
    dk.images("Doctor: reviewing a session", [shot("session_detail.jpg"), shot("session_feedback.jpg")],
              ["Score per repetition", "Feedback the patient received"], kicker="Doctor portal",
              takeaway="The doctor sees exactly what the patient was told, repetition by repetition.")
    dk.image("Patient: today's exercises", shot("patient_home.jpg"), kicker="Patient portal",
             takeaway="Demonstration, prescription and weekly progress on one screen.")
    dk.image("Patient: live session", shot("live_session_done.jpg"), kicker="Patient portal",
             notes=["Skeleton overlay", "Counter + movement gauge", "Score + spoken correction", "Joint highlighted",
                    "Framing, view and light checks", "Summary at the end"],
             takeaway="Feedback is spoken, so the patient does not need to look at the screen.")
    dk.section(4, "Security and validation", "Protecting health data, and checking it all works")
    dk.table("Threats and mitigations", threats, kicker="Security", font=12, col_w=[3.2, 8.8])
    dk.image("Two-factor login", shot("mfa.jpg"), kicker="Security",
             notes=["TOTP (RFC 6238), any authenticator app", "QR-code set-up", "Doctors prompted to enable it",
                    "Half-way token only valid for the code step"])
    dk.table("Live pipeline on unseen people", lt.round(2), kicker="Validation", font=11,
             takeaway=f"Live AUC {allr.auc:.2f}; correct repetitions score {allr.mean_score_correct:.0f} on average vs "
                      f"{allr.mean_score_incorrect:.0f} for incorrect ones.")
    dk.cards("Browser test: a real bug found and fixed", [
        ("Test", "demo arm-abduction video played as the camera, analysed live in the browser"),
        ("Result", "5 / 5 repetitions; scores 71, 89, 73, 74, 87 vs offline 69, 88, 68, 68, 87"),
        ("Bug found", "hidden browser tabs slow animation -> frames skipped -> movements measured too shallow"),
        ("Fix", "video analysed frame by frame; server no longer fills long gaps"),
    ], kicker="Validation", cols=2)
    dk.table("Automated tests (all pass)", tests, kicker="Validation", font=11, col_w=[3.2, 8.8])
    dk.cards("Design choices vs the proposal", [
        ("Plain JavaScript app", "instead of React/Flutter: no build tools, runs in any browser"),
        ("SQLite / PostgreSQL", "instead of Firebase: data stays under the clinic's control"),
        ("Pose in the browser", "lower latency, light server, video never leaves the device"),
    ], kicker="Conclusions", takeaway="Milestone 4 adds personal baselines, progress tracking and PDF reports.")
    dk.closing(lines=["Questions and discussion", "Live demo: .\\run_app.ps1  ->  http://localhost:8000"])
    dk.save(os.path.join(ROOT, "deliverables", "M3", "M3_Presentation.pptx"))


def m4():
    be = pd.read_csv(os.path.join(R4, "baseline_eval_summary.csv"))
    bc = pd.read_csv(os.path.join(R4, "baseline_count_summary.csv"))
    be_t = be.copy()
    be_t["exercise"] = be_t.exercise.map(lambda e: NICE.get(e, "Mean"))
    be_t.columns = ["Exercise", "Test reps", "General refs", "Personal only", "Combined (used)"]
    bc_t = bc.copy()
    bc_t.columns = ["Counter thresholds", "Recall", "Precision", "Count error"]
    st = pd.DataFrame([
        ["Per-patient baseline + dynamic thresholds", "Done", "baseline from a supervised session; counter adapts for limited range"],
        ["Multi-session progress tracking + trends", "Done", "score & range-of-motion trend per week, adherence"],
        ["Automated report generation (PDF / in-app)", "Done", "PDF for doctor and patient; in-app session pages"],
        ["UI/UX improvements", "Done", "guided overlays, spoken feedback, joint highlight, framing & view checks"],
    ], columns=["Task (from proposal)", "Status", "Note"])
    mean = be[be.exercise == "MEAN"].iloc[0]
    rp = Report("Milestone 4: Personalisation and Progress", "Personal baselines, adaptive counting, progress tracking, PDF reports",
                "Planned: December 2026 - January 2027")
    rp.header_footer("Milestone 4")
    rp.abstract(
        "This report describes how RehabNet adapts to the individual patient and follows them over time. A doctor can mark "
        "a supervised session as the patient's personal baseline; the patient's own good repetitions are then added to the "
        "general reference set, and if the patient's comfortable range of motion is clearly smaller than the reference, the "
        "repetition counter switches to thresholds derived from the patient. Both choices were tested before being built: "
        f"on unseen people, combined references reached AUC {mean['combined']:.3f} versus {mean['generic']:.3f} for general "
        f"references and {mean['personal']:.3f} for personal references alone. Progress is summarised as least-squares "
        "trends per week for the score and the range of motion, with a noise-aware label (improving, stable, getting worse), "
        "plus weekly adherence to the prescription. Doctors and patients can download an automatically generated PDF report.",
        ["personalisation", "baseline", "progress tracking", "adherence", "automated reports", "user experience"])
    rp.toc()
    introduction(rp, 4, [
        ("Personal baselines", "score each patient also against their own supervised good repetitions."),
        ("Adaptive thresholds", "keep counting repetitions for patients with a limited range of motion."),
        ("Progress over weeks", "trends per exercise and adherence to the prescription."),
        ("Reports and usability", "printable reports and a friendlier live session."),
    ], "Every design choice in this milestone was first tested on REHAB24-6 with the live pipeline on unseen people, "
       "so that personalisation is only used where it actually helps.")
    rp.h1("Background")
    rp.p("Patients differ: an older patient or someone after surgery may move correctly but through a smaller range than "
         "the healthy volunteers who recorded the references. Comparing such a patient only with others would mark every "
         "repetition as poor and might not even count it. Jánošová et al. [4] showed that personalised references can "
         "improve evaluation of rehabilitation exercises from video. In Milestone 2 we found that five noisy camera "
         "repetitions alone are not a stable reference, so here personal repetitions are combined with general ones.")
    rp.p("Doctors also need to see change over time, not single sessions: whether the score and range of motion are "
         "improving, and whether the patient actually exercises as prescribed. Low adherence is common in home programmes, "
         "and making progress visible to both patient and clinician is one way connected-health tools support it [2].")
    rp.h1("Summary of results")
    rp.bullets([
        ("Personal baseline:", "a doctor marks a supervised session as the patient's baseline; later sessions are scored against "
                               "the general references plus the patient's own good repetitions."),
        ("Evidence-based design:", f"on unseen people, combined references gave AUC {mean['combined']:.3f} vs {mean['generic']:.3f} "
                                   f"for general references and only {mean['personal']:.3f} for the patient's own five repetitions alone, "
                                   "so personal repetitions are added to, not used instead of, the general references."),
        ("Adaptive counting:", "if a patient's comfortable range is below 70 % of the reference, the counter switches to the "
                               "patient's own thresholds, so their repetitions are still counted."),
        ("Progress:", "score and range-of-motion trends per week (improving / stable / getting worse), weekly adherence, PDF report."),
    ])
    rp.table(st, "Milestone 4 tasks", font=9)
    rp.h1("Personal baseline")
    rp.h2("Method")
    rp.numbered([
        ("Supervised session:", "the patient exercises once with the physiotherapist watching."),
        ("Mark as baseline:", "on the session page the doctor presses 'Use this session as baseline'; the good repetitions "
                              "(score at least 50; at least 5 are needed and the best 15 are kept) are stored encrypted for that patient and exercise."),
        ("Combined references:", "later sessions are scored against the general references plus the personal repetitions, "
                                 "each personal repetition counted three times."),
        ("Adaptive counter:", "if the baseline range is below 70 % of the reference range, the counter thresholds are taken "
                              "from the patient's baseline instead."),
        ("Clear when needed:", "the doctor can remove the baseline at any time; the patient sees 'Using your personal baseline' during sessions."),
    ])
    rp.h2("Evaluation")
    rp.p("Evaluation: for every person and exercise in REHAB24-6, the first five counted repetitions that the physiotherapist "
         "labelled correct act as the supervised baseline session. All later repetitions of that person are scored three "
         "ways, with references from other people only, from the person's baseline only, or both (personal repetitions "
         "counted three times). Thresholds and references never include the person being tested.")
    rp.table(be_t.round(3), "AUC for correct vs incorrect repetitions after the baseline session (camera, live pipeline)", font=9)
    rp.table(bc_t.round(3), "Repetition counting after the baseline session: general vs personal thresholds", font=9)
    rp.p("Personal thresholds counted slightly worse on these healthy volunteers, whose range matches the references. "
         "Their value is for patients with a limited range of motion; therefore the app only switches to them when the "
         "patient's baseline range is below 70 % of the reference range.")
    rp.h1("Progress tracking")
    rp.h2("Trends")
    rp.p("For each exercise the session scores are fitted with a least-squares line over time. The slope per week is "
         "reported as 'improving' or 'getting worse' only if it exceeds both 1 point per week and the scatter of the "
         "sessions around the line; otherwise 'stable'. The same is done for the range-of-motion sub-score. At least "
         "three sessions over more than one day are needed before a trend is shown.")
    rp.code("slope_per_week = 7 × polyfit(days, scores, 1)[0]\n"
            "noise          = std(residuals) / √n\n"
            "label          = 'stable'          if |slope| < max(1, noise)\n"
            "                 'improving'       if slope > 0\n"
            "                 'getting worse'   otherwise")
    rp.h2("Adherence")
    rp.p("Adherence is the number of sessions done divided by the sessions prescribed per week, for each of the last "
         "eight weeks (capped at 100 %). A prescription counts for a week only if it existed for at least half of that "
         "week, so a newly added exercise does not make the patient look non-adherent.")
    rp.figure(shot("patient_overview.jpg"), "Doctor's overview of a (demo) patient: sessions, adherence and score over time.", 15)
    rp.figure(shot("patient_trends.jpg"), "Trend table and weekly adherence.", 15)
    rp.figure(shot("session_angles.jpg"), "Joint-angle traces of a session, with each counted repetition marked.", 15)
    rp.h1("Automated reports")
    rp.p("Doctors (and patients for themselves) can download a PDF progress report: prescriptions, progress chart, trend "
         "table, weekly adherence and the last sessions with the most frequent correction in each. The report is generated "
         "on the server with ReportLab and matplotlib and every export is written to the audit log.")
    rp.figure(os.path.join(S, "report_page1.png"), "First page of the automatically generated PDF report (demo patient).", 13)
    rp.figure(os.path.join(S, "report_page2.png"), "Second page: recent sessions and their most frequent correction.", 13)
    rp.h1("User experience improvements")
    rp.table(pd.DataFrame([
        ["Spoken feedback", "browser speech reads each correction", "patients need not look at the screen while exercising"],
        ["Joint highlight", "the joint a correction refers to glows on the skeleton for 3 s", "shows where to correct"],
        ["Framing check", "'step back', 'come closer'", "avoids the far-away failure seen in testing"],
        ["View check", "asks to face the camera as the exercise needs", "counting is most reliable front / 45°"],
        ["Encouraging tone", "'Good form. Next time: ...' when only size or speed could improve", "motivation"],
        ["Movement gauge", "how far into the movement the patient is vs the reference", "guides the range of motion"],
        ["Demo clips", "a short demonstration with every exercise", "correct form before starting"],
        ["Responsive layout", "phones and laptops", "no app installation"],
    ], columns=["Feature", "What it does", "Why"]), "User-experience features", font=8.5)
    rp.h1("Limitations")
    rp.bullets(["The baseline evaluation uses healthy volunteers; the adaptive counter for limited range needs patient data to tune the 70 % rule.",
                "Demo progress history is replayed REHAB24-6 data ordered to show a trend; it demonstrates the features, not real recovery.",
                "Trend labels use a simple linear model; plateaus and sudden changes (e.g. after a flare-up) are summarised as one slope."])
    rp.h1("Conclusion")
    rp.p("Milestone 4 made RehabNet personal and longitudinal. Testing before building showed that a patient's own "
         "repetitions help only when combined with general references, and that personal counting thresholds should be "
         "reserved for patients with a clearly limited range - so that is exactly how the app behaves. Doctors now see "
         "trends and adherence at a glance and can print a report; patients get spoken, encouraging, targeted guidance. "
         "Milestone 5 tests the whole system under real-world conditions and prepares it for deployment.")
    rp.references(refs("cieza", "argent", "blazepose", "janosova", "rehab24"))
    rp.save(os.path.join(ROOT, "deliverables", "M4", "M4_Report.docx"))

    dk = Deck("Milestone 4: Personalisation and Progress", "Personal baselines, adaptive counting, progress tracking, PDF reports",
              "Planned: December 2026 - January 2027")
    dk.agenda([("Problem and plan", "where Milestone 4 fits"), ("Personal baseline", "method and evidence"),
               ("Adaptive counting", "patients with limited range"), ("Progress tracking", "trends and adherence"),
               ("Reports", "automatic PDF"), ("User experience", "spoken, guided, encouraging")])
    deck_intro(dk, 4, [
        ("Personal baselines", "score against the patient's own good repetitions too"),
        ("Adaptive thresholds", "count repetitions even with limited range"),
        ("Progress over weeks", "trends and adherence"),
        ("Reports and usability", "PDF reports, friendlier live session"),
    ])
    dk.stats("Milestone 4 in numbers", [
        (f"{mean['combined']:.3f}", "AUC combined refs", "general + personal (used)"),
        (f"{mean['generic']:.3f}", "AUC general refs", "other people only"),
        (f"{mean['personal']:.3f}", "AUC personal only", "5 camera reps: too noisy"),
        ("70 %", "range rule", "own counter thresholds below this"),
        ("8 weeks", "adherence history", "per prescription"),
    ], kicker="Summary", takeaway="Personal repetitions are added to - not used instead of - the general references.")
    dk.table("Milestone 4 tasks", st, kicker="Status", font=12, col_w=[4.6, 1.2, 6.2])
    dk.section(1, "Personal baseline", "Comparing each patient with their own good repetitions")
    dk.flow("How a personal baseline works", [
        ("Supervised session", "patient exercises with the physiotherapist"),
        ("Mark as baseline", "doctor presses one button on the session page"),
        ("Store encrypted", "patient's good repetitions for that exercise"),
        ("Combined scoring", "general refs + personal reps (counted 3×)"),
        ("Adaptive counter", "own thresholds if range < 70 % of reference"),
    ], kicker="Baseline", takeaway="The patient sees 'Using your personal baseline'; the doctor can clear it at any time.")
    dk.table("Does a personal baseline help?", be_t.round(3), kicker="Baseline", font=13,
             notes=["Unseen people, camera", "5 supervised reps as baseline", "Personal alone is too noisy", "Combined is used"],
             takeaway="Tested before building: combined references are the safe choice.")
    dk.table("Counting with personal thresholds", bc_t.round(3), kicker="Baseline", font=14,
             notes=["Healthy volunteers: no gain", "Kept for limited-range patients only"],
             takeaway="Own thresholds only when the patient really moves less than the references.")
    dk.section(2, "Progress and reports", "Following the patient over weeks")
    dk.cards("How progress is measured", [
        ("Score trend", ["least-squares slope per week", "'improving' only if > 1 point/week and above noise"]),
        ("Range-of-motion trend", ["same method on the ROM sub-score"]),
        ("Adherence", ["sessions done / sessions prescribed per week", "last 8 weeks, fair to new prescriptions"]),
    ], kicker="Progress")
    dk.image("Progress for the doctor", shot("patient_overview.jpg"), kicker="Progress",
             takeaway="Sessions, adherence and score over time on one page.")
    dk.images("Trends and joint angles", [shot("patient_trends.jpg"), shot("session_angles.jpg")],
              ["Trend table and adherence", "Joint angles of a session"], kicker="Progress")
    dk.images("Automated PDF report", [os.path.join(S, "report_page1.png"), os.path.join(S, "report_page2.png")],
              ["Page 1: prescriptions, chart, trends", "Page 2: recent sessions"], kicker="Reports",
              takeaway="One click for the doctor or the patient; every export is audit-logged.")
    dk.section(3, "User experience", "Guidance the patient can follow without looking at the screen")
    dk.cards("User-experience improvements", [
        ("Spoken feedback", "each correction read aloud"),
        ("Joint highlight", "the joint to correct glows on the skeleton"),
        ("Framing and view checks", "'step back', 'come closer', 'turn to face the camera'"),
        ("Encouraging tone", "'Good form. Next time: ...'"),
        ("Movement gauge", "how deep into the movement you are"),
        ("Demo clip", "with every exercise"),
    ], kicker="UX")
    dk.cards("Limitations", [
        ("Healthy volunteers", "the 70 % rule needs tuning on real patients"),
        ("Demo history", "replayed REHAB24-6 data shows the features, not real recovery"),
        ("Linear trends", "plateaus and sudden changes become one slope"),
    ], kicker="Conclusions", takeaway="Milestone 5 tests the whole system under real-world conditions and prepares deployment.")
    dk.closing()
    dk.save(os.path.join(ROOT, "deliverables", "M4", "M4_Presentation.pptx"))


if __name__ == "__main__":
    m3(); m4()
    print("ok")
