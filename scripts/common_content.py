"""Text shared by all milestone reports and decks: background, problem, objectives, plan and references."""
import pandas as pd

MILESTONES = [
    ("M1", "Data and pose-estimation foundation", "June - July 2026",
     "Datasets, MediaPipe pipeline, normalisation, joint-angle validation vs motion capture"),
    ("M2", "Movement analysis and scoring engine", "August - September 2026",
     "Repetition counter, DTW + NCC comparison, multi-factor quality score, feedback"),
    ("M3", "Doctor-patient platform", "October - November 2026",
     "FastAPI backend, doctor portal, patient portal with live camera feedback, security"),
    ("M4", "Personalisation and progress", "December 2026 - January 2027",
     "Personal baselines, adaptive counting, multi-session progress, PDF reports"),
    ("M5", "Testing, optimisation and deployment", "February - March 2027",
     "Robustness, real-time speed, deployment, documentation"),
]

BIB = {
    "cieza": "A. Cieza, K. Causey, K. Kamenov, S. W. Hanson, S. Chatterji, T. Vos, \"Global estimates of the need for "
             "rehabilitation based on the Global Burden of Disease study 2019,\" The Lancet, vol. 396, no. 10267, "
             "pp. 2006-2017, 2020.",
    "argent": "R. Argent, A. Daly, B. Caulfield, \"Patient involvement with home-based exercise programs: can connected health "
              "interventions influence adherence?\" JMIR mHealth and uHealth, vol. 6, no. 3, e47, 2018.",
    "blazepose": "V. Bazarevsky, I. Grishchenko, K. Raveendran, T. Zhu, F. Zhang, M. Grundmann, \"BlazePose: On-device "
                 "real-time body pose tracking,\" arXiv:2006.10204, 2020.",
    "mediapipe": "C. Lugaresi et al., \"MediaPipe: A framework for building perception pipelines,\" arXiv:1906.08172, 2019.",
    "openpose": "Z. Cao, G. Hidalgo, T. Simon, S.-E. Wei, Y. Sheikh, \"OpenPose: Realtime multi-person 2D pose estimation "
                "using part affinity fields,\" IEEE Trans. Pattern Anal. Mach. Intell., vol. 43, no. 1, pp. 172-186, 2021.",
    "rehab24": "A. Černek, J. Sedmidubsky, P. Budikova, \"REHAB24-6: Physical therapy dataset for analyzing pose estimation "
               "methods,\" in Proc. SISAP, LNCS 15268, Springer, 2024, pp. 18-33. Data: doi:10.5281/zenodo.13305826.",
    "kimore": "M. Capecci et al., \"The KIMORE dataset: KInematic assessment of MOvement and clinical scores for remote "
              "monitoring of physical REhabilitation,\" IEEE Trans. Neural Syst. Rehabil. Eng., vol. 27, no. 7, "
              "pp. 1436-1448, 2019.",
    "uiprmd": "A. Vakanski, H. Jun, D. Paul, R. Baker, \"A data set of human body movements for physical rehabilitation "
              "exercises,\" Data, vol. 3, no. 1, art. 2, 2018.",
    "rehabpile": "A. Ismail-Fawaz, M. Devanne, S. Berretti, J. Weber, G. Forestier, \"A standardized benchmark for "
                 "skeleton-based rehabilitation assessment using deep learning\" (Rehab-Pile), arXiv:2507.21018, 2025.",
    "liao": "Y. Liao, A. Vakanski, M. Xian, \"A deep learning framework for assessing physical rehabilitation exercises,\" "
            "IEEE Trans. Neural Syst. Rehabil. Eng., vol. 28, no. 2, pp. 468-477, 2020.",
    "casiez": "G. Casiez, N. Roussel, D. Vogel, \"1 Euro filter: A simple speed-based low-pass filter for noisy input in "
              "interactive systems,\" in Proc. ACM CHI, 2012, pp. 2527-2530.",
    "sakoe": "H. Sakoe, S. Chiba, \"Dynamic programming algorithm optimization for spoken word recognition,\" IEEE Trans. "
             "Acoust., Speech, Signal Process., vol. 26, no. 1, pp. 43-49, 1978.",
    "dtaidistance": "W. Meert et al., \"DTAIDistance: time series distances, dynamic time warping\" (software), "
                    "https://github.com/wannesm/dtaidistance.",
    "lewis": "J. P. Lewis, \"Fast normalized cross-correlation,\" in Proc. Vision Interface, 1995, pp. 120-123.",
    "janosova": "K. Jánošová, P. Budikova, J. Sedmidubsky, \"Personalized similarity models for evaluating rehabilitation "
                "exercises from monocular videos,\" in Proc. SISAP, LNCS 15268, Springer, 2024, pp. 73-87.",
    "hanley": "J. A. Hanley, B. J. McNeil, \"The meaning and use of the area under a receiver operating characteristic "
              "(ROC) curve,\" Radiology, vol. 143, no. 1, pp. 29-36, 1982.",
    "sklearn": "F. Pedregosa et al., \"Scikit-learn: Machine learning in Python,\" J. Mach. Learn. Res., vol. 12, "
               "pp. 2825-2830, 2011.",
    "bcrypt": "N. Provos, D. Mazières, \"A future-adaptable password scheme,\" in Proc. USENIX Annual Technical Conference "
              "(FREENIX track), 1999.",
    "jwt": "M. Jones, J. Bradley, N. Sakimura, \"JSON Web Token (JWT),\" IETF RFC 7519, 2015.",
    "totp": "D. M'Raihi, S. Machani, M. Pei, J. Rydell, \"TOTP: Time-based one-time password algorithm,\" IETF RFC 6238, 2011.",
    "websocket": "I. Fette, A. Melnikov, \"The WebSocket protocol,\" IETF RFC 6455, 2011.",
    "fastapi": "S. Ramírez, \"FastAPI\" (software), https://fastapi.tiangolo.com.",
    "owasp": "OWASP Foundation, \"OWASP Top 10: The ten most critical web application security risks,\" 2021.",
    "docker": "D. Merkel, \"Docker: lightweight Linux containers for consistent development and deployment,\" Linux Journal, "
              "no. 239, 2014.",
}


def refs(*keys):
    return [BIB[k] for k in keys]


def plan_table(current):
    rows = []
    for i, (code, name, period, content) in enumerate(MILESTONES, 1):
        rows.append([code, name, period, content, "This report" if i == current else "Done"])
    return pd.DataFrame(rows, columns=["", "Milestone", "Period", "Content", "Status"])


def introduction(rp, current, objectives, scope_text):
    """Chapter 1 of every report: background, problem, project objectives, plan, this milestone's objectives."""
    rp.h1("Introduction")
    rp.h2("Background and motivation")
    rp.p("Physical rehabilitation helps people recover movement after injury, surgery, stroke or long-term conditions. "
         "The need is very large: a Global Burden of Disease analysis estimated that about 2.4 billion people worldwide live "
         "with a condition that would benefit from rehabilitation [1]. Most of the work happens at home: the "
         "physiotherapist demonstrates a set of exercises in the clinic and the patient is expected to repeat them several "
         "times a week without supervision.")
    rp.p("At home nobody checks whether an exercise is done correctly, how many repetitions were actually done, or whether "
         "the patient is improving. Patients forget the correct form, compensate with other joints, lose motivation, and the "
         "physiotherapist only learns about it at the next visit. Connected-health tools that give feedback and let the "
         "clinician follow progress are a recognised way to improve engagement with home exercise programmes [2].")
    rp.p("Existing measuring tools are either accurate but impractical at home (marker-based motion capture, depth cameras "
         "that are no longer manufactured, body-worn sensors) or simple apps that only play a demonstration video. Recent "
         "pose-estimation networks such as BlazePose [3] find 33 body landmarks in an ordinary camera image in real time on a "
         "phone, which makes it possible to measure exercises with the camera every patient already owns.")
    rp.h2("Problem statement")
    rp.callout("Build a system that watches a patient exercise through an ordinary phone or laptop camera, counts the "
               "repetitions, scores how well each one was done, tells the patient immediately how to correct it, and gives "
               "the physiotherapist a secure, objective record of every session and of progress over weeks.",
               title="Problem")
    rp.h2("Project objectives")
    rp.numbered([
        ("Accurate pose measurement", "from a single RGB camera, validated against motion capture."),
        ("Automatic exercise analysis", "repetition counting, quality scoring and specific corrective feedback."),
        ("Doctor-patient platform", "prescriptions, live guided sessions, session history and secure storage."),
        ("Personalisation and progress", "per-patient baselines, trends, adherence and printable reports."),
        ("A tested, deployable system", "robust to real home conditions, real-time on ordinary hardware, documented."),
    ])
    rp.h2("Project plan")
    rp.table(plan_table(current), "The five milestones of RehabNet", font=8.5)
    rp.h2(f"Objectives of Milestone {current}")
    rp.numbered(objectives)
    rp.p(scope_text)


def deck_intro(dk, current, objectives):
    """Problem + plan + objectives slides shared by every deck."""
    dk.cards("The problem", [
        ("Huge need", "About 2.4 billion people live with a condition that benefits from rehabilitation (Cieza et al., Lancet 2020)."),
        ("Unsupervised at home", "Most exercise is done at home, with no one checking form, counting repetitions or tracking progress."),
        ("Tools do not fit homes", "Motion capture, depth cameras and wearables are accurate but costly or impractical."),
        ("Our idea", "Use the phone or laptop camera the patient already has, with real-time pose estimation."),
    ], kicker="Motivation", cols=4, icons=False,
        takeaway="RehabNet turns any camera into a rehabilitation assistant for the patient and a measuring tool for the doctor.",
        say="Start with why: rehabilitation needs are large, home exercise is unsupervised, and existing measuring tools "
            "don't fit in a home. Our approach uses only a normal camera.")
    dk.flow("Project plan: five milestones", [
        (f"{c}\n{n}", [p, d]) for c, n, p, d in MILESTONES
    ], kicker="Plan", takeaway=f"This presentation covers Milestone {current}: {MILESTONES[current - 1][1]}.",
        say="The project follows the five milestones from the proposal; each ends with a report and this presentation.")
    dk.cards(f"Objectives of Milestone {current}", [(o[0], o[1]) if isinstance(o, tuple) else (o, "") for o in objectives],
             kicker="Objectives")
