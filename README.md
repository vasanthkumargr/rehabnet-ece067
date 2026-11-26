# RehabNet — AI-Driven Rehabilitation Exercise Monitoring and Correction

B.Tech project ECE 067 (mentor: Dr. Anusha K S). A patient exercises in front of an ordinary camera (laptop or
phone); RehabNet tracks the body with pose estimation, counts repetitions, scores each one, and gives spoken and
on-screen corrections. Doctors prescribe exercises remotely, follow adherence and progress, set personal baselines
and download PDF reports.

## Quick start (Windows)

```powershell
.\run_app.ps1
```

Then open **http://localhost:8000**. The first run creates the Python environment, builds the exercise
references and creates demo accounts — their sign-in details are written to `app\data\DEMO_ACCOUNTS.md`
(doctor `dr.demo`, patients `asha` with three weeks of demo history, and `ravi`).

To use a **phone or tablet** on the same Wi-Fi (browsers only allow the camera on https pages):

```powershell
.\run_app.ps1 -Https
```

and open the `https://<computer-ip>:8443` address it prints (accept the one-time certificate warning).

## What it does

| Who | Feature |
|---|---|
| Patient | today's exercises with demo video and instructions; live session: skeleton overlay, repetition counter, movement gauge, score per repetition, spoken corrections, camera-position and framing checks; session summary; progress chart; PDF report |
| Doctor | patients list with last score and weekly adherence; create patients (temporary password); exercise library with demos; prescribe exercises (side, sets, reps, sessions/week, notes); per-patient overview (score trends, range-of-motion trend, adherence); session detail (per-repetition scores, joint-angle traces, feedback given); personal baseline from a supervised session; PDF report |
| Security | bcrypt passwords, forced change of temporary passwords, lock-out after 5 failed logins, two-factor login (TOTP authenticator apps), signed expiring tokens, session data encrypted at rest (Fernet/AES), audit log of logins and record access, security headers; video never leaves the device — only body landmarks are sent |
| Demo / testing | any exercise can be run on a video file instead of the camera (frame-by-frame analysis); three REHAB24-6 demo videos are built in |

## How it works

```
browser: camera -> MediaPipe BlazePose (WASM/GPU) -> 33 3-D landmarks --WebSocket--> server
server : One-Euro smoothing -> joint angles (10 Hz) -> repetition counter -> DTW + NCC + stability scoring
         vs correct reference repetitions -> feedback messages -> encrypted session record
```

Exercise library: arm abduction, arm V-W raise, push-up against a table, leg abduction, lunge, squat
(references from REHAB24-6, both sides).

## Results (all on people the system never saw)

| Part | Result | Report |
|---|---|---|
| Pose detection | > 99.9 % of 84,532 frames | M1 |
| Joint angles vs OptiTrack | 14.8° mean error; range of motion per repetition r = 0.81 | M1 |
| Repetition counter | 92 % of reps found, 95 % precision; ±1 per 16-rep set (front/45°) | M2 |
| Correct vs incorrect (physiotherapist labels) | AUC 0.71 (annotated reps), 0.68 live pipeline | M2, M3 |
| UI-PRMD / KIMORE agreement | Spearman 0.64 / 0.34 | M2 |
| Personal baseline | combined references AUC 0.705 vs 0.701 general | M4 |
| Robustness (full model) | blur 13.5°, partly hidden 14.9°, far away 71 % detected / 23°, dark 12 % detected (app now warns "too dark") | M5 |
| Speed | lite model 46 ms/frame on a 2-core laptop CPU; server 0.4 ms/frame | M1/M5 |

## Milestones

| # | Content | Deliverables |
|---|---|---|
| 1 | Datasets, MediaPipe pipeline, normalisation, joint-angle validation | `deliverables/M1/` |
| 2 | Repetition counter, DTW + NCC scoring, multi-factor score, feedback | `deliverables/M2/` |
| 3 | Web platform: backend API, doctor portal, patient portal, live feedback, security | `deliverables/M3/` |
| 4 | Personal baselines, progress tracking, PDF reports | `deliverables/M4/` |
| 5 | Testing, robustness, speed, deployment, documentation | `deliverables/M5/` (+ `RehabNet_User_Guide`) |

All five milestones are done; each folder holds the report and slides as .docx/.pptx and .pdf.
`deliverables/Final/` holds a 12-slide final-review deck (HTML + PDF; also imported into Adobe Express).
Rebuild documents with `scripts/m*_deliverables.py`, then `tools/to_pdf.ps1 -Folders deliverables\M<n>` (needs Word/PowerPoint).

## Layout

```
app/server/      FastAPI app: main.py (API + WebSocket), engine.py (live session), security.py, db.py,
                 baseline.py, progress.py, reports.py
app/web/         single-page web app (no build step) + vendored MediaPipe for offline use + demo media
app/tests/       API / end-to-end tests  ->  .venv\Scripts\python -m pytest app/tests -q
app/build_references.py, app/seed.py
rehabnet/core/   pose, normalisation, angles, exercises, repetition counter, scoring, feedback
scripts/         experiments for each milestone (m1_* … m5_*) and Kaggle jobs
tools/           report/slide builder, Kaggle job packer, HTTPS certificate tool
```

## Deployment

The server is a standard ASGI app with a `Dockerfile` (no MediaPipe needed server-side):

```powershell
docker build -t rehabnet .
docker run -p 8000:8000 -v rehabnet-data:/data -e REHABNET_JWT_SECRET=<random> -e REHABNET_DATA_KEY=<fernet key> rehabnet
```

Run it behind HTTPS (cloud load balancer such as Google Cloud Run / AWS App Runner, or a reverse proxy with a real
certificate). Set `REHABNET_DB_URL` to a PostgreSQL URL for a managed database; SQLite is used otherwise.
The data key can be created with `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`.

## Data and licences

* REHAB24-6 — Černek, Sedmidubsky, Budikova, SISAP 2024, doi:10.5281/zenodo.13305826, CC BY-NC 4.0 (non-commercial).
  Downloaded to `data/rehab24_6/` by the Milestone 1 scripts; the demo videos in `app/web/media/samples/` come from it.
* KIMORE and UI-PRMD skeletons — Rehab-Pile release (Ismail-Fawaz et al., arXiv:2507.21018).
* MediaPipe Tasks (Apache 2.0) is vendored in `app/web/vendor/mediapipe/`.

RehabNet supports, but does not replace, clinical judgement.
