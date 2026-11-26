"""Create demo accounts and demo history (run once:  .venv\\Scripts\\python -P app/seed.py).

Accounts are written to app/data/DEMO_ACCOUNTS.md (random passwords, local demo only).
The demo patient's history is produced by replaying real REHAB24-6 camera recordings through the live engine with
back-dated timestamps, so the dashboards, trends and PDF report have something to show. It is labelled as demo data.
"""
import datetime as dt
import os
import random
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from app.server import security as sec  # noqa: E402
from app.server.config import DATA_DIR  # noqa: E402
from app.server.db import Assignment, SessionLocal, User, init_db  # noqa: E402
from app.server.db import Session as ExSession  # noqa: E402
from app.server.engine import LiveSession  # noqa: E402

MP = os.path.join(ROOT, "data", "mp_rehab24", "full_every3")


def replay(exercise, side, path, target):
    z = np.load(path)
    live = LiveSession(exercise, side, target)
    for f, w in zip(z["frame"], z["world"]):
        live.add_frame(float(f) * 1000 / 30, None if np.isnan(w).all() else w.tolist())
    return live.finish()


def main():
    init_db()
    s = SessionLocal()
    if s.query(User).filter(User.username == "dr.demo").first():
        print("demo data already present - delete app/data/rehabnet.db to start fresh")
        return
    pw = {k: sec.new_temp_password() for k in ("dr.demo", "asha", "ravi")}
    doc = User(role="doctor", username="dr.demo", full_name="Dr. Demo Physio", pw_hash=sec.hash_password(pw["dr.demo"]))
    s.add(doc); s.commit()
    asha = User(role="patient", username="asha", full_name="Asha Raman (demo)", pw_hash=sec.hash_password(pw["asha"]),
                doctor_id=doc.id, condition="Knee rehabilitation - DEMO data replayed from REHAB24-6", birth_year=1968)
    ravi = User(role="patient", username="ravi", full_name="Ravi Kumar (demo)", pw_hash=sec.hash_password(pw["ravi"]),
                doctor_id=doc.id, condition="Shoulder rehabilitation", birth_year=1975)
    s.add_all([asha, ravi]); s.commit()
    t0 = dt.datetime.utcnow() - dt.timedelta(days=21)
    a_sq = Assignment(patient_id=asha.id, doctor_id=doc.id, exercise="squat", sets=2, reps=10, per_week=4,
                      notes="Keep your chest up. Stop if your knee hurts.", created=t0)
    a_la = Assignment(patient_id=asha.id, doctor_id=doc.id, exercise="leg_abduction", side="l", sets=2, reps=8, per_week=3,
                      notes="Slow and controlled; do not lean sideways.", created=t0)
    a_ab = Assignment(patient_id=ravi.id, doctor_id=doc.id, exercise="arm_abduction", side="r", sets=2, reps=8, per_week=5,
                      notes="Raise only as high as comfortable.")
    s.add_all([a_sq, a_la, a_ab]); s.commit()

    # demo history for Asha: replay recordings, then order sessions by score so the demo shows a recovery trend
    rng = random.Random(1)
    plans = [(a_sq, "squat", None, 6, None),
             (a_la, "leg_abduction", "l", 4, None)]
    import glob
    for a, exn, side, ex_id, vids in plans:
        files = sorted(glob.glob(os.path.join(MP, f"Ex{ex_id}", "*_c17.npz")))
        if vids:
            files = [f for f in files if os.path.basename(f).split("_c")[0] in vids] or files
        if side == "l":
            import pandas as pd
            seg = pd.read_csv(os.path.join(ROOT, "data", "rehab24_6", "Segmentation.csv"), sep=";")
            left = set(seg[(seg.exercise_id == 4) & (seg.exercise_subtype == "left leg")].video_id)
            files = [f for f in files if os.path.basename(f).split("_c")[0] in left]
        results = []
        for f in files[:9]:
            res = replay(exn, side, f, a.sets * a.reps)
            if res["summary"].get("reps"):
                results.append(res)
            print(exn, os.path.basename(f), res["summary"].get("reps"), round(res["summary"].get("mean_score", 0)))
        results.sort(key=lambda r: r["summary"]["mean_score"])
        n = len(results)
        for i, res in enumerate(results):
            day = t0 + dt.timedelta(days=i * 19.5 / max(1, n - 1)) + dt.timedelta(hours=rng.randint(0, 6), minutes=rng.randint(0, 59))
            day = min(day, dt.datetime.utcnow() - dt.timedelta(hours=2))
            sm = res["summary"]
            f = sm.get("factors", {})
            s.add(ExSession(patient_id=asha.id, assignment_id=a.id, exercise=exn, side=side, started=day,
                            ended=day + dt.timedelta(seconds=res["duration_s"]), target_reps=a.sets * a.reps, reps=sm["reps"],
                            mean_score=round(sm["mean_score"], 1), consistency=round(sm["consistency"], 1),
                            rom_score=f.get("rom"), stability_score=f.get("stability"), weakest=sm.get("weakest"),
                            camera_view="front", source="video", data_enc=sec.encrypt_json(res)))
        s.commit()
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(os.path.join(DATA_DIR, "DEMO_ACCOUNTS.md"), "w") as fh:
        fh.write("# RehabNet demo accounts (local demo only)\n\n| Role | Username | Password |\n|---|---|---|\n")
        fh.write(f"| Doctor | dr.demo | {pw['dr.demo']} |\n| Patient (with 3 weeks of demo history) | asha | {pw['asha']} |\n"
                 f"| Patient (new) | ravi | {pw['ravi']} |\n\nPatients are asked to change their password at first sign-in "
                 "only when created by a doctor in the app; these demo accounts are not.\n")
    print("accounts written to", os.path.join(DATA_DIR, "DEMO_ACCOUNTS.md"))


if __name__ == "__main__":
    main()
