"""RehabNet web server (FastAPI).

Run:  .venv\\Scripts\\python -m uvicorn app.server.main:app --port 8000      (from the project root)
Then open http://localhost:8000
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sys

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from rehabnet.core.exercises import EXERCISES  # noqa: E402

from . import security as sec  # noqa: E402
from .config import WEB_DIR  # noqa: E402
from .db import Assignment, AuditLog, Baseline, SessionLocal, User, init_db, now  # noqa: E402
from .db import Session as ExSession  # noqa: E402
from .engine import LiveSession  # noqa: E402

app = FastAPI(title="RehabNet", version="1.0")
init_db()


def _cleanup_unfinished():
    """Sessions left open by a server crash/stop (no repetitions saved) are removed at start-up."""
    s = SessionLocal()
    s.query(ExSession).filter(ExSession.ended.is_(None), ExSession.started < now() - dt.timedelta(minutes=30)).delete()
    s.commit()
    s.close()


_cleanup_unfinished()

RECOMMENDED_VIEW = {"arm_abduction": "front", "arm_vw": "front", "pushup_table": "side or 45°",
                    "leg_abduction": "front", "lunge": "45° or side", "squat": "front or 45°"}


# ------------------------------------------------------------------ helpers
def db():
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


def audit(s, user_id, action, detail=""):
    s.add(AuditLog(user_id=user_id, action=action, detail=detail[:500]))
    s.commit()


def current_user(request: Request, s=Depends(db)) -> User:
    auth = request.headers.get("authorization", "")
    tok = auth[7:] if auth.lower().startswith("bearer ") else request.cookies.get("rn_token", "")
    data = sec.read_token(tok) if tok else None
    if not data or data.get("stage") != "full":
        raise HTTPException(401, "Please log in")
    u = s.get(User, int(data["sub"]))
    if not u:
        raise HTTPException(401, "Please log in")
    return u


def doctor_only(u: User = Depends(current_user)) -> User:
    if u.role != "doctor":
        raise HTTPException(403, "Doctors only")
    return u


def patient_of(s, doctor: User, pid: int) -> User:
    p = s.get(User, pid)
    if not p or p.role != "patient" or p.doctor_id != doctor.id:
        raise HTTPException(404, "Patient not found")
    return p


def user_json(u: User):
    return dict(id=u.id, role=u.role, username=u.username, full_name=u.full_name, mfa_enabled=u.mfa_enabled,
                must_change_pw=u.must_change_pw, condition=u.condition, birth_year=u.birth_year)


def assignment_json(a: Assignment):
    ex = EXERCISES[a.exercise]
    return dict(id=a.id, exercise=a.exercise, name=ex.name, side=a.side, sets=a.sets, reps=a.reps,
                per_week=a.per_week, notes=a.notes, active=a.active, created=a.created.isoformat())


def session_json(x: ExSession, detail=False):
    d = dict(id=x.id, patient_id=x.patient_id, exercise=x.exercise, name=EXERCISES[x.exercise].name, side=x.side,
             started=x.started.isoformat() + "Z", ended=x.ended.isoformat() + "Z" if x.ended else None,
             reps=x.reps, target_reps=x.target_reps, mean_score=x.mean_score, consistency=x.consistency,
             rom_score=x.rom_score, stability_score=x.stability_score, weakest=x.weakest,
             camera_view=x.camera_view, source=x.source, assignment_id=x.assignment_id)
    if detail:
        d["data"] = sec.decrypt_json(x.data_enc)
    return d


# ------------------------------------------------------------------ auth
class LoginIn(BaseModel):
    username: str
    password: str


class CodeIn(BaseModel):
    code: str
    mfa_token: str | None = None


@app.post("/api/auth/login")
def login(body: LoginIn, s=Depends(db)):
    u = s.query(User).filter(User.username == body.username.strip().lower()).first()
    if u and u.locked_until and u.locked_until > now():
        raise HTTPException(429, "Too many failed attempts. Try again in a few minutes.")
    if not u or not sec.check_password(body.password, u.pw_hash):
        if u:
            u.failed_logins += 1
            if u.failed_logins >= 5:
                u.locked_until = now() + dt.timedelta(minutes=5)
                u.failed_logins = 0
            s.commit()
            audit(s, u.id, "login_failed")
        raise HTTPException(401, "Wrong username or password")
    u.failed_logins = 0
    s.commit()
    if u.mfa_enabled:
        return dict(mfa_required=True, mfa_token=sec.make_token(u.id, u.role, stage="mfa"))
    audit(s, u.id, "login")
    return dict(token=sec.make_token(u.id, u.role), user=user_json(u))


@app.post("/api/auth/mfa/verify")
def mfa_verify(body: CodeIn, s=Depends(db)):
    data = sec.read_token(body.mfa_token or "")
    if not data or data.get("stage") != "mfa":
        raise HTTPException(401, "Login expired, start again")
    u = s.get(User, int(data["sub"]))
    if not sec.check_totp(u.totp_secret, body.code):
        audit(s, u.id, "mfa_failed")
        raise HTTPException(401, "Wrong code")
    audit(s, u.id, "login_mfa")
    return dict(token=sec.make_token(u.id, u.role), user=user_json(u))


@app.get("/api/me")
def me(u: User = Depends(current_user)):
    return user_json(u)


class PwIn(BaseModel):
    old_password: str
    new_password: str = Field(min_length=8)


@app.post("/api/me/password")
def change_password(body: PwIn, u: User = Depends(current_user), s=Depends(db)):
    u = s.get(User, u.id)
    if not sec.check_password(body.old_password, u.pw_hash):
        raise HTTPException(400, "Current password is wrong")
    u.pw_hash = sec.hash_password(body.new_password)
    u.must_change_pw = False
    s.commit()
    audit(s, u.id, "password_changed")
    return dict(ok=True)


@app.post("/api/me/mfa/setup")
def mfa_setup(u: User = Depends(current_user), s=Depends(db)):
    u = s.get(User, u.id)
    u.totp_secret = sec.new_totp_secret()
    u.mfa_enabled = False
    s.commit()
    uri = sec.totp_uri(u.totp_secret, u.username)
    return dict(secret=u.totp_secret, uri=uri, qr_png_b64=sec.totp_qr_png_b64(uri))


@app.post("/api/me/mfa/enable")
def mfa_enable(body: CodeIn, u: User = Depends(current_user), s=Depends(db)):
    u = s.get(User, u.id)
    if not sec.check_totp(u.totp_secret, body.code):
        raise HTTPException(400, "Wrong code - check the time on your phone and try again")
    u.mfa_enabled = True
    s.commit()
    audit(s, u.id, "mfa_enabled")
    return dict(ok=True)


@app.post("/api/me/mfa/disable")
def mfa_disable(body: CodeIn, u: User = Depends(current_user), s=Depends(db)):
    u = s.get(User, u.id)
    if u.mfa_enabled and not sec.check_totp(u.totp_secret, body.code):
        raise HTTPException(400, "Wrong code")
    u.mfa_enabled, u.totp_secret = False, None
    s.commit()
    audit(s, u.id, "mfa_disabled")
    return dict(ok=True)


# ------------------------------------------------------------------ exercise library
@app.get("/api/exercises")
def exercises():
    media = os.path.join(WEB_DIR, "media")
    out = []
    for k, e in EXERCISES.items():
        clip = next((f"/media/{f}" for f in (f"{k}.webm", f"{k}.gif", f"{k}.mp4") if os.path.exists(os.path.join(media, f))), None)
        out.append(dict(key=k, name=e.name, instructions=e.instructions, sided=e.sided, demo=clip,
                        view=RECOMMENDED_VIEW.get(k, "front")))
    return out


@app.get("/api/samples")
def samples():
    """Demo videos (REHAB24-6 recordings, CC BY-NC 4.0) that can replace the camera for demonstrations."""
    f = os.path.join(WEB_DIR, "media", "samples", "samples.json")
    if not os.path.exists(f):
        return []
    return [dict(url=f"/media/samples/{x['file']}", exercise=x["exercise"], side=x.get("side"), label=x["label"])
            for x in json.load(open(f))]


# ------------------------------------------------------------------ doctor: patients and assignments
class PatientIn(BaseModel):
    full_name: str = Field(min_length=2, max_length=120)
    username: str = Field(min_length=3, max_length=64, pattern=r"^[a-zA-Z0-9_.-]+$")
    condition: str | None = None
    birth_year: int | None = None


@app.get("/api/patients")
def list_patients(d: User = Depends(doctor_only), s=Depends(db)):
    out = []
    for p in s.query(User).filter(User.role == "patient", User.doctor_id == d.id).order_by(User.full_name):
        ss = s.query(ExSession).filter(ExSession.patient_id == p.id, ExSession.ended.isnot(None)).order_by(ExSession.started.desc()).all()
        week = [x for x in ss if x.started > now() - dt.timedelta(days=7)]
        act = s.query(Assignment).filter(Assignment.patient_id == p.id, Assignment.active.is_(True)).all()
        planned = sum(a.per_week for a in act)
        out.append(dict(**user_json(p), assignments=len(act), sessions=len(ss),
                        last_session=ss[0].started.isoformat() + "Z" if ss else None,
                        last_score=ss[0].mean_score if ss else None,
                        adherence_week=(min(100, round(100 * len(week) / planned)) if planned else None)))
    return out


@app.post("/api/patients")
def create_patient(body: PatientIn, d: User = Depends(doctor_only), s=Depends(db)):
    uname = body.username.strip().lower()
    if s.query(User).filter(User.username == uname).first():
        raise HTTPException(400, "That username is taken")
    pw = sec.new_temp_password()
    p = User(role="patient", username=uname, full_name=body.full_name.strip(), pw_hash=sec.hash_password(pw),
             must_change_pw=True, doctor_id=d.id, condition=body.condition, birth_year=body.birth_year)
    s.add(p)
    s.commit()
    audit(s, d.id, "patient_created", uname)
    return dict(patient=user_json(p), temporary_password=pw)


@app.get("/api/patients/{pid}")
def get_patient(pid: int, d: User = Depends(doctor_only), s=Depends(db)):
    p = patient_of(s, d, pid)
    asg = s.query(Assignment).filter(Assignment.patient_id == p.id).order_by(Assignment.created.desc()).all()
    ss = s.query(ExSession).filter(ExSession.patient_id == p.id, ExSession.ended.isnot(None)).order_by(ExSession.started.desc()).all()
    return dict(patient=user_json(p), assignments=[assignment_json(a) for a in asg], sessions=[session_json(x) for x in ss])


@app.post("/api/patients/{pid}/reset_password")
def reset_password(pid: int, d: User = Depends(doctor_only), s=Depends(db)):
    p = patient_of(s, d, pid)
    pw = sec.new_temp_password()
    p.pw_hash, p.must_change_pw, p.mfa_enabled, p.totp_secret = sec.hash_password(pw), True, False, None
    s.commit()
    audit(s, d.id, "patient_password_reset", p.username)
    return dict(temporary_password=pw)


class AssignIn(BaseModel):
    exercise: str
    side: str | None = None
    sets: int = Field(2, ge=1, le=10)
    reps: int = Field(10, ge=1, le=50)
    per_week: int = Field(5, ge=1, le=14)
    notes: str = ""


@app.post("/api/patients/{pid}/assignments")
def add_assignment(pid: int, body: AssignIn, d: User = Depends(doctor_only), s=Depends(db)):
    p = patient_of(s, d, pid)
    if body.exercise not in EXERCISES:
        raise HTTPException(400, "Unknown exercise")
    side = body.side if EXERCISES[body.exercise].sided and body.side in ("l", "r") else None
    if EXERCISES[body.exercise].sided and side is None:
        side = "r"
    a = Assignment(patient_id=p.id, doctor_id=d.id, exercise=body.exercise, side=side, sets=body.sets, reps=body.reps,
                   per_week=body.per_week, notes=body.notes[:1000])
    s.add(a)
    s.commit()
    audit(s, d.id, "assignment_added", f"{p.username}:{body.exercise}")
    return assignment_json(a)


class AssignPatch(BaseModel):
    sets: int | None = Field(None, ge=1, le=10)
    reps: int | None = Field(None, ge=1, le=50)
    per_week: int | None = Field(None, ge=1, le=14)
    notes: str | None = None
    active: bool | None = None


@app.patch("/api/assignments/{aid}")
def edit_assignment(aid: int, body: AssignPatch, d: User = Depends(doctor_only), s=Depends(db)):
    a = s.get(Assignment, aid)
    if not a or a.doctor_id != d.id:
        raise HTTPException(404, "Not found")
    for k, v in body.model_dump(exclude_none=True).items():
        setattr(a, k, v)
    s.commit()
    return assignment_json(a)


@app.get("/api/sessions/{sid}")
def get_session(sid: int, u: User = Depends(current_user), s=Depends(db)):
    x = s.get(ExSession, sid)
    if not x:
        raise HTTPException(404, "Not found")
    owner = s.get(User, x.patient_id)
    if not (u.id == x.patient_id or (u.role == "doctor" and owner.doctor_id == u.id)):
        raise HTTPException(404, "Not found")
    if u.role == "doctor":
        audit(s, u.id, "session_viewed", str(sid))
    return session_json(x, detail=True)


@app.get("/api/patients/{pid}/progress")
def progress(pid: int, u: User = Depends(current_user), s=Depends(db)):
    if u.role == "doctor":
        patient_of(s, u, pid)
    elif u.id != pid:
        raise HTTPException(404, "Not found")
    from .progress import progress_report
    return progress_report(s, pid)


@app.get("/api/patients/{pid}/report.pdf")
def report_pdf(pid: int, u: User = Depends(current_user), s=Depends(db)):
    if u.role == "doctor":
        p = patient_of(s, u, pid)
    elif u.id == pid:
        p = s.get(User, pid)
    else:
        raise HTTPException(404, "Not found")
    from .reports import patient_pdf
    pdf = patient_pdf(s, p)
    audit(s, u.id, "report_pdf", p.username)
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": f'inline; filename="rehabnet_{p.username}.pdf"'})


# ------------------------------------------------------------------ patient
@app.get("/api/my/assignments")
def my_assignments(u: User = Depends(current_user), s=Depends(db)):
    asg = s.query(Assignment).filter(Assignment.patient_id == u.id, Assignment.active.is_(True)).all()
    out = []
    for a in asg:
        done = s.query(ExSession).filter(ExSession.assignment_id == a.id, ExSession.ended.isnot(None),
                                         ExSession.started > now() - dt.timedelta(days=7)).count()
        today = s.query(ExSession).filter(ExSession.assignment_id == a.id, ExSession.ended.isnot(None),
                                          ExSession.started > now().replace(hour=0, minute=0, second=0)).count()
        out.append(dict(**assignment_json(a), done_this_week=done, done_today=today))
    return out


@app.get("/api/my/sessions")
def my_sessions(u: User = Depends(current_user), s=Depends(db)):
    ss = s.query(ExSession).filter(ExSession.patient_id == u.id, ExSession.ended.isnot(None)).order_by(ExSession.started.desc()).limit(100)
    return [session_json(x) for x in ss]


# ------------------------------------------------------------------ live session (WebSocket)
@app.websocket("/ws/session")
async def ws_session(ws: WebSocket):
    """Protocol (JSON text frames):
    client -> {"type":"start","token":..., "assignment_id":.. | "exercise":..,"side":..,"target":.., "source":"camera"|"video"}
    client -> {"type":"frame","t":ms,"w":[[x,y,z]*33] | null,"view":"front"|...}   (MediaPipe world landmarks)
    client -> {"type":"end"}
    server -> {"type":"ready",...} / {"type":"tick"|"rep_counted"|"rep_scored"|"no_person",...} / {"type":"summary",...}
    """
    await ws.accept()
    s = SessionLocal()
    live = rec = None
    views = {}
    try:
        msg = json.loads(await ws.receive_text())
        data = sec.read_token(msg.get("token", ""))
        if msg.get("type") != "start" or not data or data.get("stage") != "full":
            await ws.send_text(json.dumps(dict(type="error", message="Please log in again")))
            return
        u = s.get(User, int(data["sub"]))
        a = None
        if msg.get("assignment_id"):
            a = s.get(Assignment, int(msg["assignment_id"]))
            if not a or a.patient_id != u.id:
                await ws.send_text(json.dumps(dict(type="error", message="Exercise not found")))
                return
            exn, side, target = a.exercise, a.side, a.reps * a.sets
        else:
            exn, side, target = msg.get("exercise"), msg.get("side"), int(msg.get("target", 10))
            if exn not in EXERCISES:
                await ws.send_text(json.dumps(dict(type="error", message="Unknown exercise")))
                return
        personal = pth = None
        b = s.query(Baseline).filter(Baseline.patient_id == u.id, Baseline.exercise == exn, Baseline.side == side).order_by(Baseline.created.desc()).first()
        if b is not None:
            from .baseline import model_from_baseline, personal_thresholds
            bdata = sec.decrypt_json(b.data_enc)
            personal, pth = model_from_baseline(exn, side, bdata), personal_thresholds(exn, side, bdata)
        live = LiveSession(exn, side, target, personal_model=personal, thresholds=pth)
        rec = ExSession(patient_id=u.id, assignment_id=a.id if a else None, exercise=exn, side=live.side,
                        target_reps=target, source=msg.get("source", "camera")[:10])
        s.add(rec)
        s.commit()
        await ws.send_text(json.dumps(dict(type="ready", session_id=rec.id, exercise=exn, name=EXERCISES[exn].name,
                                           side=live.side, target=target, personal=personal is not None)))
        last_tick = 0.0
        while True:
            m = json.loads(await ws.receive_text())
            if m.get("type") == "frame":
                if m.get("view"):
                    views[m["view"]] = views.get(m["view"], 0) + 1
                evs = live.add_frame(float(m.get("t", 0)), m.get("w"))
                out = [e for e in evs if e["type"] != "tick"]
                ticks = [e for e in evs if e["type"] == "tick"]
                if ticks and float(m.get("t", 0)) - last_tick >= 100:
                    out.append(ticks[-1])
                    last_tick = float(m.get("t", 0))
                for e in out:
                    await ws.send_text(json.dumps(e))
            elif m.get("type") == "end":
                break
    except WebSocketDisconnect:
        pass
    finally:
        if live is not None and rec is not None:
            n_before = len(live.results)
            res = live.finish()
            sm = res["summary"]
            rec.ended = now()
            rec.reps = sm.get("reps", 0)
            rec.mean_score = round(sm["mean_score"], 1) if sm.get("reps") else None
            rec.consistency = round(sm["consistency"], 1) if sm.get("reps") else None
            f = sm.get("factors", {})
            rec.rom_score = round(f["rom"], 1) if f.get("rom") is not None else None
            rec.stability_score = round(f["stability"], 1) if f.get("stability") is not None else None
            rec.weakest = sm.get("weakest")
            rec.camera_view = max(views, key=views.get) if views else None
            rec.data_enc = sec.encrypt_json(res)
            if rec.reps == 0:
                s.delete(rec)
            s.commit()
            try:
                for r in res["reps"][n_before:]:   # repetitions scored only when the session ended
                    await ws.send_text(json.dumps(dict(type="rep_scored", **r)))
                await ws.send_text(json.dumps(dict(type="summary", session_id=rec.id if rec.reps else None, **sm)))
                await ws.close()
            except Exception:
                pass
        s.close()


# ------------------------------------------------------------------ baseline (Milestone 4)
@app.post("/api/patients/{pid}/baseline/{sid}")
def set_baseline(pid: int, sid: int, d: User = Depends(doctor_only), s=Depends(db)):
    """Doctor marks a supervised session as the patient's personal reference for that exercise."""
    p = patient_of(s, d, pid)
    x = s.get(ExSession, sid)
    if not x or x.patient_id != p.id:
        raise HTTPException(404, "Session not found")
    from .baseline import baseline_from_session
    data = baseline_from_session(sec.decrypt_json(x.data_enc), x.exercise, x.side)
    if data is None:
        raise HTTPException(400, "Not enough good repetitions in this session to use it as a baseline (need 5 or more)")
    s.add(Baseline(patient_id=p.id, exercise=x.exercise, side=x.side, data_enc=sec.encrypt_json(data)))
    s.commit()
    audit(s, d.id, "baseline_set", f"{p.username}:{x.exercise}:{sid}")
    return dict(ok=True, reps=len(data["reps"]))


@app.delete("/api/patients/{pid}/baseline/{exercise}")
def clear_baseline(pid: int, exercise: str, d: User = Depends(doctor_only), s=Depends(db)):
    p = patient_of(s, d, pid)
    s.query(Baseline).filter(Baseline.patient_id == p.id, Baseline.exercise == exercise).delete()
    s.commit()
    return dict(ok=True)


# ------------------------------------------------------------------ static web app
@app.get("/api/health")
def health():
    return dict(ok=True, time=now().isoformat())


@app.middleware("http")
async def security_headers(request: Request, call_next):
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "no-referrer"
    resp.headers["Permissions-Policy"] = "camera=(self)"
    if request.url.path.startswith("/api/"):
        resp.headers["Cache-Control"] = "no-store"
    elif request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"   # always revalidate app code so updates reach users
    return resp


@app.exception_handler(HTTPException)
async def http_err(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content=dict(detail=exc.detail))


app.mount("/vendor", StaticFiles(directory=os.path.join(WEB_DIR, "vendor")), name="vendor")
app.mount("/media", StaticFiles(directory=os.path.join(WEB_DIR, "media")), name="media")
app.mount("/static", StaticFiles(directory=os.path.join(WEB_DIR, "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB_DIR, "index.html"), headers={"Cache-Control": "no-cache"})
