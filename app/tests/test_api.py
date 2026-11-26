"""End-to-end API tests (run: .venv\\Scripts\\python -m pytest app/tests -q).

Uses a throw-away database and data directory, so the real app data is never touched.
"""
import json
import os
import sys
import tempfile

import numpy as np
import pyotp
import pytest

TMP = tempfile.mkdtemp(prefix="rehabnet_test_")
os.environ["REHABNET_DATA_DIR"] = TMP
os.environ["REHABNET_DB_URL"] = "sqlite:///" + os.path.join(TMP, "t.db").replace("\\", "/")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from fastapi.testclient import TestClient  # noqa: E402

from app.server import security as sec  # noqa: E402
from app.server.db import SessionLocal, User  # noqa: E402
from app.server.db import Session as ExSession  # noqa: E402
from app.server.main import app  # noqa: E402

client = TestClient(app)
SQUAT = os.path.join(ROOT, "data", "mp_rehab24", "full", "Ex6", "PM_008_c17.npz")


def make_doctor(username, pw="doctor-pass-123"):
    s = SessionLocal()
    s.add(User(role="doctor", username=username, full_name="Dr " + username, pw_hash=sec.hash_password(pw)))
    s.commit()
    return pw


def login(username, pw):
    r = client.post("/api/auth/login", json=dict(username=username, password=pw))
    assert r.status_code == 200, r.text
    return {"Authorization": "Bearer " + r.json()["token"]}


@pytest.fixture(scope="module")
def doc():
    pw = make_doctor("doc1")
    return login("doc1", pw)


def test_health():
    assert client.get("/api/health").json()["ok"]


def test_login_wrong_password_and_lockout():
    make_doctor("doc_lock", "right-password-1")
    for _ in range(5):
        assert client.post("/api/auth/login", json=dict(username="doc_lock", password="nope")).status_code == 401
    r = client.post("/api/auth/login", json=dict(username="doc_lock", password="right-password-1"))
    assert r.status_code == 429  # locked for 5 minutes after 5 failures


def test_requires_login():
    assert client.get("/api/patients").status_code == 401
    assert client.get("/api/patients", headers={"Authorization": "Bearer junk"}).status_code == 401


def test_patient_flow_and_permissions(doc):
    r = client.post("/api/patients", json=dict(full_name="Test Patient", username="pat1", condition="knee"), headers=doc)
    assert r.status_code == 200
    pid, tmp_pw = r.json()["patient"]["id"], r.json()["temporary_password"]
    assert client.post("/api/patients", json=dict(full_name="Another Person", username="pat1"), headers=doc).status_code == 400
    a = client.post(f"/api/patients/{pid}/assignments", json=dict(exercise="squat", sets=1, reps=10), headers=doc)
    assert a.status_code == 200 and a.json()["side"] is None
    b = client.post(f"/api/patients/{pid}/assignments", json=dict(exercise="leg_abduction", side="l"), headers=doc)
    assert b.json()["side"] == "l"
    assert client.post(f"/api/patients/{pid}/assignments", json=dict(exercise="nope"), headers=doc).status_code == 400

    # patient signs in with the temporary password and must change it
    pat = client.post("/api/auth/login", json=dict(username="pat1", password=tmp_pw)).json()
    assert pat["user"]["must_change_pw"]
    ph = {"Authorization": "Bearer " + pat["token"]}
    assert client.post("/api/me/password", json=dict(old_password=tmp_pw, new_password="new-secret-99"), headers=ph).status_code == 200
    assert client.get("/api/me", headers=ph).json()["must_change_pw"] is False
    assert len(client.get("/api/my/assignments", headers=ph).json()) == 2

    # patients cannot use doctor endpoints; another doctor cannot see this patient
    assert client.get("/api/patients", headers=ph).status_code == 403
    other = login("doc2", make_doctor("doc2"))
    assert client.get(f"/api/patients/{pid}", headers=other).status_code == 404
    assert client.get(f"/api/patients/{pid}/report.pdf", headers=other).status_code == 404


def test_mfa(doc):
    make_doctor("doc_mfa", "mfa-pass-123")
    h = login("doc_mfa", "mfa-pass-123")
    setup = client.post("/api/me/mfa/setup", headers=h).json()
    assert setup["qr_png_b64"]
    assert client.post("/api/me/mfa/enable", json=dict(code="000000"), headers=h).status_code == 400
    code = pyotp.TOTP(setup["secret"]).now()
    assert client.post("/api/me/mfa/enable", json=dict(code=code), headers=h).status_code == 200
    r = client.post("/api/auth/login", json=dict(username="doc_mfa", password="mfa-pass-123")).json()
    assert r.get("mfa_required") and "token" not in r
    # the half-way token is not a login
    assert client.get("/api/me", headers={"Authorization": "Bearer " + r["mfa_token"]}).status_code == 401
    assert client.post("/api/auth/mfa/verify", json=dict(code="123456", mfa_token=r["mfa_token"])).status_code == 401
    ok = client.post("/api/auth/mfa/verify", json=dict(code=pyotp.TOTP(setup["secret"]).now(), mfa_token=r["mfa_token"]))
    assert ok.status_code == 200 and ok.json()["token"]


@pytest.mark.skipif(not os.path.exists(SQUAT), reason="REHAB24-6 MediaPipe output not available")
def test_live_session_websocket_and_reports(doc):
    r = client.post("/api/patients", json=dict(full_name="Live Patient", username="live1"), headers=doc).json()
    pid, pw = r["patient"]["id"], r["temporary_password"]
    aid = client.post(f"/api/patients/{pid}/assignments", json=dict(exercise="squat", sets=2, reps=10), headers=doc).json()["id"]
    tok = client.post("/api/auth/login", json=dict(username="live1", password=pw)).json()["token"]
    z = np.load(SQUAT)
    events = []
    with client.websocket_connect("/ws/session") as ws:
        ws.send_text(json.dumps(dict(type="start", token=tok, assignment_id=aid, source="video")))
        ready = json.loads(ws.receive_text())
        assert ready["type"] == "ready" and ready["target"] == 20
        for f, w in zip(z["frame"][:2400], z["world"][:2400]):   # ~80 s of a 30 fps recording
            ws.send_text(json.dumps(dict(type="frame", t=float(f) * 1000 / 30, w=None if np.isnan(w).all() else w.tolist(), view="front")))
        ws.send_text(json.dumps(dict(type="end")))
        while True:
            e = json.loads(ws.receive_text())
            events.append(e)
            if e["type"] == "summary":
                break
    counted = [e for e in events if e["type"] == "rep_counted"]
    summary = events[-1]
    assert len(counted) >= 8, f"only {len(counted)} reps counted"
    assert summary["reps"] >= 8 and 0 <= summary["mean_score"] <= 100
    # stored encrypted; visible to the patient's doctor
    s = SessionLocal()
    row = s.get(ExSession, summary["session_id"])
    assert row.data_enc and b"feedback" not in row.data_enc
    d = client.get(f"/api/sessions/{row.id}", headers=doc).json()
    assert len(d["data"]["reps"]) == summary["reps"] and d["data"]["reps"][0]["feedback"]
    # progress + PDF
    prog = client.get(f"/api/patients/{pid}/progress", headers=doc).json()
    assert prog["total_sessions"] == 1
    pdf = client.get(f"/api/patients/{pid}/report.pdf", headers=doc)
    assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"
    # baseline from this session, then a second session uses it
    bl = client.post(f"/api/patients/{pid}/baseline/{row.id}", headers=doc)
    assert bl.status_code == 200, bl.text
    with client.websocket_connect("/ws/session") as ws:
        ws.send_text(json.dumps(dict(type="start", token=tok, assignment_id=aid)))
        assert json.loads(ws.receive_text())["personal"] is True
        ws.send_text(json.dumps(dict(type="end")))
        assert json.loads(ws.receive_text())["type"] == "summary"


def test_websocket_rejects_bad_token():
    with client.websocket_connect("/ws/session") as ws:
        ws.send_text(json.dumps(dict(type="start", token="junk", exercise="squat")))
        assert json.loads(ws.receive_text())["type"] == "error"
