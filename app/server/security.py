"""Passwords (bcrypt), login tokens (JWT), two-factor codes (TOTP) and encryption of stored session data (Fernet).

Secrets are generated on first start and kept in app/data/secrets.json (never committed). In deployment they can be
supplied through the environment variables REHABNET_JWT_SECRET and REHABNET_DATA_KEY instead.
"""
from __future__ import annotations

import base64
import io
import json
import os
import secrets
import time

import bcrypt
import jwt
import pyotp
import qrcode
from cryptography.fernet import Fernet

from .config import DATA_DIR

_SECRETS_FILE = os.path.join(DATA_DIR, "secrets.json")


def _load_secrets():
    os.makedirs(DATA_DIR, exist_ok=True)
    s = {}
    if os.path.exists(_SECRETS_FILE):
        s = json.load(open(_SECRETS_FILE))
    changed = False
    if "jwt" not in s:
        s["jwt"] = secrets.token_urlsafe(48); changed = True
    if "data_key" not in s:
        s["data_key"] = Fernet.generate_key().decode(); changed = True
    if changed:
        with open(_SECRETS_FILE, "w") as f:
            json.dump(s, f)
    return s


_S = _load_secrets()
JWT_SECRET = os.environ.get("REHABNET_JWT_SECRET", _S["jwt"])
_FERNET = Fernet(os.environ.get("REHABNET_DATA_KEY", _S["data_key"]).encode())
TOKEN_HOURS = 12


# ------------------------------------------------------------------ passwords
def hash_password(pw: str) -> str:
    return bcrypt.hashpw(pw.encode(), bcrypt.gensalt(rounds=12)).decode()


def check_password(pw: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(pw.encode(), hashed.encode())
    except ValueError:
        return False


def new_temp_password() -> str:
    words = ["river", "maple", "stone", "cloud", "tiger", "lotus", "amber", "orbit", "cedar", "delta"]
    return f"{secrets.choice(words)}-{secrets.choice(words)}-{secrets.randbelow(900) + 100}"


# ------------------------------------------------------------------ tokens
def make_token(user_id: int, role: str, stage: str = "full", hours: float = TOKEN_HOURS) -> str:
    """stage 'full' = logged in; stage 'mfa' = password ok, waiting for the 6-digit code (valid 5 minutes)."""
    now = int(time.time())
    exp = now + (300 if stage == "mfa" else int(hours * 3600))
    return jwt.encode({"sub": str(user_id), "role": role, "stage": stage, "iat": now, "exp": exp}, JWT_SECRET, algorithm="HS256")


def read_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.PyJWTError:
        return None


# ------------------------------------------------------------------ two-factor (TOTP, e.g. Google Authenticator)
def new_totp_secret() -> str:
    return pyotp.random_base32()


def totp_uri(secret: str, username: str) -> str:
    return pyotp.TOTP(secret).provisioning_uri(name=username, issuer_name="RehabNet")


def totp_qr_png_b64(uri: str) -> str:
    img = qrcode.make(uri)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def check_totp(secret: str, code: str) -> bool:
    return bool(secret) and pyotp.TOTP(secret).verify(str(code).strip(), valid_window=1)


# ------------------------------------------------------------------ encryption at rest
def encrypt_json(obj) -> bytes:
    return _FERNET.encrypt(json.dumps(obj, separators=(",", ":")).encode())


def decrypt_json(blob: bytes):
    return json.loads(_FERNET.decrypt(blob).decode()) if blob else None
