"""Milestone 3-5 figures: system architecture and live-session data flow."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "results", "m3")
NAVY, TEAL, GREY, LIGHT, ORANGE = "#122B4A", "#0F766E", "#6B7280", "#E6F4F2", "#D97706"


def box(ax, x, y, w, h, title, lines=(), fc=NAVY, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="none"))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", color=tc, fontsize=11.5, fontweight="bold")
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 0.62 - i * 0.3, ln, ha="center", va="top", color=tc, fontsize=9)


def arrow(ax, a, b, text="", color=GREY):
    ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12, text, ha="center", fontsize=8, color=color)


def architecture():
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.axis("off"); ax.set_xlim(0, 13); ax.set_ylim(0, 6.2)
    ax.text(1.65, 6.0, "Patient device (phone / laptop browser)", ha="center", fontsize=10, color=GREY)
    box(ax, 0.1, 3.3, 3.1, 2.5, "Patient portal", ["camera / video file", "MediaPipe BlazePose (WASM, GPU)",
                                                    "skeleton overlay, counter,", "gauge, spoken feedback"], fc=TEAL)
    box(ax, 0.1, 0.4, 3.1, 2.4, "Doctor portal", ["patients, prescriptions", "progress & adherence charts",
                                                   "session detail, baselines", "PDF reports, 2-factor login"], fc=TEAL)
    ax.text(7.25, 6.0, "RehabNet server (FastAPI)", ha="center", fontsize=10, color=GREY)
    box(ax, 5.0, 3.3, 4.5, 2.5, "Live engine (WebSocket)", ["One-Euro smoothing -> joint angles 10 Hz",
                                                            "repetition counter (M2)", "DTW + NCC + stability score",
                                                            "feedback messages"])
    box(ax, 5.0, 0.4, 4.5, 2.4, "REST API", ["auth: bcrypt, JWT, TOTP, lock-out", "patients, assignments, sessions",
                                             "progress, baselines, PDF reports", "audit log, security headers"])
    box(ax, 10.3, 3.3, 2.5, 2.5, "Exercise refs", ["REHAB24-6 correct", "repetitions,", "counter thresholds",
                                                   "(+ personal baseline)"], fc="#334155")
    box(ax, 10.3, 0.4, 2.5, 2.4, "Database", ["SQLite / PostgreSQL", "session data", "encrypted (Fernet)", ""], fc="#334155")
    arrow(ax, (3.2, 4.9), (5.0, 4.9), "33 landmarks, 15/s", TEAL)
    arrow(ax, (5.0, 4.2), (3.2, 4.2), "count, score, feedback", ORANGE)
    arrow(ax, (3.2, 1.6), (5.0, 1.6), "HTTPS / JSON")
    arrow(ax, (9.5, 4.55), (10.3, 4.55))
    arrow(ax, (9.5, 1.6), (10.3, 1.6))
    arrow(ax, (7.25, 3.3), (7.25, 2.8), "")
    ax.text(1.65, 0.05, "video never leaves the device", ha="center", fontsize=9, color=TEAL, style="italic")
    os.makedirs(R, exist_ok=True)
    fig.savefig(os.path.join(R, "fig_architecture.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def sequence():
    """Live session message sequence between browser and server."""
    fig, ax = plt.subplots(figsize=(12, 6.4))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 6.6)
    cols = [(1.2, "Patient browser", TEAL), (6.0, "RehabNet server", NAVY), (10.6, "Database", "#334155")]
    for x, name, c in cols:
        ax.add_patch(FancyBboxPatch((x - 1.0, 5.9), 2.0, 0.55, boxstyle="round,pad=0.02,rounding_size=0.08", fc=c, ec="none"))
        ax.text(x, 6.17, name, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        ax.plot([x, x], [0.2, 5.9], color="#CBD5E1", lw=1.5, ls="--", zorder=0)
    msgs = [
        (1.2, 6.0, "start (token, prescription)", TEAL), (6.0, 1.2, "ready (target reps, personal baseline?)", ORANGE),
        (1.2, 6.0, "frame: 33 landmarks + view   (~15 per second)", TEAL),
        (6.0, 1.2, "tick: movement progress (gauge)", ORANGE), (6.0, 1.2, "rep_counted: n", ORANGE),
        (6.0, 1.2, "rep_scored: score + feedback", ORANGE), (1.2, 6.0, "end", TEAL),
        (6.0, 10.6, "summary + encrypted repetition data", GREY), (6.0, 1.2, "summary: reps, mean score, focus", ORANGE),
    ]
    y = 5.45
    for a, b, text, c in msgs:
        ax.annotate("", xy=(b, y), xytext=(a, y), arrowprops=dict(arrowstyle="-|>", color=c, lw=1.8))
        ax.text((a + b) / 2, y + 0.08, text, ha="center", va="bottom", fontsize=9.5, color="#1F2937")
        y -= 0.58
    ax.add_patch(FancyBboxPatch((4.9, 2.05), 2.2, 1.65, boxstyle="round,pad=0.02,rounding_size=0.05", fc=LIGHT, ec=TEAL, lw=1, zorder=0))
    ax.text(7.25, 2.9, "per frame:\nsmooth -> angles\n-> counter\nper rep: DTW/NCC\nscore + feedback", fontsize=8.5, color=TEAL, va="center")
    fig.savefig(os.path.join(R, "fig_sequence.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def datamodel():
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.axis("off"); ax.set_xlim(0, 12.3); ax.set_ylim(-0.3, 5.4)
    tables = [
        (0.2, 1.0, "users", ["id, role (doctor | patient)", "username, full_name", "pw_hash (bcrypt)", "totp_secret, mfa_enabled",
                             "doctor_id -> users", "failed_logins, locked_until"], NAVY),
        (3.3, 2.6, "assignments", ["patient_id, doctor_id", "exercise, side", "sets, reps, per_week", "notes, active"], TEAL),
        (3.3, 0.0, "baselines", ["patient_id, exercise, side", "data_enc (encrypted", "personal references)"], TEAL),
        (6.6, 1.3, "sessions", ["patient_id, assignment_id", "started, ended, source", "reps, target_reps", "mean_score, consistency",
                                "rom / stability score", "data_enc (encrypted reps,", "feedback, angle traces)"], NAVY),
        (9.6, 2.0, "audit", ["at, user_id", "action, detail"], "#334155"),
    ]
    for x, y, name, rows, c in tables:
        h = 0.62 + 0.34 * len(rows)
        ax.add_patch(FancyBboxPatch((x, y), 2.4, h, boxstyle="round,pad=0.02,rounding_size=0.05", fc=LIGHT, ec=c, lw=1.5))
        ax.add_patch(FancyBboxPatch((x, y + h - 0.45), 2.4, 0.45, boxstyle="round,pad=0.02,rounding_size=0.05", fc=c, ec=c))
        ax.text(x + 1.2, y + h - 0.22, name, ha="center", va="center", color="white", fontsize=11, fontweight="bold")
        for i, r in enumerate(rows):
            ax.text(x + 0.12, y + h - 0.72 - i * 0.34, r, fontsize=8.8, color="#1F2937", va="center")
    for a, b in [((2.6, 3.0), (3.3, 3.5)), ((2.6, 1.8), (3.3, 0.8)), ((5.7, 3.4), (6.6, 3.0)), ((2.6, 2.4), (6.6, 2.2))]:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.3))
    fig.savefig(os.path.join(R, "fig_datamodel.png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    sequence()
    datamodel()
    architecture()
    print("ok")
