"""Automated PDF progress report for a patient (Milestone 4)."""
from __future__ import annotations

import datetime as dt
import io

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import cm  # noqa: E402
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle  # noqa: E402

from rehabnet.core.exercises import EXERCISES  # noqa: E402

from . import security as sec  # noqa: E402
from .db import Assignment, Session, User  # noqa: E402
from .progress import progress_report  # noqa: E402

NAVY = colors.HexColor("#122B4A")
TEAL = colors.HexColor("#0F766E")
FACTOR = {"posture": "movement path", "similarity": "movement shape", "rom": "range of motion",
          "stability": "stability", "tempo": "tempo", "symmetry": "symmetry"}


def _chart(series_by_ex):
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    pal = ["#0F766E", "#122B4A", "#D97706", "#7C3AED", "#B91C1C", "#0369A1"]
    for i, (name, ser) in enumerate(series_by_ex.items()):
        d = [dt.datetime.fromisoformat(p["date"].rstrip("Z")) for p in ser if p["score"] is not None]
        v = [p["score"] for p in ser if p["score"] is not None]
        if v:
            ax.plot(d, v, "o-", color=pal[i % len(pal)], lw=1.8, ms=4, label=name)
    ax.set_ylim(0, 100); ax.set_ylabel("session score")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower left")
    fig.autofmt_xdate()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf


def patient_pdf(s, p: User) -> bytes:
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], textColor=NAVY, fontSize=18, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], textColor=TEAL, fontSize=12.5, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", parent=styles["BodyText"], fontSize=9.5, leading=13)
    small = ParagraphStyle("s", parent=body, fontSize=8, textColor=colors.grey)
    doctor = s.get(User, p.doctor_id) if p.doctor_id else None
    prog = progress_report(s, p.id)
    asg = s.query(Assignment).filter(Assignment.patient_id == p.id, Assignment.active.is_(True)).all()
    sess = (s.query(Session).filter(Session.patient_id == p.id, Session.ended.isnot(None))
            .order_by(Session.started.desc()).limit(12).all())

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.6 * cm,
                            bottomMargin=1.6 * cm, title=f"RehabNet report - {p.full_name}")
    el = [Paragraph("RehabNet - Rehabilitation progress report", h1),
          Paragraph(f"<b>Patient:</b> {p.full_name} ({p.username})"
                    + (f" &nbsp; <b>Condition:</b> {p.condition}" if p.condition else "")
                    + (f" &nbsp; <b>Doctor:</b> {doctor.full_name}" if doctor else ""), body),
          Paragraph(f"Generated {dt.datetime.now().strftime('%d %b %Y %H:%M')} &nbsp; | &nbsp; "
                    f"{prog['total_sessions']} sessions recorded", small), Spacer(1, 6)]

    el.append(Paragraph("Prescribed exercises", h2))
    if asg:
        rows = [["Exercise", "Side", "Sets x reps", "Per week", "Notes"]]
        for a in asg:
            rows.append([EXERCISES[a.exercise].name, {"l": "left", "r": "right"}.get(a.side, "both"),
                         f"{a.sets} x {a.reps}", str(a.per_week), Paragraph(a.notes or "-", small)])
        el.append(_table(rows, [4.3, 1.6, 2.2, 1.8, 7.5]))
    else:
        el.append(Paragraph("No active exercises.", body))

    el.append(Paragraph("Progress", h2))
    if prog["exercises"]:
        el.append(Image(_chart({e["name"]: e["series"] for e in prog["exercises"]}), width=17 * cm, height=6.3 * cm))
        rows = [["Exercise", "Sessions", "First 3 avg", "Last 3 avg", "Score trend", "Range-of-motion trend"]]
        for e in prog["exercises"]:
            st = e["score_trend"] + (f" ({e['score_trend_per_week']:+.1f}/week)" if e["score_trend_per_week"] is not None else "")
            rt = e["rom_trend"] + (f" ({e['rom_trend_per_week']:+.1f}/week)" if e["rom_trend_per_week"] is not None else "")
            rows.append([e["name"], str(e["sessions"]), _n(e["first_avg"]), _n(e["last_avg"]), st, rt])
        el.append(_table(rows, [3.6, 1.6, 2.0, 2.0, 4.0, 4.2]))
    else:
        el.append(Paragraph("No sessions yet.", body))

    wk = [w for w in prog["weeks"] if w["planned"]]
    if wk:
        el.append(Paragraph("Adherence (sessions done / planned per week)", h2))
        rows = [["Week from"] + [w["week_start"][5:] for w in wk], ["Done / planned"] + [f"{w['done']}/{w['planned']}" for w in wk],
                ["Adherence"] + [f"{w['adherence']} %" for w in wk]]
        el.append(_table(rows, [2.6] + [14.4 / len(wk)] * len(wk)))

    el.append(Paragraph("Recent sessions", h2))
    if sess:
        rows = [["Date", "Exercise", "Reps", "Score", "Consistency", "Weakest factor", "Most frequent feedback"]]
        for x in sess:
            data = sec.decrypt_json(x.data_enc) or {}
            msgs = {}
            for r in data.get("reps", []):
                for m in r.get("feedback", []):
                    if m["code"] != "good":
                        k = m["text"].split(" (")[0]
                        msgs[k] = msgs.get(k, 0) + 1
            top = max(msgs, key=msgs.get) if msgs else "Good repetitions"
            rows.append([x.started.strftime("%d %b %H:%M"), EXERCISES[x.exercise].name, f"{x.reps}/{x.target_reps}",
                         _n(x.mean_score), _n(x.consistency), FACTOR.get(x.weakest, x.weakest or "-"), Paragraph(top, small)])
        el.append(_table(rows, [2.3, 2.9, 1.3, 1.3, 1.9, 2.4, 5.3]))
    el.append(Spacer(1, 10))
    el.append(Paragraph("Scores are 0-100 and compare each repetition with correct demonstrations measured the same way "
                        "(movement path, shape and stability). They support, but do not replace, clinical judgement.", small))
    doc.build(el)
    return buf.getvalue()


def _n(v):
    return "-" if v is None else f"{v:.0f}"


def _table(rows, widths_cm):
    t = Table(rows, colWidths=[w * cm for w in widths_cm], repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t
