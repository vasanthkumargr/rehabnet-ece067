"""Multi-session progress tracking (Milestone 4): trends per exercise and weekly adherence."""
from __future__ import annotations

import datetime as dt

import numpy as np

from rehabnet.core.exercises import EXERCISES

from .db import Assignment, Session, now


def _trend(days, values):
    """Least-squares slope per week and a plain-language label."""
    x, y = np.asarray(days, float), np.asarray(values, float)
    ok = np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(y) < 3 or np.ptp(x) < 1:
        return None, "not enough sessions yet"
    slope = float(np.polyfit(x, y, 1)[0]) * 7
    resid = y - np.polyval(np.polyfit(x, y, 1), x)
    noise = float(np.std(resid)) / max(np.sqrt(len(y)), 1)
    if abs(slope) < max(1.0, noise):
        label = "stable"
    else:
        label = "improving" if slope > 0 else "getting worse"
    return round(slope, 2), label


def progress_report(s, patient_id: int) -> dict:
    ss = (s.query(Session).filter(Session.patient_id == patient_id, Session.ended.isnot(None))
          .order_by(Session.started).all())
    by_ex = {}
    for x in ss:
        by_ex.setdefault(x.exercise, []).append(x)
    exercises = []
    for exn, lst in by_ex.items():
        t0 = lst[0].started
        days = [(x.started - t0).total_seconds() / 86400 for x in lst]
        scores = [x.mean_score for x in lst]
        slope, label = _trend(days, [v if v is not None else np.nan for v in scores])
        rom = [x.rom_score for x in lst]
        rslope, rlabel = _trend(days, [v if v is not None else np.nan for v in rom])
        first = [v for v in scores[:3] if v is not None]
        last = [v for v in scores[-3:] if v is not None]
        exercises.append(dict(
            exercise=exn, name=EXERCISES[exn].name, sessions=len(lst),
            series=[dict(id=x.id, date=x.started.isoformat() + "Z", score=x.mean_score, reps=x.reps,
                         target=x.target_reps, rom=x.rom_score, stability=x.stability_score,
                         consistency=x.consistency) for x in lst],
            score_trend_per_week=slope, score_trend=label, rom_trend_per_week=rslope, rom_trend=rlabel,
            first_avg=round(float(np.mean(first)), 1) if first else None,
            last_avg=round(float(np.mean(last)), 1) if last else None))
    # weekly adherence over the last 8 weeks: sessions done / sessions planned
    asg = s.query(Assignment).filter(Assignment.patient_id == patient_id).all()
    weeks = []
    today = now()
    for w in range(7, -1, -1):
        a = today - dt.timedelta(days=7 * (w + 1))
        b = today - dt.timedelta(days=7 * w)
        # a prescription counts for a week only if it existed for at least half of that week
        planned = sum(x.per_week for x in asg if x.created <= a + dt.timedelta(days=3.5) and (x.active or x.created > a))
        done = sum(1 for x in ss if a <= x.started < b)
        weeks.append(dict(week_start=a.date().isoformat(), planned=planned, done=done,
                          adherence=(min(100, round(100 * done / planned)) if planned else None)))
    return dict(exercises=exercises, weeks=weeks, total_sessions=len(ss))
