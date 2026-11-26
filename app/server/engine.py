"""Live exercise session: turns streamed pose landmarks into repetitions, scores and feedback.

The browser runs MediaPipe and sends the 33 world landmarks (+ visibility) of each frame. Here they are
resampled to 10 Hz, smoothed (One-Euro), converted to joint angles, and fed to the validated Milestone 2 repetition
counter. A counted repetition is scored (DTW/NCC/factors vs reference repetitions) once its end is known, i.e. when
the next repetition starts or after 1.5 s of rest, and the feedback messages are sent back.
"""
from __future__ import annotations

import os
import pickle
import time
from dataclasses import dataclass, field

import numpy as np

from rehabnet.core.angles import ANGLE_NAMES, angle_matrix, to_common
from rehabnet.core.exercises import EXERCISES, angle_index, resolve
from rehabnet.core.feedback import feedback, session_summary
from rehabnet.core.normalize import OneEuro
from rehabnet.core.reps import RepCounter, trim_rep
from rehabnet.core.scoring import RATE, ReferenceModel

from .config import REF_DIR

_CACHE: dict[str, dict] = {}


def reference(exercise: str, side: str | None) -> dict:
    ex = EXERCISES[exercise]
    key = f"{exercise}_{(side or 'r') if ex.sided else 'both'}"
    if key not in _CACHE:
        _CACHE[key] = pickle.load(open(os.path.join(REF_DIR, key + ".pkl"), "rb"))
    return _CACHE[key]


@dataclass
class LiveSession:
    exercise: str
    side: str | None = None
    target_reps: int = 10
    personal_model: ReferenceModel | None = None
    thresholds: object | None = None                     # personal counter thresholds (baseline), else reference
    angles: list = field(default_factory=list)          # 10 Hz angle rows
    results: list = field(default_factory=list)          # scored repetitions
    segments: list = field(default_factory=list)         # angle rows of each scored repetition (for baselines)
    _next_t: float | None = None
    _filter: OneEuro | None = None
    _waiting: list = field(default_factory=list)          # counted, not yet scored
    _last_seen: float = 0.0
    _t0: float = 0.0

    def __post_init__(self):
        ex = EXERCISES[self.exercise]
        if not ex.sided:
            self.side = None
        elif self.side not in ("l", "r"):
            self.side = "r"
        ref = reference(self.exercise, self.side)
        self.model: ReferenceModel = self.personal_model or ref["model"]
        self.counter = RepCounter(self.thresholds or ref["thresholds"])
        self.count_idx = angle_index(ex.count_angle, self.side)
        self._filter = OneEuro(fps=RATE, min_cutoff=1.5, beta=0.01)
        self._t0 = time.time()

    # -------------------------------------------------------------- input
    def add_frame(self, t_ms: float, world: list | None) -> list[dict]:
        """world: 33 x [x, y, z] MediaPipe world landmarks (or None if no person). Returns events for the client."""
        events = []
        if world is None:
            if self._last_seen and t_ms - self._last_seen > 2000:
                events.append(dict(type="no_person"))
                self._last_seen = t_ms
            return events
        self._last_seen = t_ms
        if self._next_t is None:
            self._next_t = t_ms
        if t_ms < self._next_t - 1000:          # time went backwards (video restarted): resync
            self._next_t = t_ms
        if t_ms < self._next_t:
            return events
        if t_ms - self._next_t > 500:           # long gap (tab hidden, dropped frames): do not invent frames
            self._next_t = t_ms
        while self._next_t <= t_ms:            # 10 Hz sampling (short gaps: repeat the latest frame)
            self._next_t += 1000.0 / RATE
            events += self._sample(np.asarray(world, float))
        return events

    def _sample(self, w: np.ndarray) -> list[dict]:
        sm = self._filter(w)
        a = angle_matrix(to_common(sm[None], "mediapipe"), up=np.array([0.0, -1.0, 0.0]))[0]
        self.angles.append(a.astype(np.float32))
        x = float(np.nanmean(a[self.count_idx]))
        events = []
        rep = self.counter.update(x)
        if rep is not None:
            self._waiting.append(rep)
            events.append(dict(type="rep_counted", count=self.counter.count, target=self.target_reps,
                               progress=float(np.clip(rep.peak_progress, 0, 1.5))))
        t = len(self.angles) - 1
        done = [r for r in self._waiting if self.counter._pending is not r or t - r.end >= 15]
        for r in done:
            self._waiting.remove(r)
            events.append(self._score(r))
        # live progress of the current movement (0 = rest, 1 = reference peak), for the on-screen gauge
        events.append(dict(type="tick", count=self.counter.count,
                           progress=float(np.clip(self.counter.th.progress(self.counter._ema if self.counter._ema is not None else x), -0.2, 1.5))))
        return events

    def _score(self, rep) -> dict:
        A = np.array(self.angles)
        a, b = trim_rep(np.nanmean(A[:, self.count_idx], 1), rep, self.counter.th)
        seg = A[a: b + 1]
        if len(seg) < 5:
            seg = A[max(0, rep.start - 2): rep.end + 3]
        out = self.model.score(seg)
        msgs = feedback(out, self.exercise, self.side)
        f = out["factors"]
        self.segments.append(np.round(seg, 1).tolist())
        res = dict(n=len(self.results) + 1, score=round(out["score"], 1),
                   sub={k: (None if v is None else round(v, 1)) for k, v in out["sub"].items()},
                   feedback=msgs, duration=round(len(seg) / RATE, 2),
                   rom={k: round(float(v), 1) for k, v in f.get("_rom", {}).items()},
                   ref_rom={k: round(float(v), 1) for k, v in f.get("_ref_rom", {}).items()},
                   start=a, end=b)
        self.results.append(res)
        return dict(type="rep_scored", **res)

    # -------------------------------------------------------------- end
    def finish(self) -> dict:
        for r in list(self._waiting):
            self._score(r)
        self._waiting.clear()
        full = [dict(score=r["score"], sub=r["sub"]) for r in self.results]
        summ = session_summary(full, [r["duration"] for r in self.results]) if self.results else dict(reps=0)
        A = np.array(self.angles) if self.angles else np.zeros((0, len(ANGLE_NAMES)))
        ex = EXERCISES[self.exercise]
        keys = [resolve(k, self.side) for k in ex.key_angles]
        trace = {k: np.round(A[::2, ANGLE_NAMES.index(k)], 1).tolist() for k in keys} if len(A) else {}
        return dict(summary=summ, reps=self.results, trace=trace, trace_rate=RATE / 2, segments=self.segments,
                    angle_names=ANGLE_NAMES,
                    duration_s=round(len(self.angles) / RATE, 1))
