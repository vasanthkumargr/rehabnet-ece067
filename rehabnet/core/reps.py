"""Angle-based repetition counter (works live, one sample at a time).

The counting signal is the mean of the exercise's counting angle(s), oriented so that the movement goes 'up'
(towards the peak angle of a correct repetition). A repetition is a full cycle: the signal must rise by at least
the hysteresis H from a low point, reach a peak, and fall back by at least H. H is a fraction of the typical
movement amplitude (|peak - rest| learned from correct reference repetitions or from a calibration repetition).

Because the counter works on the swing relative to the last turning point, not on fixed angle levels, it keeps
working when the patient does not fully return to the start position between repetitions (e.g. stays low in a
lunge) or drifts slowly. A repetition is reported the moment the patient comes back by H after the peak.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RepThresholds:
    rest: float
    peak: float
    frac: float = 0.25            # hysteresis as a fraction of |peak - rest|
    min_samples: int = 8          # minimum rep length in samples (10 Hz -> 0.8 s)
    tau: float = 0.8              # smoothing time constant in seconds (causal exponential average)
    rate: float = 10.0            # samples per second

    @property
    def sign(self):
        return 1.0 if self.peak >= self.rest else -1.0

    @property
    def h(self):
        return self.frac * abs(self.peak - self.rest)

    def progress(self, x):
        return (np.asarray(x, float) - self.rest) / (self.peak - self.rest)


@dataclass
class Rep:
    start: int
    peak: int
    end: int
    peak_progress: float
    amplitude: float = 0.0


@dataclass
class RepCounter:
    th: RepThresholds
    reps: list[Rep] = field(default_factory=list)
    _t: int = -1
    _mode: str = "up"            # 'up' = looking for a peak, 'down' = looking for the next low
    _lo: float = np.inf
    _lo_t: int = 0
    _hi: float = -np.inf
    _hi_t: int = 0
    _pending: Rep | None = None
    _ema: float | None = None

    def update(self, x: float) -> Rep | None:
        """Feed one sample. Returns a Rep the moment one is completed (patient came back by H after the peak)."""
        self._t += 1
        if x is None or not np.isfinite(x):
            return None
        if self.th.tau > 0:
            a = 1 - np.exp(-1 / (self.th.tau * self.th.rate))
            self._ema = float(x) if self._ema is None else self._ema + a * (float(x) - self._ema)
            x = self._ema
        y = self.th.sign * float(x)
        t, H = self._t, self.th.h
        if self._mode == "up":
            if y < self._lo and self._hi - self._lo < H:
                self._lo, self._lo_t, self._hi, self._hi_t = y, t, y, t
            if y > self._hi:
                self._hi, self._hi_t = y, t
            if self._hi - self._lo >= H and y <= self._hi - H and self._hi_t - self._lo_t >= 1:
                if t - self._lo_t >= self.th.min_samples:
                    rep = Rep(self._lo_t, self._hi_t, t, float(self.th.progress(self.th.sign * self._hi)),
                              float(self._hi - self._lo))
                    self.reps.append(rep)
                    self._pending = rep
                else:
                    rep = None
                self._mode, self._lo, self._lo_t = "down", y, t
                return rep
            return None
        # mode 'down': follow the signal down to the next low point
        if y < self._lo:
            self._lo, self._lo_t = y, t
            if self._pending is not None:
                self._pending.end = t
        if y >= self._lo + H:
            self._mode, self._hi, self._hi_t, self._pending = "up", y, t, None
        return None

    @property
    def count(self):
        return len(self.reps)


def count_reps(signal: np.ndarray, th: RepThresholds) -> list[Rep]:
    rc = RepCounter(th)
    for x in signal:
        rc.update(x)
    return rc.reps


def smooth(x, tau=0.8, rate=10.0):
    """Causal exponential average (same filter the live counter applies)."""
    x = np.asarray(x, float)
    if tau <= 0 or len(x) == 0:
        return x
    a = 1 - np.exp(-1 / (tau * rate))
    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = y[i - 1] + a * (x[i] - y[i - 1]) if np.isfinite(x[i]) else y[i - 1]
    return y


def thresholds_from_reps(rep_signals: list[np.ndarray], **kw) -> RepThresholds:
    """Learn rest and peak angles from correct repetitions (each array = counting signal of one rep).
    The signals are smoothed with the same filter the counter uses, so thresholds match what it sees."""
    rests, peaks = [], []
    tau, rate = kw.get("tau", RepThresholds.tau), kw.get("rate", RepThresholds.rate)
    for s in rep_signals:
        s = smooth(s, tau, rate)
        s = s[np.isfinite(s)]
        if len(s) < 3:
            continue
        k = max(1, len(s) // 10)
        rest = np.median(np.r_[s[:k], s[-k:]])
        rests.append(rest)
        peaks.append(s[np.argmax(np.abs(s - rest))])
    return RepThresholds(rest=float(np.median(rests)), peak=float(np.median(peaks)), **kw)


def match_reps(pred: list[Rep], gt: list[tuple[int, int]]) -> dict:
    """One-to-one matching: a predicted rep matches a ground-truth rep if its peak lies inside it."""
    used = set()
    tp = 0
    for r in pred:
        for k, (a, b) in enumerate(gt):
            if k not in used and a <= r.peak <= b:
                used.add(k)
                tp += 1
                break
    fp, fn = len(pred) - tp, len(gt) - tp
    return dict(tp=tp, fp=fp, fn=fn, pred=len(pred), gt=len(gt))


def trim_rep(signal: np.ndarray, rep: Rep, th: RepThresholds, band: float = 0.15) -> tuple[int, int]:
    """Tighten a counted repetition to the actual movement: drop the quiet time before it started and after it
    ended. Start = last sample before the peak still within band*H of the starting low; end = first sample after
    the peak that is back within band*H of the final low. Uses the same smoothing as the counter."""
    s = th.sign * smooth(np.asarray(signal, float), th.tau, th.rate)
    a, p, b = rep.start, rep.peak, min(rep.end, len(s) - 1)
    if p <= a or b <= p:
        return a, b
    tol = band * th.h
    lo_a = np.nanmin(s[a:p + 1])
    start = a
    for i in range(p, a - 1, -1):
        if s[i] <= lo_a + tol:
            start = i
            break
    lo_b = np.nanmin(s[p:b + 1])
    end = b
    for i in range(p, b + 1):
        if s[i] <= lo_b + tol:
            end = i
            break
    return int(start), int(end)
