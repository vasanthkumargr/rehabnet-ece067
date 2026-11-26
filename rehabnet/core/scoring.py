"""Exercise-quality scoring engine (Milestone 2).

A repetition (T x n_angles joint-angle array at RATE Hz) is compared with a set of correct reference repetitions of
the same exercise, measured by the same pipeline. Factors:

  posture   : DTW distance to the nearest references (shape of the movement, all key angles together)
  similarity: NCC between the repetition and its DTW-nearest references (timing-tolerant shape correlation)
  rom       : how far the range of motion of the moving angles deviates from the references (too small or too large)
  stability : extra movement of angles that should stay still (trunk, standing leg, ...)
  tempo     : repetition duration relative to the references
  symmetry  : left-right difference for bilateral exercises

Each factor is turned into a 0-100 sub-score by comparing it with the spread of the same factor among the correct
references themselves (leave-one-out): a value typical of correct repetitions scores ~100, a value as bad as the
worst 5 % of correct repetitions scores 50, twice as far scores 0. The overall score is a weighted mean.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from dtaidistance import dtw_ndim

from .angles import ANGLE_NAMES
from .exercises import EXERCISES, resolve

RATE = 10.0  # Hz used for scoring
# Weights learned on REHAB24-6 camera data (logistic regression on the sub-scores, all exercises pooled; see
# scripts/m2_score_eval.py). ROM, tempo and symmetry did not predict the physiotherapist's correct/incorrect
# label, so they get no weight in the repetition score but are still reported as sub-scores and used for feedback.
WEIGHTS = dict(posture=0.40, similarity=0.29, rom=0.0, stability=0.31, tempo=0.0, symmetry=0.0)
HANDSET_WEIGHTS = dict(posture=0.25, similarity=0.15, rom=0.30, stability=0.15, tempo=0.10, symmetry=0.05)
LOWER_IS_BETTER = {"posture": True, "similarity": False, "rom": True, "stability": True, "tempo": True, "symmetry": True}


def resample_time(x: np.ndarray, fps: float, rate: float = RATE) -> np.ndarray:
    """Resample (T, ...) from fps to rate by linear interpolation."""
    T = len(x)
    if T < 2 or abs(fps - rate) < 1e-6:
        return x
    n = max(2, int(round((T - 1) * rate / fps)) + 1)
    t0, t1 = np.arange(T) / fps, np.arange(n) / rate
    flat = x.reshape(T, -1)
    out = np.stack([np.interp(t1, t0, flat[:, j]) for j in range(flat.shape[1])], 1)
    return out.reshape((n,) + x.shape[1:])


def fill_nan(x):
    x = np.array(x, float)
    flat = x.reshape(len(x), -1)
    t = np.arange(len(x))
    for j in range(flat.shape[1]):
        m = np.isfinite(flat[:, j])
        if m.any() and (~m).any():
            flat[~m, j] = np.interp(t[~m], t[m], flat[m, j])
    return flat.reshape(x.shape)


def _rom(a):
    return np.nanpercentile(a, 95, axis=0) - np.nanpercentile(a, 5, axis=0)


def _ncc(a, b, n=50, max_lag=0.15):
    """Mean over angles of the max normalised cross-correlation (lags up to 15 % of the length).
    Angles are weighted by how much they move in b (the reference)."""
    a = resample_time(a, len(a) - 1, n - 1)
    b = resample_time(b, len(b) - 1, n - 1)
    L = int(max_lag * n)
    vals, w = [], []
    for j in range(a.shape[1]):
        x, y = a[:, j] - a[:, j].mean(), b[:, j] - b[:, j].mean()
        sx, sy = np.linalg.norm(x), np.linalg.norm(y)
        if sy < 1e-6:
            continue
        best = -1.0
        for lag in range(-L, L + 1):
            if lag >= 0:
                xx, yy = x[lag:], y[:n - lag]
            else:
                xx, yy = x[:n + lag], y[-lag:]
            den = np.linalg.norm(xx) * np.linalg.norm(yy)
            if den > 1e-9:
                best = max(best, float(np.dot(xx, yy) / den))
        vals.append(best if sx > 1e-6 else 0.0)
        w.append(np.ptp(b[:, j]))
    w = np.array(w)
    return float(np.average(vals, weights=w + 1e-6)) if vals else 0.0


@dataclass
class ReferenceModel:
    """Correct reference repetitions of one exercise (and side) + the factor spread among them."""
    exercise: str
    side: str | None
    key: list[str]                    # resolved key angle names
    stab: list[str]
    sym: list[tuple[str, str]]
    reps: list[np.ndarray]            # each (T, n_angles) at RATE, all ANGLE_NAMES
    scale: np.ndarray = None          # per key angle normaliser (deg)
    ref_rom: np.ndarray = None        # median ROM per key angle
    ref_stab_rom: np.ndarray = None
    ref_dur: float = None
    moving: np.ndarray = None         # mask of key angles that move (ROM > 15 deg)
    spread: dict = field(default_factory=dict)  # factor -> (q50, q95) among references
    k: int = 3
    window: float = 0.3

    @classmethod
    def build(cls, exercise, side, reps, k=3, max_refs=60, seed=0):
        ex = EXERCISES[exercise] if isinstance(exercise, str) else exercise
        exercise = ex.key
        rng = np.random.default_rng(seed)
        reps = [fill_nan(r) for r in reps if len(r) >= 5 and np.isfinite(r).any()]
        if len(reps) > max_refs:
            reps = [reps[i] for i in sorted(rng.choice(len(reps), max_refs, replace=False))]
        m = cls(exercise, side, [resolve(a, side) for a in ex.key_angles],
                [resolve(a, side) for a in ex.stability_angles],
                [(resolve(a, side), resolve(b, side)) for a, b in ex.symmetric_pairs], reps, k=k)
        ki = [ANGLE_NAMES.index(a) for a in m.key]
        si = [ANGLE_NAMES.index(a) for a in m.stab]
        allk = np.concatenate([r[:, ki] for r in reps])
        m.scale = np.maximum(np.nanstd(allk, 0), 5.0)
        m.ref_rom = np.median([_rom(r[:, ki]) for r in reps], 0)
        m.moving = m.ref_rom > 15
        m.ref_stab_rom = np.median([_rom(r[:, si]) for r in reps], 0)
        m.ref_dur = float(np.median([len(r) for r in reps]) / RATE)
        # factor spread among references, leave-one-out
        vals = {f: [] for f in WEIGHTS}
        for i, r in enumerate(reps):
            f = m.factors(r, exclude=i)
            for kk, v in f.items():
                if kk in vals and v is not None and np.isfinite(v):
                    vals[kk].append(v)
        for f, v in vals.items():
            if len(v) >= 3:
                m.spread[f] = (float(np.percentile(v, 50)), float(np.percentile(v, 95 if LOWER_IS_BETTER[f] else 5)))
        return m

    # ------------------------------------------------------------------ factors
    def _key(self, r):
        return (r[:, [ANGLE_NAMES.index(a) for a in self.key]] / self.scale).astype(np.double)

    def nearest(self, rep, exclude=None):
        a = self._key(fill_nan(rep))
        d = []
        for i, r in enumerate(self.reps):
            if i == exclude:
                continue
            b = self._key(r)
            w = max(3, int(self.window * max(len(a), len(b))), abs(len(a) - len(b)) + 2)
            d.append((dtw_ndim.distance(a, b, window=w, use_c=True) / np.sqrt(len(a) + len(b)), i))
        d.sort()
        return d[: self.k]

    def factors(self, rep, exclude=None, detail=False):
        rep = fill_nan(rep)
        ki = [ANGLE_NAMES.index(a) for a in self.key]
        si = [ANGLE_NAMES.index(a) for a in self.stab]
        near = self.nearest(rep, exclude)
        f = {}
        f["posture"] = float(np.mean([d for d, _ in near]))
        f["similarity"] = float(np.mean([_ncc(rep[:, ki][:, self.moving], self.reps[i][:, ki][:, self.moving])
                                         for _, i in near])) if self.moving.any() else None
        rom = _rom(rep[:, ki])
        ratio = rom / np.maximum(self.ref_rom, 1e-6)
        # deviation of the range of motion from the references, too small OR too large (mean |log ratio|)
        f["rom"] = float(np.mean(np.abs(np.log(np.clip(ratio[self.moving], 0.05, 20))))) if self.moving.any() else None
        srom = _rom(rep[:, si])
        f["stability"] = float(np.max(np.maximum(srom - self.ref_stab_rom, 0))) if len(si) else None
        dur = len(rep) / RATE
        f["tempo"] = float(abs(np.log(dur / self.ref_dur)))
        if self.sym:
            diffs = [np.nanmean(np.abs(rep[:, ANGLE_NAMES.index(a)] - rep[:, ANGLE_NAMES.index(b)])) for a, b in self.sym]
            f["symmetry"] = float(np.mean(diffs))
        else:
            f["symmetry"] = None
        if detail:
            f["_rom_ratio"] = dict(zip(self.key, ratio))
            f["_rom"] = dict(zip(self.key, rom))
            f["_ref_rom"] = dict(zip(self.key, self.ref_rom))
            f["_stab_excess"] = dict(zip(self.stab, np.maximum(srom - self.ref_stab_rom, 0)))
            f["_duration"] = dur
            f["_ref_duration"] = self.ref_dur
            # which key angle deviates most along the DTW-nearest reference (mean |diff| after resampling)
            b = self.reps[near[0][1]]
            n = 50
            ra = resample_time(rep[:, ki], len(rep) - 1, n - 1)
            rb = resample_time(b[:, ki], len(b) - 1, n - 1)
            f["_angle_dev"] = dict(zip(self.key, np.mean(np.abs(ra - rb), 0)))
        return f

    # ------------------------------------------------------------------ scores
    def subscore(self, name, v):
        if v is None or name not in self.spread or not np.isfinite(v):
            return None
        q50, q95 = self.spread[name]
        span = abs(q95 - q50) + 1e-9
        bad = (v - q50) / span if LOWER_IS_BETTER[name] else (q50 - v) / span
        return float(100 * np.clip(1 - 0.5 * max(0.0, bad), 0, 1))

    def score(self, rep, exclude=None, weights=None):
        weights = weights or WEIGHTS
        f = self.factors(rep, exclude, detail=True)
        subs = {k: self.subscore(k, f.get(k)) for k in WEIGHTS}
        use = {k: w for k, w in weights.items() if subs.get(k) is not None and w > 0}
        total = sum(subs[k] * w for k, w in use.items()) / sum(use.values())
        return dict(score=float(total), sub=subs, factors=f)
