"""Shared helpers so that reference building, evaluation and the live app cut repetitions the same way."""
from __future__ import annotations

import numpy as np

from .exercises import EXERCISES, angle_index
from .reps import RepThresholds, count_reps, trim_rep


def segment_stream(angles: np.ndarray, exercise: str, side, th: RepThresholds):
    """Count repetitions in a 10 Hz angle stream and return [(start, end, rep)] trimmed to the movement."""
    idx = angle_index(EXERCISES[exercise].count_angle, side)
    sig = np.nanmean(angles[:, idx], 1)
    out = []
    for r in count_reps(sig, th):
        a, b = trim_rep(sig, r, th)
        if b - a >= 4:
            out.append((a, b, r))
    return out


def label_segments(segs, gt_reps, gt_labels):
    """Give each counted repetition the label of the annotated repetition that contains its peak (None if none)."""
    labels = []
    for a, b, r in segs:
        lab = None
        for (ga, gb), gl in zip(gt_reps, gt_labels):
            if ga <= r.peak <= gb:
                lab = gl
                break
        labels.append(lab)
    return labels
