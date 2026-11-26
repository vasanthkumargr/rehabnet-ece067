"""Personal baseline (Milestone 4).

A doctor can mark a supervised session as the patient's baseline for an exercise. Its good repetitions are stored
(encrypted) and, in later sessions, added to the reference set. Milestone 2 showed that on camera data five personal
repetitions alone were not a stable reference, so they are combined with the general references (see
scripts/m4_baseline_eval.py for the evaluation that chose this design). The personal repetitions also define the
patient's own expected range of motion, which adapts the counter: a patient whose comfortable range is smaller
than the reference still gets every repetition counted.
"""
from __future__ import annotations

import numpy as np

from rehabnet.core.exercises import EXERCISES, angle_index
from rehabnet.core.reps import RepThresholds, thresholds_from_reps
from rehabnet.core.scoring import ReferenceModel

from .engine import reference

MIN_REPS = 5
MIN_SCORE = 50.0
PERSONAL_WEIGHT = 3   # each personal repetition is included this many times among the references


def baseline_from_session(data: dict, exercise: str, side) -> dict | None:
    if not data:
        return None
    reps = [(r, seg) for r, seg in zip(data.get("reps", []), data.get("segments", [])) if r["score"] >= MIN_SCORE]
    if len(reps) < MIN_REPS:
        return None
    reps.sort(key=lambda x: -x[0]["score"])
    keep = [seg for _, seg in reps[:15]]
    return dict(exercise=exercise, side=side, reps=keep, angle_names=data.get("angle_names"))


def model_from_baseline(exercise: str, side, data: dict) -> ReferenceModel:
    ref = reference(exercise, side)
    general = ref["model"].reps
    personal = [np.array(r, float) for r in data["reps"]]
    return ReferenceModel.build(exercise, side, list(general) + personal * PERSONAL_WEIGHT, max_refs=10_000)


def personal_thresholds(exercise: str, side, data: dict) -> RepThresholds | None:
    """Use the patient's own counter thresholds only when their movement range is clearly smaller than the
    reference (< 70 %): on healthy volunteers personal thresholds counted slightly worse (scripts/m4_baseline_eval.py),
    but a patient with a limited range would otherwise have repetitions missed."""
    idx = angle_index(EXERCISES[exercise].count_angle, side)
    pth = thresholds_from_reps([np.nanmean(np.array(r, float)[:, idx], 1) for r in data["reps"]])
    ref = reference(exercise, side)["thresholds"]
    return pth if abs(pth.peak - pth.rest) < 0.7 * abs(ref.peak - ref.rest) else None
