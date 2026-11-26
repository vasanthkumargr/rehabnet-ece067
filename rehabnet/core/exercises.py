"""Exercise library: which joint angles matter for each exercise and how repetitions are counted.

Angle names use 'S' for the side the patient is exercising and 'O' for the other side; they are resolved with
`resolve(ex, side)`. Bilateral exercises use plain names. These definitions drive the repetition counter, the
scoring engine and the feedback messages, both for the datasets and for the app's exercise library.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .angles import ANGLE_NAMES


@dataclass
class Exercise:
    key: str
    name: str
    key_angles: list[str]            # angles that should move / be controlled (S/O placeholders allowed)
    count_angle: list[str]           # angle(s) averaged to make the repetition signal
    stability_angles: list[str] = field(default_factory=lambda: ["trunk_lean"])  # should stay still
    symmetric_pairs: list[tuple[str, str]] = field(default_factory=list)          # for symmetry checks
    sided: bool = False
    instructions: str = ""
    cues: dict[str, str] = field(default_factory=dict)  # angle -> plain-language body part for feedback


SIDE_ANGLE = {"shoulder": "{}_shoulder", "elbow": "{}_elbow", "hip": "{}_hip", "knee": "{}_knee"}

EXERCISES: dict[str, Exercise] = {
    "arm_abduction": Exercise(
        "arm_abduction", "Arm abduction", key_angles=["S_shoulder", "S_elbow", "trunk_lean"], count_angle=["S_shoulder"],
        stability_angles=["trunk_lean"], sided=True,
        instructions="Stand facing the camera. Raise your straight arm sideways up to shoulder height or above, then lower it slowly.",
        cues={"S_shoulder": "arm", "S_elbow": "elbow (keep it straight)", "trunk_lean": "upper body (keep it upright)"}),
    "arm_vw": Exercise(
        "arm_vw", "Arm V-W raise", key_angles=["l_shoulder", "r_shoulder", "l_elbow", "r_elbow"],
        count_angle=["l_elbow", "r_elbow"], symmetric_pairs=[("l_shoulder", "r_shoulder"), ("l_elbow", "r_elbow")],
        instructions="Arms straight up in a V, then pull elbows down into a W with hands up. Move both arms together.",
        cues={"l_shoulder": "left arm", "r_shoulder": "right arm", "l_elbow": "left elbow", "r_elbow": "right elbow"}),
    "pushup_table": Exercise(
        "pushup_table", "Push-up against a table", key_angles=["l_elbow", "r_elbow", "l_hip", "r_hip"],
        count_angle=["l_elbow", "r_elbow"], stability_angles=["l_hip", "r_hip"],
        symmetric_pairs=[("l_elbow", "r_elbow")],
        instructions="Hands on the table, body straight. Bend the elbows to lower your chest, then push back up.",
        cues={"l_elbow": "left elbow", "r_elbow": "right elbow", "l_hip": "hips (keep body straight)", "r_hip": "hips (keep body straight)"}),
    "leg_abduction": Exercise(
        "leg_abduction", "Leg abduction", key_angles=["hip_abd_S", "S_knee", "trunk_lean"], count_angle=["hip_abd_S"],
        stability_angles=["trunk_lean", "O_knee"], sided=True,
        instructions="Stand tall. Lift your straight leg out to the side, then lower it slowly. Do not lean your body.",
        cues={"hip_abd_S": "leg", "S_knee": "knee (keep it straight)", "trunk_lean": "upper body (keep it upright)",
              "O_knee": "standing knee"}),
    "lunge": Exercise(
        "lunge", "Lunge", key_angles=["S_knee", "O_knee", "S_hip", "trunk_lean"], count_angle=["S_knee", "O_knee"],
        stability_angles=["trunk_lean"], sided=True,
        instructions="Step forward. Lower the back knee towards the floor while the front knee stays at a right angle.",
        cues={"S_knee": "front knee", "O_knee": "back knee", "S_hip": "front hip", "trunk_lean": "upper body (keep it upright)"}),
    "squat": Exercise(
        "squat", "Squat", key_angles=["l_knee", "r_knee", "l_hip", "r_hip", "trunk_lean"], count_angle=["l_knee", "r_knee"],
        stability_angles=["trunk_lean"], symmetric_pairs=[("l_knee", "r_knee")],
        instructions="Feet shoulder-width apart. Bend knees and hips to sit back, keep your chest up, then stand up.",
        cues={"l_knee": "left knee", "r_knee": "right knee", "l_hip": "hips", "r_hip": "hips", "trunk_lean": "chest (keep it up)"}),
}

# REHAB24-6 exercise id -> library key, and its subtype text -> side
REHAB24 = {1: "arm_abduction", 2: "arm_vw", 3: "pushup_table", 4: "leg_abduction", 5: "lunge", 6: "squat"}


def side_from_subtype(subtype) -> str | None:
    if not isinstance(subtype, str):
        return None
    return "l" if "left" in subtype else ("r" if "right" in subtype else None)


def resolve(name: str, side: str | None) -> str:
    """'S_knee' -> 'l_knee' for side 'l'; 'hip_abd_S' -> 'hip_abd_l'; plain names unchanged."""
    s = side or "r"
    o = "l" if s == "r" else "r"
    if name.startswith("S_"):
        return f"{s}_{name[2:]}"
    if name.startswith("O_"):
        return f"{o}_{name[2:]}"
    if name.endswith("_S"):
        return name[:-1] + s
    if name.endswith("_O"):
        return name[:-1] + o
    return name


def angle_index(names, side):
    return [ANGLE_NAMES.index(resolve(n, side)) for n in names]
