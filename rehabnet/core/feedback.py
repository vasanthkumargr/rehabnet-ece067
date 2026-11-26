"""Turn a repetition's score breakdown into short, plain-language corrective feedback.

Rules (in priority order, at most `max_msgs` messages):
  1. range of motion too small on a moving joint     -> "Raise your arm higher / bend your knees more ..."
  2. a joint that should stay still moved too much   -> "Keep your upper body upright"
  3. left/right difference (bilateral exercises)     -> "Move both arms evenly"
  4. too fast / too slow                              -> "Slow down" / "A little quicker"
  5. movement shape differs most at one joint         -> "Watch your elbow"
Thresholds are relative to correct reference repetitions, so they adapt to each exercise. A message is only given
when the sub-score is below THRESH: values chosen on REHAB24-6 (camera, unseen people) so that each message fires on at
most ~10 % of correct repetitions (5 % for ROM, tempo and symmetry, which did not separate correct from incorrect).
"""
from __future__ import annotations

import math

from .exercises import EXERCISES, resolve

# verb phrase for "make the movement bigger" per body-part cue
BIGGER = {
    "arm": "Raise your arm higher",
    "left arm": "Raise your left arm higher", "right arm": "Raise your right arm higher",
    "leg": "Lift your leg further out",
    "left knee": "Bend your left knee more", "right knee": "Bend your right knee more",
    "front knee": "Bend your front knee more", "back knee": "Lower your back knee further",
    "left elbow": "Bend your left elbow more", "right elbow": "Bend your right elbow more",
    "hips": "Bend at the hips more", "front hip": "Sink lower into the lunge",
}


THRESH = dict(stability=53, posture=16, similarity=60, rom=53, symmetry=30, tempo=14)


TOGETHER = {
    "squat": ("Squat a little deeper", "Don't go deeper than needed - keep it controlled"),
    "lunge": ("Sink a little lower into the lunge", "Don't drop too low - keep it controlled"),
    "pushup_table": ("Bend your elbows more to bring your chest closer", "Keep the push-up controlled"),
    "arm_vw": ("Pull your elbows further down into the W", "Keep the arm movement controlled"),
}


def _cue(ex, resolved_name, side):
    for raw, text in ex.cues.items():
        if resolve(raw, side) == resolved_name:
            return text
    return resolved_name.replace("_", " ")


def feedback(result: dict, exercise: str, side: str | None = None, max_msgs: int = 2) -> list[dict]:
    """result = ReferenceModel.score(...) output. Returns [{'code','text','severity'}] (severity 1 = minor, 2 = major)."""
    ex = EXERCISES[exercise]
    f, sub = result["factors"], result["sub"]
    msgs = []
    # 1. range of motion
    if sub.get("rom") is not None and sub["rom"] < THRESH["rom"]:
        # only the exercise's main (counting) joints: they are measured most reliably by a single camera
        main = [resolve(a, side) for a in ex.count_angle]
        ratios = {k: v for k, v in f.get("_rom_ratio", {}).items() if k in main and f["_ref_rom"].get(k, 0) > 15 and v > 0}
        if len(main) > 1 and len(ratios) == len(main):
            # two joints drive the movement (both knees, both elbows): judge them together, one camera often
            # under-measures one side
            got = sum(f["_rom"][k] for k in main) / len(main)
            want = sum(f["_ref_rom"][k] for k in main) / len(main)
            r = got / max(want, 1e-6)
            if abs(math.log(max(r, 1e-3))) > 0.25:
                text = TOGETHER.get(exercise, ("Make the movement bigger", "Control the movement"))[0 if r < 1 else 1]
                msgs.append(dict(code="rom", severity=2 if sub["rom"] < THRESH["rom"] / 2 else 1,
                                 text=f"{text} (moved {got:.0f}°, aim for about {want:.0f}°)."))
            ratios = {}
        if ratios:
            worst = max(ratios, key=lambda k: abs(math.log(ratios[k])))
            cue = _cue(ex, worst, side)
            part = cue.split(" (")[0]
            got, want = f["_rom"][worst], f["_ref_rom"][worst]
            if ratios[worst] < 1:
                text = BIGGER.get(part, f"Move your {part} through a bigger range")
            else:
                text = f"Control your {part} - you moved it further than needed"
            msgs.append(dict(code="rom", severity=2 if sub["rom"] < THRESH["rom"] / 2 else 1,
                             text=f"{text} (moved {got:.0f}°, aim for about {want:.0f}°)."))
    # 2. stability
    if sub.get("stability") is not None and sub["stability"] < THRESH["stability"]:
        ex_ = f.get("_stab_excess", {})
        if ex_:
            worst = max(ex_, key=ex_.get)
            cue = _cue(ex, worst, side)
            text = f"Keep your {cue.split(' (')[0]} still" if "upright" not in cue and "straight" not in cue else \
                   f"Keep your {cue.split(' (')[0]} {'upright' if 'upright' in cue else 'straight'}"
            msgs.append(dict(code="stability", severity=2 if sub["stability"] < THRESH["stability"] / 2 else 1,
                             text=f"{text} (it moved {ex_[worst]:.0f}° more than it should)."))
    # 3. symmetry
    if sub.get("symmetry") is not None and sub["symmetry"] < THRESH["symmetry"]:
        msgs.append(dict(code="symmetry", severity=1, text="Move both sides evenly - one side is doing more than the other."))
    # 4. tempo
    if sub.get("tempo") is not None and sub["tempo"] < THRESH["tempo"] and f.get("_duration"):
        fast = f["_duration"] < f["_ref_duration"]
        msgs.append(dict(code="tempo", severity=1,
                         text=(f"Slow down - that took {f['_duration']:.1f} s, aim for about {f['_ref_duration']:.1f} s."
                               if fast else f"A little quicker - that took {f['_duration']:.1f} s, aim for about "
                               f"{f['_ref_duration']:.1f} s.")))
    # 5. movement shape
    shape_bad = ((sub.get("posture") is not None and sub["posture"] < THRESH["posture"]) or
                 (sub.get("similarity") is not None and sub["similarity"] < THRESH["similarity"]))
    if shape_bad and f.get("_angle_dev"):
        worst = max(f["_angle_dev"], key=f["_angle_dev"].get)
        msgs.append(dict(code="posture", severity=2,
                         text=f"Watch your {_cue(ex, worst, side).split(' (')[0]} - the movement path differs from the demonstration."))
    msgs.sort(key=lambda m: -m["severity"])
    if msgs and result.get("score", 0) >= 85 and all(m["code"] in ("rom", "tempo", "symmetry") for m in msgs):
        # the scored movement quality is good; the message is only about size or speed - say so
        first = msgs[0]["text"]
        msgs[0] = dict(msgs[0], severity=0, text="Good form. Next time: " + first[0].lower() + first[1:])
    if not msgs:
        msgs = [dict(code="good", severity=0, text="Good repetition - keep it up!")]
    return msgs[:max_msgs]


def session_summary(rep_results: list[dict], durations: list[float] | None = None) -> dict:
    """Aggregate repetition scores into the per-session numbers stored and shown to the doctor."""
    import numpy as np
    sc = np.array([r["score"] for r in rep_results], float)
    if len(sc) == 0:
        return dict(reps=0)
    subs = {k: float(np.nanmean([r["sub"][k] for r in rep_results if r["sub"].get(k) is not None]))
            for k in rep_results[0]["sub"] if any(r["sub"].get(k) is not None for r in rep_results)}
    out = dict(reps=int(len(sc)), mean_score=float(sc.mean()), best=float(sc.max()), worst=float(sc.min()),
               consistency=float(100 * max(0.0, 1 - sc.std() / max(sc.mean(), 1e-6))), factors=subs,
               weakest=min(subs, key=subs.get) if subs else None)
    if durations:
        d = np.array(durations, float)
        out["tempo_consistency"] = float(100 * max(0.0, 1 - d.std() / max(d.mean(), 1e-6)))
    return out
