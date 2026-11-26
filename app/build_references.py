"""Build the exercise reference models used by the live app (run once; output in app/data/references/).

For every library exercise and side: the correct REHAB24-6 repetitions measured by the camera pipeline (MediaPipe,
all views, all people) become the scoring references, and the repetition-counter thresholds are learned from them.
Exercises recorded on one side only are mirrored to provide the other side. Also extracts a short demonstration
clip (one correct repetition, front camera) per exercise for the patient guide.
"""
import os
import pickle
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rehabnet.core.angles import ANGLE_NAMES  # noqa: E402
from rehabnet.core.exercises import EXERCISES, angle_index  # noqa: E402
from rehabnet.core.pipeline import label_segments, segment_stream  # noqa: E402
from rehabnet.core.reps import thresholds_from_reps  # noqa: E402
from rehabnet.core.scoring import ReferenceModel  # noqa: E402

OUT = os.path.join(ROOT, "app", "data", "references")
MEDIA = os.path.join(ROOT, "app", "web", "media")
SWAP = {}
for n in ANGLE_NAMES:
    if n.startswith("l_"):
        SWAP[n] = "r_" + n[2:]
    elif n.startswith("r_"):
        SWAP[n] = "l_" + n[2:]
    elif n.endswith("_l"):
        SWAP[n] = n[:-1] + "r"
    elif n.endswith("_r"):
        SWAP[n] = n[:-1] + "l"
MIRROR_IDX = [ANGLE_NAMES.index(SWAP.get(n, n)) for n in ANGLE_NAMES]


def mirror(a):
    return a[:, MIRROR_IDX]


def demo_clip(exn, rep):
    """Write a short browser-playable clip of one correct repetition (front view) as WebM, fall back to GIF."""
    os.makedirs(MEDIA, exist_ok=True)
    ex_id = [k for k, v in {1: "arm_abduction", 2: "arm_vw", 3: "pushup_table", 4: "leg_abduction", 5: "lunge",
                            6: "squat"}.items() if v == exn][0]
    cam_file = "Camera17-30fps.mp4" if rep["cam"] == "c17" else "Camera18-30fps-transposed.mp4"
    path = os.path.join(ROOT, "data", "rehab24_6", "videos", f"Ex{ex_id}", f"{rep['video']}-{cam_file}")
    cap = cv2.VideoCapture(path)
    a, b = rep["start"] * 3, rep["end"] * 3 + 3
    cap.set(cv2.CAP_PROP_POS_FRAMES, a)
    frames = []
    for _ in range(a, b, 2):  # 15 fps
        ok, f = cap.read()
        if not ok:
            break
        cap.grab()
        h, w = f.shape[:2]
        s = 480 / max(h, w)
        frames.append(cv2.resize(f, (int(w * s) // 2 * 2, int(h * s) // 2 * 2), interpolation=cv2.INTER_AREA))
    cap.release()
    dst = os.path.join(MEDIA, f"{exn}.webm")
    vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"VP80"), 15, (frames[0].shape[1], frames[0].shape[0]))
    if vw.isOpened():
        for f in frames:
            vw.write(f)
        vw.release()
        if os.path.getsize(dst) > 1000:
            return os.path.basename(dst)
    from PIL import Image
    ims = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).quantize(128) for f in frames]
    dst = os.path.join(MEDIA, f"{exn}.gif")
    ims[0].save(dst, save_all=True, append_images=ims[1:], duration=66, loop=0)
    return os.path.basename(dst)


def main():
    os.makedirs(OUT, exist_ok=True)
    d = pickle.load(open(os.path.join(ROOT, "data", "processed", "rehab24_reps.pkl"), "rb"))
    reps, streams = d["reps"], d["streams"]
    for st in streams:  # correctness label of each annotated repetition, in stream order
        lab = {(r["rep"]): r["correct"] for r in reps if r["video"] == st["video"] and r["cam"] == st["cam"]}
        order = sorted(lab)
        st["gt_labels"] = [lab[k] for k in order][: len(st["gt_reps"])]
    index = {}
    for exn, ex in EXERCISES.items():
        sides = ["l", "r"] if ex.sided else [None]
        for side in sides:
            # 1) counter thresholds from the annotated correct repetitions (mirrored if this side was not recorded)
            own = [r["mp"] for r in reps if r["exercise"] == exn and r["correct"] and (r["side"] == side or not ex.sided)]
            mir = [mirror(r["mp"]) for r in reps if ex.sided and r["exercise"] == exn and r["correct"]
                   and r["side"] == ("r" if side == "l" else "l")]
            use_mirror = len(own) < 20
            idx = angle_index(ex.count_angle, side)
            th = thresholds_from_reps([np.nanmean(r[:, idx], 1) for r in (own + mir if use_mirror else own)])
            # 2) references = correct repetitions cut by the live counter itself (same boundaries as in the app)
            src = []
            for st in streams:
                if st["exercise"] != exn or (ex.sided and st["side"] != side and not use_mirror):
                    continue
                A = st["mp"]
                if ex.sided and st["side"] != side:
                    A = mirror(A)
                segs = segment_stream(A, exn, side, th)
                labs = label_segments(segs, st["gt_reps"], st["gt_labels"])
                src += [A[a:b + 1] for (a, b, _), lab in zip(segs, labs) if lab == 1]
            model = ReferenceModel.build(exn, side, src, max_refs=80)
            key = f"{exn}_{side or 'both'}"
            pickle.dump(dict(model=model, thresholds=th, n_source=len(src), mirrored=use_mirror),
                        open(os.path.join(OUT, key + ".pkl"), "wb"))
            index[key] = dict(n=len(src), mirrored=use_mirror, rest=th.rest, peak=th.peak)
            print(key, len(src), "mirrored" if use_mirror else "", f"rest {th.rest:.0f} peak {th.peak:.0f}")
        # demo clip: a correct, front-view repetition with a typical length
        cands = [r for r in reps if r["exercise"] == exn and r["correct"] and r["view"] == "front"]
        cands.sort(key=lambda r: abs(len(r["mp"]) - np.median([len(c["mp"]) for c in cands])))
        print("  demo:", demo_clip(exn, cands[0]))
    pickle.dump(index, open(os.path.join(OUT, "index.pkl"), "wb"))


if __name__ == "__main__":
    main()
