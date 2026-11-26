"""Milestone 2 - build the per-stream / per-repetition joint-angle dataset for REHAB24-6.

For every video stream (65 recordings x 2 cameras) the MediaPipe 3-D world landmarks from Milestone 1 (10 fps) are
smoothed (One-Euro), converted to joint angles, and cut into repetitions using the dataset annotation. The same is
done with the OptiTrack ground truth (resampled to the same frames) so camera-based and mocap-based scoring can be
compared.  Output: data/processed/rehab24_reps.pkl
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rehabnet.core.angles import angle_matrix, to_common  # noqa: E402
from rehabnet.core.exercises import REHAB24, side_from_subtype  # noqa: E402
from rehabnet.core.normalize import one_euro_sequence  # noqa: E402
from rehabnet.core.scoring import fill_nan  # noqa: E402

D = os.path.join(ROOT, "data", "rehab24_6")
MP = os.path.join(ROOT, "data", "mp_rehab24", "full_every3")
VIEW = {("c17", "front"): "front", ("c17", "half-profile"): "half-profile", ("c17", "profile"): "profile",
        ("c18", "front"): "profile", ("c18", "half-profile"): "half-profile", ("c18", "profile"): "front"}


def main():
    seg = pd.read_csv(os.path.join(D, "Segmentation.csv"), sep=";")
    streams, reps = [], []
    for (vid, ex), g in seg.groupby(["video_id", "exercise_id"]):
        gt3 = np.load(os.path.join(D, "joints3d", f"Ex{ex}", f"{vid}-30fps.npy"))[..., :3]
        gt_ang_all = angle_matrix(to_common(gt3, "optitrack"), up=np.array([0, 1.0, 0]))
        side = side_from_subtype(g.exercise_subtype.iloc[0])
        for cam in ("c17", "c18"):
            z = np.load(os.path.join(MP, f"Ex{ex}", f"{vid}_{cam}.npz"))
            F = z["frame"]
            F = F[F < len(gt3)]
            w = to_common(z["world"][: len(F)], "mediapipe")
            fps = 30.0 / int(F[1] - F[0])
            w = fill_nan(w)
            ang = angle_matrix(one_euro_sequence(w, fps, 1.5, 0.01), up=np.array([0, -1.0, 0])).astype(np.float32)
            gt_ang = gt_ang_all[F].astype(np.float32)
            view = VIEW[(cam, g.cam17_orientation.iloc[0])]
            gt_reps = []
            for _, r in g.sort_values("repetition_number").iterrows():
                idx = np.where((F >= r.first_frame) & (F <= r.last_frame))[0]
                if len(idx) < 3:
                    continue
                a, b = int(idx[0]), int(idx[-1]) + 1
                gt_reps.append((a, b - 1))
                reps.append(dict(video=vid, cam=cam, view=view, ex=int(ex), exercise=REHAB24[int(ex)], side=side,
                                 person=int(r.person_id), rep=int(r.repetition_number), correct=int(r.correctness),
                                 lights=int(r.lights_on), mocap_err=int(r.mocap_erroneous),
                                 mp=ang[a:b], gt=gt_ang[a:b], start=a, end=b - 1))
            streams.append(dict(video=vid, cam=cam, view=view, ex=int(ex), exercise=REHAB24[int(ex)], side=side,
                                person=int(g.person_id.iloc[0]), fps=fps, mp=ang, gt=gt_ang, gt_reps=gt_reps))
        print(vid, ex, len(g), flush=True)
    out = os.path.join(ROOT, "data", "processed", "rehab24_reps.pkl")
    pickle.dump(dict(streams=streams, reps=reps), open(out, "wb"))
    print("streams", len(streams), "reps", len(reps))


if __name__ == "__main__":
    main()
