"""Milestone 1 - validate the MediaPipe pipeline on REHAB24-6 against OptiTrack ground truth.

1. Detection rate (per camera view / lighting)
2. 2-D keypoint accuracy vs projected OptiTrack joints (error / torso length, PCK)
3. Bounding-box normalisation + One-Euro smoothing: jitter and scale stability
4. Joint-angle accuracy: MediaPipe 3-D (world) and 2-D angles vs OptiTrack 3-D angles (MAE, r, ROM error per rep)

Only frames inside annotated repetitions with valid mocap are used.
Outputs -> results/m1/*.csv and results/m1/fig_*.png
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rehabnet.core.angles import ANGLE_NAMES, COMMON, angle_matrix, to_common  # noqa: E402
from rehabnet.core.normalize import bbox_normalize, jitter, one_euro_sequence  # noqa: E402

D = os.path.join(ROOT, "data", "rehab24_6")
MPD = os.path.join(ROOT, "data", "mp_rehab24", os.environ.get("VARIANT", "full"))
TAG = os.environ.get("TAG", "")
RES = os.path.join(ROOT, "results", "m1")
LIMB = ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist",
        "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
LI = [COMMON.index(j) for j in LIMB]
# which angles matter for which exercise (the "moving" joints)
KEY_ANGLES = {1: ["r_shoulder", "r_elbow", "trunk_lean"], 2: ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow"],
              3: ["l_elbow", "r_elbow", "l_shoulder", "r_shoulder"], 4: ["hip_abd_l", "hip_abd_r", "l_knee", "r_knee", "trunk_lean"],
              5: ["l_knee", "r_knee", "l_hip", "r_hip", "trunk_lean"], 6: ["l_knee", "r_knee", "l_hip", "r_hip", "trunk_lean"]}
VIEW = {("c17", "front"): "front", ("c17", "half-profile"): "half-profile", ("c17", "profile"): "profile",
        ("c18", "front"): "profile", ("c18", "half-profile"): "half-profile", ("c18", "profile"): "front"}


def load_gt3d(vid, ex):
    a = np.load(os.path.join(D, "joints3d", f"Ex{ex}", f"{vid}-30fps.npy"))
    return a[..., :3]


def load_gt2d(vid, ex, cam):
    return np.load(os.path.join(D, "joints2d", f"Ex{ex}", f"{vid}-{cam}-30fps.npy"))[..., :2]


def main():
    os.makedirs(RES, exist_ok=True)
    seg = pd.read_csv(os.path.join(D, "Segmentation.csv"), sep=";")
    kp_rows, ang_rows, rep_rows, det_rows, norm_rows = [], [], [], [], []
    for (vid, ex), g in seg.groupby(["video_id", "exercise_id"]):
        gt3 = load_gt3d(vid, ex)
        gt_c = to_common(gt3, "optitrack")
        gt_ang = angle_matrix(gt_c, up=np.array([0.0, 1.0, 0.0]))
        for cam in ("c17", "c18"):
            f = os.path.join(MPD, f"Ex{ex}", f"{vid}_{cam}.npz")
            if not os.path.exists(f):
                continue
            z = np.load(f)
            F = z["frame"]
            F = F[F < len(gt3)]
            img, world = z["img"][:len(F)], z["world"][:len(F)]
            step = int(F[1] - F[0]) if len(F) > 1 else 1
            fps = 30.0 / step
            G3, GA = gt_c[F], gt_ang[F]
            gt2 = to_common(load_gt2d(vid, ex, cam), "optitrack")[F]
            mp2 = to_common(img[..., :2], "mediapipe")
            mp3 = to_common(world, "mediapipe")
            vis = to_common(img[..., 3:4], "mediapipe")[..., 0]
            T = len(F)
            inrep = np.zeros(len(gt3), bool)
            for _, r in g.iterrows():
                if r.mocap_erroneous == 0:
                    inrep[r.first_frame:r.last_frame + 1] = True
            mask = inrep[F]
            det = np.isfinite(mp2[:, 0, 0])
            view = VIEW[(cam, g.cam17_orientation.iloc[0])]
            light = int(g.lights_on.iloc[0])
            det_rows.append(dict(video=vid, ex=ex, cam=cam, view=view, lights=light, frames=int(mask.sum()),
                                 detected=float(det[mask].mean()) if mask.any() else np.nan))
            m = mask & det
            if m.sum() < 10:
                continue
            # ---- 2-D keypoints
            torso = np.linalg.norm(gt2[:, COMMON.index("neck")] - gt2[:, COMMON.index("pelvis")], axis=-1)
            err = np.linalg.norm(mp2 - gt2, axis=-1) / torso[:, None]
            for j, name in zip(LI, LIMB):
                e = err[m, j]
                kp_rows.append(dict(video=vid, ex=ex, cam=cam, view=view, lights=light, joint=name,
                                    nme=float(np.mean(e)), pck10=float(np.mean(e < 0.1)), pck20=float(np.mean(e < 0.2)),
                                    vis=float(np.nanmean(vis[m, j]))))
            # ---- normalisation / smoothing
            mp2i = img[:, :, :2].copy()
            raw_c = to_common(mp2i, "mediapipe")
            filled = pd.DataFrame(raw_c.reshape(T, -1)).interpolate(limit_direction="both").values.reshape(raw_c.shape)
            sm = one_euro_sequence(filled, fps=fps, min_cutoff=1.5, beta=0.01)
            nb, boxes = bbox_normalize(filled, body=LI)
            nb_s, _ = bbox_normalize(sm, body=LI)
            seglen = lambda x: np.linalg.norm(x[:, COMMON.index("l_shoulder")] - x[:, COMMON.index("l_hip")], axis=-1)
            e_raw = np.linalg.norm(filled - gt2, axis=-1) / torso[:, None]
            e_sm = np.linalg.norm(sm - gt2, axis=-1) / torso[:, None]
            norm_rows.append(dict(video=vid, ex=ex, cam=cam, view=view, fps=fps,
                                  jitter_raw=jitter(nb[m][:, LI]), jitter_smooth=jitter(nb_s[m][:, LI]),
                                  nme_raw=float(np.mean(e_raw[m][:, LI])), nme_smooth=float(np.mean(e_sm[m][:, LI])),
                                  cv_torso_px=float(np.std(seglen(filled)[m]) / np.mean(seglen(filled)[m])),
                                  cv_torso_norm=float(np.std(seglen(nb)[m]) / np.mean(seglen(nb)[m])),
                                  box_size_cv=float(np.std(boxes[m, 2]) / np.mean(boxes[m, 2]))))
            # ---- joint angles (3-D world, smoothed 3-D, and 2-D image)
            a3 = angle_matrix(mp3, up=np.array([0.0, -1.0, 0.0]))
            w_f = pd.DataFrame(mp3.reshape(T, -1)).interpolate(limit_direction="both").values.reshape(mp3.shape)
            a3s = angle_matrix(one_euro_sequence(w_f, fps, 1.5, 0.01), up=np.array([0.0, -1.0, 0.0]))
            a2 = angle_matrix(sm, up=np.array([0.0, -1.0]))
            for k, name in enumerate(ANGLE_NAMES):
                key = name in KEY_ANGLES[ex]
                for src, A in (("mp3d", a3), ("mp3d_smooth", a3s), ("mp2d", a2)):
                    d = A[m, k] - GA[m, k]
                    ok = np.isfinite(d)
                    if ok.sum() < 10:
                        continue
                    r = np.corrcoef(A[m, k][ok], GA[m, k][ok])[0, 1]
                    ang_rows.append(dict(video=vid, ex=ex, cam=cam, view=view, angle=name, key=key, src=src,
                                         mae=float(np.mean(np.abs(d[ok]))), bias=float(np.mean(d[ok])), r=float(r)))
            # ---- range of motion per repetition (the clinically used number)
            for _, r in g.iterrows():
                if r.mocap_erroneous:
                    continue
                sel = (F >= r.first_frame) & (F <= r.last_frame)
                for name in KEY_ANGLES[ex]:
                    k = ANGLE_NAMES.index(name)
                    seg_gt = gt_ang[r.first_frame:r.last_frame + 1, k]
                    gt_rom = np.nanmax(seg_gt) - np.nanmin(seg_gt)
                    for src, A in (("mp3d_smooth", a3s), ("mp2d", a2)):
                        x = A[sel, k]
                        if np.isfinite(x).sum() < 5:
                            continue
                        rep_rows.append(dict(video=vid, ex=ex, cam=cam, view=view, rep=r.repetition_number,
                                             angle=name, src=src, gt_rom=gt_rom,
                                             mp_rom=float(np.nanmax(x) - np.nanmin(x)), correct=r.correctness))
        print("done", vid, flush=True)

    det, kp, ang, rep, nrm = map(pd.DataFrame, (det_rows, kp_rows, ang_rows, rep_rows, norm_rows))
    for n, d in (("detection", det), ("keypoints", kp), ("angles", ang), ("rom", rep), ("normalisation", nrm)):
        d.to_csv(os.path.join(RES, f"raw{TAG}_{n}.csv"), index=False)
    rep["rom_err"] = (rep.mp_rom - rep.gt_rom).abs()
    summ = {
        "detection_by_view": det.groupby(["cam", "view"]).detected.mean().round(4),
        "detection_by_light": det.groupby("lights").detected.mean().round(4),
        "kp_by_joint": kp.groupby("joint")[["nme", "pck10", "pck20"]].mean().round(3),
        "kp_by_view": kp.groupby("view")[["nme", "pck10", "pck20"]].mean().round(3),
        "kp_by_light": kp.groupby("lights")[["nme", "pck10", "pck20"]].mean().round(3),
        "angles_key_by_src": ang[ang.key].groupby("src")[["mae", "r"]].mean().round(2),
        "angles_key_by_src_view": ang[ang.key].groupby(["src", "view"])[["mae", "r"]].mean().round(2).unstack(0),
        "angles_by_angle": ang[ang.key & (ang.src == "mp3d_smooth")].groupby("angle")[["mae", "bias", "r"]].mean().round(2),
        "angles_by_ex": ang[ang.key].groupby(["ex", "src"]).mae.mean().round(2).unstack(),
        "rom_err_by_src": rep.groupby("src").rom_err.agg(["mean", "median"]).round(2),
        "rom_corr": rep.groupby("src").apply(lambda d: d[["gt_rom", "mp_rom"]].corr().iloc[0, 1]).round(3),
        "normalisation": nrm[["jitter_raw", "jitter_smooth", "nme_raw", "nme_smooth", "cv_torso_px",
                              "cv_torso_norm", "box_size_cv"]].mean().round(4),
    }
    with open(os.path.join(RES, f"summary{TAG}.txt"), "w") as f:
        for k, v in summ.items():
            f.write(f"== {k}\n{v.to_string()}\n\n")
    print(open(os.path.join(RES, f"summary{TAG}.txt")).read())


if __name__ == "__main__":
    main()
