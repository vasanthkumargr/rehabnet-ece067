"""Milestone 1 figures (results/m1/fig_*.png)."""
import os
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rehabnet.core.angles import ANGLE_NAMES, COMMON, angle_matrix, to_common  # noqa: E402
from rehabnet.core.normalize import one_euro_sequence  # noqa: E402

RES = os.path.join(ROOT, "results", "m1")
D = os.path.join(ROOT, "data", "rehab24_6")
NAVY, TEAL, ORANGE, GREY, RED = "#122B4A", "#0F766E", "#D97706", "#6B7280", "#B91C1C"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold"})
EXN = {1: "Arm abduction", 2: "Arm V-W", 3: "Push-ups", 4: "Leg abduction", 5: "Leg lunge", 6: "Squat"}
PRETTY = {"l_elbow": "L elbow", "r_elbow": "R elbow", "l_shoulder": "L shoulder", "r_shoulder": "R shoulder",
          "l_hip": "L hip flex.", "r_hip": "R hip flex.", "l_knee": "L knee", "r_knee": "R knee",
          "trunk_lean": "Trunk lean", "hip_abd_l": "L hip abd.", "hip_abd_r": "R hip abd."}
SK = [("l_shoulder", "r_shoulder"), ("l_shoulder", "l_elbow"), ("l_elbow", "l_wrist"), ("r_shoulder", "r_elbow"),
      ("r_elbow", "r_wrist"), ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip"), ("l_hip", "r_hip"),
      ("l_hip", "l_knee"), ("l_knee", "l_ankle"), ("r_hip", "r_knee"), ("r_knee", "r_ankle")]


def save(fig, name):
    fig.savefig(os.path.join(RES, name), dpi=170, bbox_inches="tight")
    plt.close(fig)


def pipeline():
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.axis("off")
    steps = [("Camera /\nvideo frame", "RGB 30 fps"), ("MediaPipe\nBlazePose", "33 landmarks\n2-D + 3-D world"),
             ("Common\njoint set", "14 joints shared by\ncamera, mocap, Kinect"),
             ("Normalisation", "hip-centred, torso-scaled\n+ One-Euro smoothing"),
             ("Joint angles", "elbow, shoulder, hip,\nknee, trunk, abduction"),
             ("Validated vs\nOptiTrack", "MAE, correlation,\nrange of motion")]
    w, gap = 1.85, 0.35
    for i, (t, sub) in enumerate(steps):
        x = i * (w + gap)
        col = TEAL if i in (1, 3, 4) else NAVY
        ax.add_patch(FancyBboxPatch((x, 0.35), w, 1.1, boxstyle="round,pad=0.04,rounding_size=0.12", fc=col, ec="none"))
        ax.text(x + w / 2, 0.98, t, ha="center", va="center", color="white", fontsize=12, fontweight="bold")
        ax.text(x + w / 2, 0.05, sub, ha="center", va="center", color=GREY, fontsize=9.5)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + w + gap - 0.03, 0.9), xytext=(x + w + 0.03, 0.9),
                        arrowprops=dict(arrowstyle="-|>", color=GREY, lw=1.8))
    ax.set_xlim(-0.1, len(steps) * (w + gap)); ax.set_ylim(-0.35, 1.6)
    save(fig, "fig_pipeline.png")


def dataset_overview():
    seg = pd.read_csv(os.path.join(D, "Segmentation.csv"), sep=";")
    c = seg.groupby(["exercise_id", "correctness"]).size().unstack()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(6)
    ax.bar(x - 0.2, c[1], 0.4, color=TEAL, label="correct")
    ax.bar(x + 0.2, c[0], 0.4, color=ORANGE, label="incorrect")
    ax.set_xticks(x, [EXN[i] for i in range(1, 7)], rotation=15)
    ax.set_ylabel("repetitions"); ax.legend(frameon=False); ax.set_title("REHAB24-6: 1,072 labelled repetitions")
    save(fig, "fig_rehab24_overview.png")


def overlay(vid="PM_000", ex=1, frame=500, variant="full"):
    ims = []
    for cam, vf in (("c17", f"{vid}-Camera17-30fps.mp4"), ("c18", f"{vid}-Camera18-30fps-transposed.mp4")):
        z = np.load(os.path.join(ROOT, "data", "mp_rehab24", variant, f"Ex{ex}", f"{vid}_{cam}.npz"))
        i = int(np.argmin(np.abs(z["frame"] - frame)))
        fr_idx = int(z["frame"][i])
        cap = cv2.VideoCapture(os.path.join(D, "videos", f"Ex{ex}", vf)); cap.set(1, fr_idx); _, im = cap.read()
        mp = to_common(z["img"][..., :2], "mediapipe")[i]
        gt = to_common(np.load(os.path.join(D, "joints2d", f"Ex{ex}", f"{vid}-{cam}-30fps.npy"))[..., :2], "optitrack")[fr_idx]
        for a, b in SK:
            pa, pb = mp[COMMON.index(a)], mp[COMMON.index(b)]
            if np.isfinite(pa).all() and np.isfinite(pb).all():
                cv2.line(im, tuple(pa.astype(int)), tuple(pb.astype(int)), (60, 220, 60), 5)
        for p in gt:
            cv2.circle(im, tuple(p.astype(int)), 11, (40, 40, 230), 4)
        ims.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw=dict(width_ratios=[16 / 9, 9 / 16]))
    for ax, im, t in zip(axs, ims, ("Camera 17 (front view)", "Camera 18 (side view)")):
        ax.imshow(im); ax.axis("off"); ax.set_title(t)
    fig.suptitle("Green: MediaPipe skeleton    Red circles: OptiTrack ground truth", y=0.04, fontsize=11, color=GREY)
    save(fig, f"fig_overlay_{vid}.png")


def angle_traj(variant):
    seg = pd.read_csv(os.path.join(D, "Segmentation.csv"), sep=";")
    fig, axs = plt.subplots(1, 2, figsize=(13, 3.8))
    for ax, (ex, ang) in zip(axs, ((6, "l_knee"), (1, "r_shoulder"))):
        g = seg[seg.exercise_id == ex]
        vid = g.video_id.iloc[0]
        gg = g[g.video_id == vid]
        a, b = gg.first_frame.min(), min(gg.first_frame.min() + 600, gg.last_frame.max())
        gt3 = np.load(os.path.join(D, "joints3d", f"Ex{ex}", f"{vid}-30fps.npy"))[..., :3]
        GA = angle_matrix(to_common(gt3, "optitrack"), up=np.array([0, 1.0, 0]))[:, ANGLE_NAMES.index(ang)]
        ax.plot(np.arange(a, b) / 30, GA[a:b], color=NAVY, lw=2.2, label="OptiTrack (ground truth)")
        for cam, col, lab in (("c17", TEAL, "MediaPipe, camera 17"), ("c18", ORANGE, "MediaPipe, camera 18")):
            z = np.load(os.path.join(ROOT, "data", "mp_rehab24", variant, f"Ex{ex}", f"{vid}_{cam}.npz"))
            F = z["frame"]; w = to_common(z["world"], "mediapipe")
            w = pd.DataFrame(w.reshape(len(w), -1)).interpolate(limit_direction="both").values.reshape(w.shape)
            fps = 30 / max(1, int(F[1] - F[0]))
            A = angle_matrix(one_euro_sequence(w, fps, 1.5, 0.01), up=np.array([0, -1.0, 0]))[:, ANGLE_NAMES.index(ang)]
            sel = (F >= a) & (F < b)
            ax.plot(F[sel] / 30, A[sel], color=col, lw=1.4, label=lab)
        for _, r in gg.iterrows():
            if r.first_frame < b:
                ax.axvspan(r.first_frame / 30, min(r.last_frame, b) / 30, color=(TEAL if r.correctness else ORANGE), alpha=0.06)
        ax.set_title(f"{EXN[ex]} - {PRETTY[ang]} angle ({vid})"); ax.set_xlabel("time (s)"); ax.set_ylabel("degrees")
    axs[0].legend(frameon=False, fontsize=9, loc="lower left")
    save(fig, "fig_angle_trajectories.png")


def result_figs(tag=""):
    kp = pd.read_csv(os.path.join(RES, f"raw{tag}_keypoints.csv"))
    ang = pd.read_csv(os.path.join(RES, f"raw{tag}_angles.csv"))
    rom = pd.read_csv(os.path.join(RES, f"raw{tag}_rom.csv"))
    # keypoint accuracy by joint and view
    t = kp.groupby(["joint", "view"]).pck20.mean().unstack()
    order = ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]
    t = t.loc[order]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    x = np.arange(len(t)); cols = {"front": TEAL, "half-profile": NAVY, "profile": ORANGE}
    for i, v in enumerate([c for c in ("front", "half-profile", "profile") if c in t]):
        ax.bar(x + (i - 1) * 0.27, t[v] * 100, 0.27, color=cols[v], label=v)
    ax.set_xticks(x, [o.replace("_", " ") for o in order], rotation=20); ax.set_ylabel("PCK@0.2 torso (%)")
    ax.set_ylim(0, 105); ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2)); ax.set_title("2-D keypoint accuracy by joint and camera view")
    save(fig, "fig_kp_accuracy.png")
    # angle MAE per exercise / view
    a = ang[ang.key & (ang.src == "mp3d_smooth")]
    t = a.groupby(["ex", "view"]).mae.mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 3.8))
    x = np.arange(len(t))
    for i, v in enumerate([c for c in ("front", "half-profile", "profile") if c in t]):
        ax.bar(x + (i - 1) * 0.27, t[v], 0.27, color=cols[v], label=v)
    ax.set_xticks(x, [EXN[e] for e in t.index], rotation=15); ax.set_ylabel("mean absolute error (deg)")
    ax.legend(frameon=False, ncol=3); ax.set_title("Joint-angle error vs OptiTrack (key joints, 3-D MediaPipe)")
    save(fig, "fig_angle_mae.png")
    # ROM scatter
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), sharex=True, sharey=True)
    for ax, src, lab in zip(axs, ("mp3d_smooth", "mp2d"), ("MediaPipe 3-D world", "MediaPipe 2-D image")):
        r = rom[rom.src == src]
        for v in ("front", "half-profile", "profile"):
            rr = r[r.view == v]
            ax.scatter(rr.gt_rom, rr.mp_rom, s=7, alpha=0.35, color=cols[v], label=v)
        ax.plot([0, 180], [0, 180], color=GREY, lw=1, ls="--")
        rr = r[["gt_rom", "mp_rom"]].dropna()
        ax.set_title(f"{lab}   r = {rr.corr().iloc[0, 1]:.2f}")
        ax.set_xlabel("ground-truth range of motion (deg)")
    axs[0].set_ylabel("measured range of motion (deg)"); axs[0].legend(frameon=False, markerscale=3)
    save(fig, "fig_rom_scatter.png")


def invariance_fig():
    p = os.path.join(RES, "normalisation_invariance_by_cond.csv")
    if not os.path.exists(p):
        return
    s = pd.read_csv(p)
    s = s[s.method.isin(["none", "bbox", "torso"])]
    t = s.pivot(index="method", columns="cond", values="diff_pct").loc[["none", "bbox", "torso"]]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(3)
    ax.bar(x - 0.2, t["far_left"], 0.4, color=NAVY, label="shrunk to 55 %, shifted left")
    ax.bar(x + 0.2, t["far_right"], 0.4, color=TEAL, label="shrunk to 70 %, shifted right")
    ax.set_xticks(x, ["no normalisation\n(raw pixels)", "bounding box", "torso (ours)"])
    ax.set_ylabel("keypoint change (% of torso)"); ax.set_yscale("log")
    for i, v in enumerate(t.values):
        for j, val in enumerate(v):
            ax.text(i + (j - 0.5) * 0.4, val * 1.12, f"{val:.1f}", ha="center", fontsize=9)
    ax.legend(frameon=False); ax.set_title("Same movement, person further away / off-centre")
    save(fig, "fig_norm_invariance.png")


if __name__ == "__main__":
    variant = os.environ.get("VARIANT", "full_every3")
    tag = os.environ.get("TAG", "")
    pipeline(); dataset_overview()
    if len(sys.argv) > 1 and sys.argv[1] == "static":
        sys.exit()
    overlay(variant="full"); angle_traj(variant); result_figs(tag); invariance_fig()
