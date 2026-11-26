"""Milestone 1 - run the MediaPipe pose pipeline over every REHAB24-6 video (both cameras).

Output per video: data/mp_rehab24/<Ex>/<video>_<cam>.npz with
  img (T,33,4) pixel landmarks + visibility, world (T,33,3) metres, frame (T,), ms (T,) processing time
Run: .venv\\Scripts\\python.exe -P scripts/m1_extract_rehab24.py [--workers 3] [--variant full] [--limit N]
"""
import argparse
import glob
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
VID = os.path.join(ROOT, "data", "rehab24_6", "videos")
OUT = os.path.join(ROOT, "data", "mp_rehab24")


def run_one(args):
    path, variant, every = args
    import cv2
    from rehabnet.core.pose import PoseExtractor

    ex = os.path.basename(os.path.dirname(path))
    name = os.path.basename(path)
    vid, cam = name.split("-")[0], ("c17" if "Camera17" in name else "c18")
    tag = variant if every == 1 else f"{variant}_every{every}"
    dst = os.path.join(OUT, tag, ex, f"{vid}_{cam}.npz")
    if os.path.exists(dst):
        return dst, 0, 0.0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    imgs, worlds, ms, idx = [], [], [], []
    with PoseExtractor(variant) as pe:
        f = 0
        while True:
            if not cap.grab():
                break
            if f % every:
                f += 1
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            hw = frame.shape[:2]
            scale = 1.0
            if max(frame.shape[:2]) > 960:
                scale = 960 / max(frame.shape[:2])
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            t0 = time.perf_counter()
            pf = pe.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), f * 1000.0 / fps)
            ms.append((time.perf_counter() - t0) * 1000)
            img = pf.img.copy()
            img[:, :3] /= scale
            imgs.append(img)
            worlds.append(pf.world)
            idx.append(f)
            f += 1
    cap.release()
    np.savez_compressed(dst, img=np.stack(imgs).astype(np.float32), world=np.stack(worlds).astype(np.float32),
                        frame=np.array(idx), ms=np.array(ms, np.float32), fps=fps,
                        hw=np.array(hw))
    return dst, len(imgs), float(np.median(ms))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--only", default="", help="comma list of video ids")
    a = ap.parse_args()
    vids = sorted(glob.glob(os.path.join(VID, "Ex*", "*.mp4")))
    if a.only:
        keep = set(a.only.split(","))
        vids = [v for v in vids if os.path.basename(v).split("-")[0] in keep]
    if a.limit:
        vids = vids[: a.limit]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for k, (dst, n, med) in enumerate(ex.map(run_one, [(v, a.variant, a.every) for v in vids]), 1):
            print(f"[{k}/{len(vids)}] {os.path.relpath(dst, ROOT)} frames={n} median={med:.1f} ms  "
                  f"elapsed={(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
