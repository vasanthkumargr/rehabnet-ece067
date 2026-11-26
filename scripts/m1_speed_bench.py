"""Time each BlazePose model on CPU and (if available) GPU. Prints the delegate to use and writes
results/m1/pose_model_speed_<host>.csv. Exit code 0; prints 'BEST_DELEGATE=cpu|gpu'."""
import os
import platform
import subprocess
import sys
import time

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
VID = os.path.join(ROOT, "data", "rehab24_6", "videos", "Ex1", "PM_000-Camera17-30fps.mp4")
host = sys.argv[1] if len(sys.argv) > 1 else platform.node()

code = r'''
import os, sys, time, cv2, numpy as np
sys.path.insert(0, ROOT)
from rehabnet.core.pose import PoseExtractor
cap = cv2.VideoCapture(VID); fr = []
for i in range(150):
    ok, f = cap.read(); fr.append(cv2.cvtColor(cv2.resize(f, (960, 540)), cv2.COLOR_BGR2RGB))
with PoseExtractor(VAR) as pe:
    ts = []; okc = 0
    for i, f in enumerate(fr):
        t = time.perf_counter(); r = pe.process(f, i * 33); ts.append((time.perf_counter() - t) * 1000); okc += r.ok
print("RESULT", np.median(ts[20:]), okc / len(fr))
'''
rows = []
for dele in ("cpu", "gpu"):
    for var in ("lite", "full", "heavy"):
        env = dict(os.environ, REHABNET_DELEGATE=dele)
        src = code.replace("ROOT", repr(ROOT)).replace("VID", repr(VID)).replace("VAR", repr(var))
        p = subprocess.run([sys.executable, "-c", src], env=env, capture_output=True, text=True, timeout=900)
        line = [ln for ln in p.stdout.splitlines() if ln.startswith("RESULT")]
        if line:
            ms, det = map(float, line[0].split()[1:])
            rows.append(dict(host=host, delegate=dele, model=var, median_ms=round(ms, 1), fps=round(1000 / ms, 1), detected=det))
        else:
            rows.append(dict(host=host, delegate=dele, model=var, median_ms=np.nan, fps=np.nan, detected=np.nan,
                             error=(p.stderr.strip().splitlines() or ["?"])[-1][:200]))
        print(rows[-1], flush=True)
df = pd.DataFrame(rows)
os.makedirs(os.path.join(ROOT, "results", "m1"), exist_ok=True)
df.to_csv(os.path.join(ROOT, "results", "m1", f"pose_model_speed_{host}.csv"), index=False)
full = df[(df.model == "full") & (df.detected > 0.5)]
best = full.sort_values("median_ms").delegate.iloc[0] if len(full) else "cpu"
print(f"BEST_DELEGATE={best}")
