"""Package the RehabNet code into one Kaggle script kernel and push it (GPU T4, internet on).

The kernel unpacks the code to /tmp/proj, runs the given job commands there, and writes ONLY
/kaggle/working/out/results.zip (so `kaggle kernels output` downloads a small file).

usage (from project root):
  .venv\\Scripts\\python -P tools/kaggle_job.py <slug> <job_file.sh> [--include extra/path ...] [--push]
The job file holds shell lines run inside /tmp/proj; $NPROC is set to the CPU count.
Results to keep must be written under /tmp/proj/results or /tmp/proj/data/mp_rehab24 (zipped automatically),
or listed in a line '# OUTPUT: <glob>'.
"""
import argparse
import base64
import io
import json
import os
import subprocess
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
USER = "vasanthkumar12345"
KAGGLE = os.path.join(ROOT, ".venv", "Scripts", "kaggle.exe")

TEMPLATE = r'''
import base64, io, os, subprocess, tarfile, time, zipfile, glob
T0 = time.time()
def sh(cmd):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, executable="/bin/bash")
    print("exit", r.returncode, f"t={(time.time()-T0)/60:.1f} min", flush=True)
    return r.returncode
PROJ = "/tmp/proj"; os.makedirs(PROJ, exist_ok=True)
tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE)), mode="r:gz").extractall(PROJ)
os.chdir(PROJ)
os.environ["NPROC"] = str(os.cpu_count())
sh("nvidia-smi --query-gpu=name,memory.total --format=csv; nproc; free -g | head -2")
open("job.sh", "w").write(JOB)
rc = sh("bash job.sh")
os.makedirs("/kaggle/working/out", exist_ok=True)
with zipfile.ZipFile("/kaggle/working/out/results.zip", "w", zipfile.ZIP_DEFLATED) as z:
    pats = ["results/**/*", "data/mp_rehab24/**/*.npz", "logs/**/*"] + OUTPUTS
    for p in pats:
        for f in glob.glob(p, recursive=True):
            if os.path.isfile(f):
                z.write(f)
print("zip MB", os.path.getsize("/kaggle/working/out/results.zip") / 1e6, "job exit", rc, flush=True)
'''


def bundle(extra):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for d in ["rehabnet", "scripts", "tools"] + extra:
            p = os.path.join(ROOT, d)
            if os.path.isfile(p):
                t.add(p, arcname=d)
                continue
            for dp, dn, fn in os.walk(p):
                dn[:] = [x for x in dn if x != "__pycache__"]
                for f in fn:
                    if f.endswith((".py", ".sh", ".json", ".csv", ".txt", ".md", ".html", ".js", ".css")):
                        full = os.path.join(dp, f)
                        t.add(full, arcname=os.path.relpath(full, ROOT))
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("job")
    ap.add_argument("--include", nargs="*", default=[])
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="no GPU")
    a = ap.parse_args()
    job = open(a.job).read()
    outputs = [ln.split(":", 1)[1].strip() for ln in job.splitlines() if ln.startswith("# OUTPUT:")]
    kdir = os.path.join(ROOT, "kaggle", a.slug)
    os.makedirs(kdir, exist_ok=True)
    code = (f"CODE = {bundle(a.include)!r}\nJOB = {job!r}\nOUTPUTS = {outputs!r}\n" + TEMPLATE)
    open(os.path.join(kdir, f"{a.slug}.py"), "w").write(code)
    meta = {"id": f"{USER}/{a.slug}", "title": a.slug, "code_file": f"{a.slug}.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": not a.cpu, "enable_tpu": False,
            "enable_internet": True, "keywords": [], "dataset_sources": [], "kernel_sources": [],
            "competition_sources": [], "model_sources": []}
    if not a.cpu:
        meta["machine_shape"] = "NvidiaTeslaT4"
    json.dump(meta, open(os.path.join(kdir, "kernel-metadata.json"), "w"), indent=2)
    print("kernel written:", kdir, f"({os.path.getsize(os.path.join(kdir, a.slug + '.py'))/1e3:.0f} kB)")
    if a.push:
        subprocess.run([KAGGLE, "kernels", "push", "-p", kdir], check=True)


if __name__ == "__main__":
    main()
