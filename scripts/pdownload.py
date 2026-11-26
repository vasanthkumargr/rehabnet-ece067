"""Parallel, resumable HTTP range downloader (Zenodo is slow per connection).
usage: python pdownload.py URL OUTFILE TOTAL_SIZE [n_workers]
Chunks are written to OUTFILE.parts/, then joined. Re-running resumes."""
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

url, out, total = sys.argv[1], sys.argv[2], int(sys.argv[3])
workers = int(sys.argv[4]) if len(sys.argv) > 4 else 8
CH = 16 * 1024 * 1024
pdir = out + ".parts"
os.makedirs(pdir, exist_ok=True)
chunks = [(i, s, min(s + CH, total) - 1) for i, s in enumerate(range(0, total, CH))]


def get(c):
    i, a, b = c
    p = os.path.join(pdir, f"{i:05d}")
    need = b - a + 1
    for attempt in range(30):
        have = os.path.getsize(p) if os.path.exists(p) else 0
        if have == need:
            return
        if have > need:
            os.remove(p); have = 0
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a + have}-{b}", "User-Agent": "rehabnet-downloader"})
            with urllib.request.urlopen(req, timeout=60) as r, open(p, "ab") as f:
                while True:
                    buf = r.read(1 << 16)
                    if not buf:
                        break
                    f.write(buf)
        except Exception:
            time.sleep(3)
    raise RuntimeError(f"chunk {i} failed")


t0 = time.time()
with ThreadPoolExecutor(workers) as ex:
    for k, _ in enumerate(ex.map(get, chunks)):
        if k % 10 == 0:
            done = sum(os.path.getsize(os.path.join(pdir, f)) for f in os.listdir(pdir))
            print(f"{os.path.basename(out)} {done/1e6:.0f}/{total/1e6:.0f} MB  {done/1e6/(time.time()-t0+1e-9):.2f} MB/s", flush=True)
with open(out, "wb") as f:
    for i, _, _ in chunks:
        with open(os.path.join(pdir, f"{i:05d}"), "rb") as g:
            f.write(g.read())
assert os.path.getsize(out) == total
for x in os.listdir(pdir):
    os.remove(os.path.join(pdir, x))
os.rmdir(pdir)
print("done", out, flush=True)
