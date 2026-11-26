"""Extract selected members of a remote zip using HTTP range requests (no full download).
usage: python remote_zip_extract.py URL SIZE OUTDIR SUBSTRING [workers]"""
import io
import os
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

url, size, outdir, pat = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
workers = int(sys.argv[5]) if len(sys.argv) > 5 else 8


def fetch(a, b):
    for k in range(20):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}", "User-Agent": "rehabnet"})
            return urllib.request.urlopen(req, timeout=60).read()
        except Exception:
            time.sleep(2 + k)
    raise RuntimeError("range failed")


class Remote(io.RawIOBase):
    def __init__(s): s.p = 0
    def seekable(s): return True
    def readable(s): return True
    def tell(s): return s.p
    def seek(s, o, w=0):
        s.p = {0: o, 1: s.p + o, 2: size + o}[w]; return s.p
    def readinto(s, b):
        if s.p >= size: return 0
        d = fetch(s.p, min(s.p + len(b), size) - 1)
        b[:len(d)] = d; s.p += len(d); return len(d)


z = zipfile.ZipFile(io.BufferedReader(Remote(), buffer_size=1 << 16))
members = [i for i in z.infolist() if pat in i.filename and not i.is_dir()]
print(f"{len(members)} members, {sum(m.compress_size for m in members)/1e6:.0f} MB compressed", flush=True)


def one(m):
    dst = os.path.join(outdir, m.filename)
    if os.path.exists(dst) and os.path.getsize(dst) == m.file_size:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    # local header: 30 bytes + name + extra; read it to find the data offset
    h = fetch(m.header_offset, m.header_offset + 29)
    n, e = int.from_bytes(h[26:28], "little"), int.from_bytes(h[28:30], "little")
    start = m.header_offset + 30 + n + e
    raw = b"".join(fetch(a, min(a + (8 << 20), start + m.compress_size) - 1)
                   for a in range(start, start + m.compress_size, 8 << 20))
    if m.compress_type == zipfile.ZIP_DEFLATED:
        import zlib
        raw = zlib.decompress(raw, -15)
    elif m.compress_type != zipfile.ZIP_STORED:
        raise RuntimeError(f"unsupported compression {m.compress_type}")
    assert len(raw) == m.file_size, m.filename
    with open(dst + ".tmp", "wb") as f:
        f.write(raw)
    os.replace(dst + ".tmp", dst)


t0 = time.time()
with ThreadPoolExecutor(workers) as ex:
    for k, _ in enumerate(ex.map(one, members), 1):
        if k % 10 == 0 or k == len(members):
            print(f"{k}/{len(members)} files  {time.time()-t0:.0f}s", flush=True)
print("done", flush=True)
