"""Render PDF pages into contact-sheet PNGs for visual review: contact_sheet.py file.pdf out_prefix [cols] [dpi] [per_sheet]"""
import sys

import pymupdf
from PIL import Image

pdf, out = sys.argv[1], sys.argv[2]
cols = int(sys.argv[3]) if len(sys.argv) > 3 else 3
dpi = int(sys.argv[4]) if len(sys.argv) > 4 else 50
per = int(sys.argv[5]) if len(sys.argv) > 5 else 12
doc = pymupdf.open(pdf)
ims = []
for p in doc:
    pix = p.get_pixmap(dpi=dpi)
    ims.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
print(pdf, len(ims), "pages")
for k in range(0, len(ims), per):
    chunk = ims[k:k + per]
    w = max(i.width for i in chunk); h = max(i.height for i in chunk)
    rows = (len(chunk) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (w + 8), rows * (h + 8)), (120, 120, 120))
    for j, im in enumerate(chunk):
        sheet.paste(im, ((j % cols) * (w + 8), (j // cols) * (h + 8)))
    sheet.save(f"{out}_{k // per + 1}.png")
    print(f"{out}_{k // per + 1}.png")
