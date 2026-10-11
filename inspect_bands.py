"""Check band counts / odd bands in your dataset:  python inspect_bands.py archive/train"""
import glob, sys
from collections import Counter
import numpy as np
import rasterio

root = sys.argv[1] if len(sys.argv) > 1 else "archive/train"
files = sorted(glob.glob(f"{root}/*/*.tif*"))
cnt = Counter(); example = {}
for f in files:
    with rasterio.open(f) as s:
        cnt[s.count] += 1; example.setdefault(s.count, f)
print("band counts:", dict(cnt))
for n, f in example.items():
    with rasterio.open(f) as s:
        a = s.read().astype(float)
        print(f"\n{n} bands  e.g. {f}\n  descriptions: {s.descriptions[:2]} ... {s.descriptions[-2:]}\n  dtype={s.dtypes[0]}  size={s.width}x{s.height}")
        print("  first-band mean/std:", a[0].mean().round(2), a[0].std().round(2),
              "| last-band mean/std:", a[-1].mean().round(2), a[-1].std().round(2))
