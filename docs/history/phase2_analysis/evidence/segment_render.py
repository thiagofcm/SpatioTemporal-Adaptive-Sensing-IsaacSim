"""Segment dark obstacles in a clean top-down render (Fig 6b background = img-019; Fig 6a background = img-017)."""
import sys, numpy as np
from PIL import Image
from scipy import ndimage
f, = sys.argv[1:2]
a = np.asarray(Image.open(f).convert("RGB")).astype(int)
R, G, B = a[..., 0], a[..., 1], a[..., 2]
dark = (np.abs(R - B) <= 12) & (R >= 35) & (R <= 90)          # neutral dark grey obstacles/walls
dark = ndimage.binary_opening(dark, iterations=2)
lab, n = ndimage.label(dark)
print(f, a.shape, "components:", n)
for i, sl in enumerate(ndimage.find_objects(lab), 1):
    m = lab[sl] == i
    if m.sum() < 200: continue
    ys, xs = sl
    print(f"  px x[{xs.start},{xs.stop}] y[{ys.start},{ys.stop}] fill={m.mean():.2f}")
    if m.mean() < 0.8:   # L-shape or merged: print row/col occupancy profile summary
        colf = m.mean(0); rowf = m.mean(1)
        print(f"     full cols px {xs.start+np.where(colf>0.9)[0].min() if (colf>0.9).any() else None}-{xs.start+np.where(colf>0.9)[0].max() if (colf>0.9).any() else None}"
              f"  full rows px {ys.start+np.where(rowf>0.9)[0].min() if (rowf>0.9).any() else None}-{ys.start+np.where(rowf>0.9)[0].max() if (rowf>0.9).any() else None}")
