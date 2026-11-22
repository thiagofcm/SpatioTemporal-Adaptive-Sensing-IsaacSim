"""Segment obstacles in the paper's Fig. 6b/6c crops (300 dpi) and convert pixel boxes to plot metres.

Calibration (axis ticks read from the 300-dpi crops):
  x: tick -4 at px 475, tick +4 at px 979  -> 63.0 px/m, x=0 at px 727 (both panels)
  y: 6c: tick +2 at 215, -2 at 590 ; 6b: +2 at 186, -2 at 561  -> 93.75 px/m
"""
import sys
import numpy as np
from PIL import Image
from scipy import ndimage

PANELS = {
    "6c": dict(f="figs/f6c.png", y2=215, ym2=590, area=(418, 96, 1052, 731)),
    "6b": dict(f="figs/f6b.png", y2=186, ym2=561, area=(418, 67, 1052, 702)),
}
XPM = (979 - 475) / 8.0
X0 = 727.0

root = sys.argv[1]
for name, p in PANELS.items():
    a = np.asarray(Image.open(f"{root}/{p['f']}").convert("RGB")).astype(int)
    ypm = (p["ym2"] - p["y2"]) / 4.0
    y0 = p["y2"] + 2 * ypm
    x1, y1, x2, y2 = p["area"]
    R, G, B = a[..., 0], a[..., 1], a[..., 2]
    gray = (np.abs(R - B) <= 6) & (R >= 135) & (R <= 172)
    red = (R - B >= 12) & (R - G >= 15)
    out = np.zeros_like(gray)
    out[y1:y2, x1:x2] = True
    for label, m in (("static", gray & out), ("red", red & out)):
        m = ndimage.binary_opening(m, iterations=2)
        lab, n = ndimage.label(m)
        print(f"## Fig {name} {label}: {n} components (px/m x={XPM:.2f} y={ypm:.2f})")
        for i, sl in enumerate(ndimage.find_objects(lab), 1):
            area = int((lab[sl] == i).sum())
            if area < 150:
                continue
            ys, xs = sl
            fill = area / ((ys.stop - ys.start) * (xs.stop - xs.start))
            xa, xb = (xs.start - X0) / XPM, (xs.stop - X0) / XPM
            ya, yb = (y0 - ys.stop) / ypm, (y0 - ys.start) / ypm
            print(f"  x[{xa:+.2f},{xb:+.2f}] y[{ya:+.2f},{yb:+.2f}]  size {xb-xa:.2f}x{yb-ya:.2f} m  "
                  f"centre ({(xa+xb)/2:+.2f},{(ya+yb)/2:+.2f})  fill={fill:.2f}")
