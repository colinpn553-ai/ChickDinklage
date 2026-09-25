#!/usr/bin/env python3
"""Fit a generated backdrop image to the video's working canvas.

The generation pipeline draws each scene on a canvas of
CANVAS_W x CANVAS_H (the 1080x1920 frame plus Ken Burns headroom) and
places characters relative to a horizon line at HORIZON_Y. A backdrop image
has its own ground line, so this script scales/crops it so that ground line
lands on HORIZON_Y, and extends the floor downward (mirrored, blended) if
the image ends short of the canvas bottom.

usage: prepare_backdrop.py SRC OUT --base 0.81 [--scale 0.85] [--shift 0]
  --base   fraction of SRC's height where the floor/ground begins
           (where building bases meet the paved ground)
  --scale  optional scale override; default is the smallest scale that
           still covers the canvas width and top
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageFilter

CANVAS_W, CANVAS_H = 1253, 2227  # 1080x1920 * 1.16, see ZOOM_MARGIN
HORIZON_Y = int(CANVAS_H * 0.62)


def prepare(src: Path, out: Path, base: float, scale: float | None, xshift: float) -> None:
    im = Image.open(src).convert("RGB")
    w, h = im.size
    if scale is None:
        scale = max(CANVAS_W / w, HORIZON_Y / (base * h))
    sw, sh = int(w * scale), int(h * scale)
    im = im.resize((sw, sh), Image.LANCZOS)

    top = HORIZON_Y - int(base * sh)  # y of the image's top edge on the canvas
    left = -int((sw - CANVAS_W) * (0.5 + xshift))
    canvas = Image.new("RGB", (CANVAS_W, CANVAS_H))
    canvas.paste(im, (left, top))

    bottom = top + sh
    if bottom < CANVAS_H:
        # Extend the floor by stretching its lower band downward (reads as
        # perspective); mirroring it looked like a kaleidoscope.
        need = CANVAS_H - bottom
        band = min(sh // 4, sh - max(0, HORIZON_Y - top) - 1)
        y0 = max(0, -left)
        strip = im.crop((y0, sh - band, y0 + CANVAS_W, sh))
        canvas.paste(strip.resize((CANVAS_W, band + need), Image.BICUBIC), (0, bottom - band))
        # Soften only the stretched part so streaks read as foreground blur.
        low = canvas.crop((0, bottom, CANVAS_W, CANVAS_H)).filter(ImageFilter.GaussianBlur(5))
        canvas.paste(low, (0, bottom))
    if top > 0:
        # Extend the sky upward by stretching the top row (rare).
        row = canvas.crop((0, top, CANVAS_W, top + 1)).resize((CANVAS_W, top))
        canvas.paste(row, (0, 0))

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out, quality=90)
    print(f"{src.name}: scale={scale:.3f} top={top} bottom={bottom} -> {out} ({CANVAS_W}x{CANVAS_H})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--base", type=float, required=True)
    ap.add_argument("--scale", type=float)
    ap.add_argument("--xshift", type=float, default=0.0)
    a = ap.parse_args()
    prepare(a.src, a.out, a.base, a.scale, a.xshift)
