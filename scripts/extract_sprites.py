#!/usr/bin/env python3
"""Cut individual character sprites out of a generated character sheet.

A sheet is a row of full-body figures on a plain (near-white) background.
The background is found by flood-filling inward from the image border, the
figures are separated by empty columns, and each is saved as a transparent
PNG scaled to 1000px tall (the size the cast code expects).

usage: extract_sprites.py SHEET OUT_DIR --prefix civ1
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

SPRITE_H = 1000
EDGE = 12


def foreground_mask(im: Image.Image) -> np.ndarray:
    """True where a figure (or an enclosed interior) is; False for the plain
    background reachable from the image border. Sheets have a soft vignette,
    so "background" means light and unsaturated rather than one exact color."""
    rgb = np.asarray(im.convert("RGB")).astype(int)
    h, w, _ = rgb.shape
    near = ((rgb.min(axis=2) > 200) & ((rgb.max(axis=2) - rgb.min(axis=2)) < 16))
    # RGB mode: PIL's floodfill silently no-ops on single-band images in some versions.
    mask = Image.fromarray(near.astype(np.uint8) * 255, "L").convert("RGB")
    inset = EDGE + 8  # the very edge often has a darker vignette ring
    seeds = [(x, y) for x in range(inset, w - inset, 8) for y in (inset, h - 1 - inset)]
    seeds += [(x, y) for y in range(inset, h - inset, 8) for x in (inset, w - 1 - inset)]
    for xy in seeds:
        if mask.getpixel(xy) == (255, 255, 255):
            ImageDraw.floodfill(mask, xy, (255, 0, 0))
    m = np.asarray(mask)
    fg = ~((m[..., 0] == 255) & (m[..., 1] == 0))
    fg[:EDGE] = fg[-EDGE:] = False  # vignette / border noise
    fg[:, :EDGE] = fg[:, -EDGE:] = False
    return fg


def split_columns(fg: np.ndarray, min_gap: int, min_w: int) -> list[tuple[int, int]]:
    # Ignore the bottom band: ground shadows under the feet join neighbours.
    rows = np.where(fg.any(axis=1))[0]
    top, bot = rows.min(), rows.max()
    body = fg[top: bot - int((bot - top) * 0.10)]
    cols = body.sum(axis=0) > 3
    spans, start, gap = [], None, 0
    for x, on in enumerate(cols):
        if on:
            if start is None:
                start = x
            gap = 0
        elif start is not None:
            gap += 1
            if gap >= min_gap:
                spans.append((start, x - gap + 1))
                start, gap = None, 0
    if start is not None:
        spans.append((start, len(cols)))
    return [s for s in spans if s[1] - s[0] >= min_w]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--min-gap", type=int, default=12)
    ap.add_argument("--min-w", type=int, default=120)
    a = ap.parse_args()

    im = Image.open(a.sheet).convert("RGB")
    fg = foreground_mask(im)
    rows = np.where(fg.any(axis=1))[0]
    y_top, y_bot = rows.min(), rows.max()
    spans = split_columns(fg, a.min_gap, a.min_w)
    W = fg.shape[1]

    # Soft ground shadow: sample its color from the bottom band between figures
    # and knock it out so figures don't carry truncated smudges.
    band0 = y_bot - int((y_bot - y_top) * 0.10)
    arr = np.asarray(im).astype(int)
    gap_cols = np.ones(W, bool)
    for x0, x1 in spans:
        gap_cols[max(0, x0 - 4): x1 + 4] = False
    shadow = None
    sel = fg[band0:y_bot + 1][:, gap_cols]
    if sel.sum() > 500:
        shadow = np.median(arr[band0:y_bot + 1][:, gap_cols][sel], axis=0)
    fg2 = fg.copy()
    if shadow is not None:
        band = arr[band0:y_bot + 1]
        is_shadow = np.abs(band - shadow).max(axis=2) < 34
        fg2[band0:y_bot + 1] &= ~is_shadow

    # White gaps between the legs / beside canes are enclosed by the outline, so
    # the border flood never reaches them; clear near-white below the waist.
    low0 = y_top + int((y_bot - y_top) * 0.55)
    low = arr[low0:y_bot + 1]
    fg2[low0:y_bot + 1] &= ~((low.min(axis=2) > 214) & ((low.max(axis=2) - low.min(axis=2)) < 16))

    alpha_full = Image.fromarray((fg2 * 255).astype(np.uint8), "L")
    alpha_full = alpha_full.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.8))
    rgba = im.copy()
    rgba.putalpha(alpha_full)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for x0, x1 in spans:
        if x0 <= EDGE + 2 or x1 >= W - EDGE - 2:
            print(f"  skipped a figure clipped by the sheet edge ({x0}-{x1})")
            continue
        col = fg2[:, x0:x1]
        r = np.where(col.sum(axis=1) > 0)[0]
        sp = rgba.crop((x0, r.min(), x1, r.max() + 1))
        sp = sp.resize((max(1, round(sp.width * SPRITE_H / sp.height)), SPRITE_H), Image.LANCZOS)
        n += 1
        out = a.out_dir / f"{a.prefix}_{n}.png"
        sp.save(out)
    print(f"{a.sheet.name}: {n} sprites -> {a.out_dir} (shadow={None if shadow is None else shadow.astype(int).tolist()})")


if __name__ == "__main__":
    main()
