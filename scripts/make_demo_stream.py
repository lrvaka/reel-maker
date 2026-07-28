#!/usr/bin/env python3
"""Generate a synthetic 16:9 "trading livestream" clip for demos and docs.

This is the hardest common case for auto-cropping, and the reason the tool exists. A
trading stream puts the payload — the chart and the price levels — off to one side, with a
webcam box in a corner and a ticker along the bottom. A centre crop lands between them and
keeps neither.

Three scenes, so the shot planner has real cuts and different treatments to choose:
  0.0-2.5s  flat title card          -> should route to `extend` (wide title reads in full)
  2.5-7.0s  chart + cam + ticker     -> wants a tracked `crop` toward the chart
  7.0-9.5s  full-frame level callout -> wants `card` or a wider crop

Uses only OpenCV + numpy, so it needs no fonts, no drawtext, and no network. Deterministic.

    python scripts/make_demo_stream.py out.mp4
"""
import argparse
import subprocess
import sys

import cv2
import numpy as np

W, H = 1920, 1080
BG = (28, 22, 18)
PANEL = (44, 34, 28)
GRID = (58, 46, 38)
UP = (120, 190, 110)
DOWN = (110, 105, 225)
TEXT = (238, 236, 232)
MUTED = (150, 140, 130)
GOLD = (70, 175, 220)

TICKER = "ES 5842.25  +0.42%     NQ 20114.75  +0.61%     CL 78.14  -1.02%     GC 2418.60  +0.18%     6E 1.0842  +0.09%"


def _text(f, s, org, scale, colour, thick=2):
    cv2.putText(f, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


def title_card(t):
    """Flat solid background, wide headline. The classic `extend` candidate."""
    f = np.full((H, W, 3), (52, 34, 22), np.uint8)
    _text(f, "MORNING MACRO", (250, 500), 3.2, TEXT, 8)
    _text(f, "pre-market levels  //  session plan", (250, 610), 1.4, (190, 178, 165), 3)
    return f


def _candles(f, x0, y0, w, h, t, seed=7):
    """A deterministic candle series that advances with time."""
    rng = np.random.default_rng(seed)
    n = 46
    closes = 0.5 + np.cumsum(rng.normal(0, 0.035, n))
    # np.ptp(), not closes.ptp() — the ndarray method was removed in NumPy 2.
    closes = (closes - closes.min()) / max(1e-6, float(np.ptp(closes)))
    step = w / n
    grown = int(min(n, 12 + t * 7))  # candles "print" over time -> real motion
    for i in range(grown):
        c = closes[i]
        o = closes[i - 1] if i else c
        top = y0 + int((1 - max(c, o)) * h)
        bot = y0 + int((1 - min(c, o)) * h)
        cx = int(x0 + i * step + step / 2)
        col = UP if c >= o else DOWN
        cv2.line(f, (cx, top - 9), (cx, bot + 9), col, 1, cv2.LINE_AA)
        cv2.rectangle(f, (cx - int(step * 0.3), top), (cx + int(step * 0.3), max(bot, top + 2)),
                      col, -1)
    return closes, grown, step


def chart_scene(t):
    """Chart left-of-centre, cam box top-right, ticker along the bottom."""
    f = np.full((H, W, 3), BG, np.uint8)

    # chart panel — deliberately NOT centred
    cv2.rectangle(f, (90, 130), (1180, 830), PANEL, -1)
    for gy in range(190, 830, 90):
        cv2.line(f, (110, gy), (1160, gy), GRID, 1)
    _candles(f, 130, 180, 1010, 600, t)

    _text(f, "ES  15m", (120, 175), 0.9, MUTED, 2)
    _text(f, "5842.25", (330, 175), 1.0, UP, 2)

    # the level callout a viewer must be able to read
    cv2.line(f, (110, 420), (1160, 420), GOLD, 2, cv2.LINE_AA)
    _text(f, "KEY LEVEL  5838.50", (740, 408), 0.85, GOLD, 2)

    # webcam box, top right
    cv2.rectangle(f, (1250, 130), (1830, 470), (38, 30, 25), -1)
    cv2.circle(f, (1540, 275), 78, (96, 108, 132), -1)
    cv2.ellipse(f, (1540, 430), (140, 78), 0, 180, 360, (96, 108, 132), -1)
    _text(f, "LIVE", (1268, 160), 0.7, DOWN, 2)

    # notes panel, lower right
    cv2.rectangle(f, (1250, 520), (1830, 830), PANEL, -1)
    for i, line in enumerate(["> hold above 5838", "> target 5851", "> invalidate 5829"]):
        _text(f, line, (1275, 575 + i * 62), 0.78, (200, 190, 178), 2)

    # ticker tape, scrolling
    cv2.rectangle(f, (0, 960), (W, 1040), (20, 16, 13), -1)
    shift = int((t * 150) % 1400)
    _text(f, TICKER, (60 - shift, 1012), 0.92, (205, 196, 184), 2)
    _text(f, TICKER, (60 - shift + 1500, 1012), 0.92, (205, 196, 184), 2)
    return f


def callout_scene(t):
    """Full-bleed zoom on the level — content reaches the edges, so `extend` would streak."""
    f = np.full((H, W, 3), PANEL, np.uint8)
    for gy in range(60, 1080, 80):
        cv2.line(f, (0, gy), (W, gy), GRID, 1)
    _candles(f, 40, 60, 1840, 940, t + 3, seed=3)
    cv2.line(f, (0, 560), (W, 560), GOLD, 3, cv2.LINE_AA)
    _text(f, "5838.50  RECLAIM", (620, 530), 1.5, GOLD, 4)
    return f


SCENES = [(title_card, 2.5), (chart_scene, 4.5), (callout_scene, 2.5)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output")
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(a.fps), "-i", "-",
           "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", a.output]
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError:
        sys.exit("ffmpeg not found on PATH")
    for fn, secs in SCENES:
        for i in range(int(secs * a.fps)):
            p.stdin.write(fn(i / a.fps).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        sys.exit("ffmpeg failed while encoding the demo clip")
    print(a.output)


if __name__ == "__main__":
    main()
