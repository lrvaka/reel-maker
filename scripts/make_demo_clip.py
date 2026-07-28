#!/usr/bin/env python3
"""Generate a synthetic 16:9 "dashboard screen recording" for demos and CI smoke tests.

The important content (KPI numbers) is deliberately placed in the LEFT third of the frame,
with a decorative panel on the right. That is the case a naive centre-crop destroys and
content-aware tracking survives — which is the whole point of the tool.

Uses only OpenCV + numpy (already core dependencies), so it needs no fonts, no drawtext,
and no network. Deterministic output.

    python scripts/make_demo_clip.py out.mp4 [--seconds 6] [--fps 30]
"""
import argparse
import subprocess
import sys

import cv2
import numpy as np

BG = (48, 36, 30)          # BGR — deep slate
PANEL = (68, 52, 43)
PANEL_R = (58, 45, 37)
MUTED = (163, 148, 138)
WHITE = (255, 255, 255)
GREEN = (149, 192, 111)
PINK = (142, 117, 226)

KPIS = [
    ("Monthly Revenue", "$284,190", WHITE),
    ("API Requests", "1.42M", GREEN),
    ("Error Rate", "0.03%", PINK),
]


def draw_frame(t, w=1920, h=1080):
    """One frame at time t. A subtle counter animates so scene-cut detection sees motion."""
    f = np.full((h, w, 3), BG, np.uint8)
    cv2.rectangle(f, (120, 180), (740, 900), PANEL, -1)      # KPI panel (left third)
    cv2.rectangle(f, (980, 180), (1800, 900), PANEL_R, -1)   # decorative panel (right)

    cv2.putText(f, "(decorative chart area)", (1130, 545),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 66, 58), 2, cv2.LINE_AA)

    y = 250
    for label, value, colour in KPIS:
        cv2.putText(f, label, (160, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, MUTED, 2, cv2.LINE_AA)
        cv2.putText(f, value, (160, y + 78), cv2.FONT_HERSHEY_SIMPLEX, 2.3, colour, 5, cv2.LINE_AA)
        y += 230

    # A live-looking counter so the clip isn't perfectly static.
    cv2.putText(f, f"uptime {t:0.1f}s", (1130, 840),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (110, 96, 86), 2, cv2.LINE_AA)
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("output")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    a = ap.parse_args()

    n = int(a.seconds * a.fps)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{a.width}x{a.height}", "-r", str(a.fps), "-i", "-",
           "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", a.output]
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    except FileNotFoundError:
        sys.exit("ffmpeg not found on PATH")
    for i in range(n):
        p.stdin.write(draw_frame(i / a.fps, a.width, a.height).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        sys.exit("ffmpeg failed while encoding the demo clip")
    print(a.output)


if __name__ == "__main__":
    main()
