#!/usr/bin/env python3
"""Build the labelled side-by-side before/after GIF used in the README.

    python scripts/make_demo_gif.py BEFORE.mp4 AFTER.mp4 out.gif

Reads two 9:16 renders of the same source, stacks them horizontally under captions,
and writes a small looping GIF. Uses OpenCV for compositing (no drawtext/freetype
dependency) and ffmpeg's palettegen/paletteuse for a clean, small palette.
"""
import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

BG = (31, 24, 20)        # BGR, matches the README's dark ground
LABEL = (163, 148, 138)
ACCENT = (142, 117, 226)
BAR_H = 54
GAP = 14
PAD = 12


def label_bar(width, text, colour):
    bar = np.full((BAR_H, width, 3), BG, np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    cv2.putText(bar, text, ((width - tw) // 2, (BAR_H + th) // 2 - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, colour, 2, cv2.LINE_AA)
    return bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("before")
    ap.add_argument("after")
    ap.add_argument("output")
    ap.add_argument("--height", type=int, default=440, help="height of each panel")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--before-label", default="naive centre crop")
    ap.add_argument("--after-label", default="reel-maker (tracked)")
    a = ap.parse_args()

    caps = [cv2.VideoCapture(a.before), cv2.VideoCapture(a.after)]
    if not all(c.isOpened() for c in caps):
        sys.exit("could not open one of the input videos")

    src_fps = caps[0].get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / a.fps)))
    max_src_frames = int(a.seconds * src_fps)

    panel_w = None
    frames = []
    idx = 0
    while idx < max_src_frames:
        reads = [c.read() for c in caps]
        if not all(ok for ok, _ in reads):
            break
        if idx % step == 0:
            panels = []
            for _, fr in reads:
                h, w = fr.shape[:2]
                pw = int(round(w * a.height / h))
                panels.append(cv2.resize(fr, (pw, a.height), interpolation=cv2.INTER_AREA))
            panel_w = panels[0].shape[1]
            bars = [label_bar(panel_w, a.before_label, LABEL),
                    label_bar(panels[1].shape[1], a.after_label, ACCENT)]
            cols = [np.vstack([bars[i], panels[i]]) for i in range(2)]
            gap = np.full((cols[0].shape[0], GAP, 3), BG, np.uint8)
            row = np.hstack([cols[0], gap, cols[1]])
            canvas = cv2.copyMakeBorder(row, PAD, PAD, PAD, PAD,
                                        cv2.BORDER_CONSTANT, value=BG)
            frames.append(canvas)
        idx += 1
    for c in caps:
        c.release()

    if not frames:
        sys.exit("no frames were produced")

    h, w = frames[0].shape[:2]
    with tempfile.TemporaryDirectory() as td:
        raw = Path(td) / "raw.mp4"
        pal = Path(td) / "pal.png"
        p = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
             "-s", f"{w}x{h}", "-r", str(a.fps), "-i", "-",
             "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(raw)],
            stdin=subprocess.PIPE)
        for f in frames:
            p.stdin.write(f.tobytes())
        p.stdin.close()
        if p.wait() != 0:
            sys.exit("ffmpeg failed while encoding intermediate video")

        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw),
                        "-vf", "palettegen=max_colors=128:stats_mode=diff", str(pal)],
                       check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-i", str(pal),
                        "-lavfi", "paletteuse=dither=bayer:bayer_scale=3",
                        "-loop", "0", a.output], check=True)

    size_mb = Path(a.output).stat().st_size / 1e6
    print(f"{a.output}  {w}x{h}  {len(frames)} frames  {size_mb:.2f} MB")
    if size_mb > 5:
        print("warning: over 5 MB — reduce --height, --fps or --seconds", file=sys.stderr)


if __name__ == "__main__":
    main()
