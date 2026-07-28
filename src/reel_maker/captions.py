#!/usr/bin/env python3
"""
captions.py — transcribe with faster-whisper and burn styled captions into a video.

Designed for vertical reels: big, bold, high-contrast, lower-third, chunked into
short phrases (long lines are unreadable on a phone). Language can be forced (en/es)
and audio in another language can be translated to English with --translate.
"""
import argparse, subprocess, os, tempfile


def fmt_ts(t):
    h = int(t // 3600); m = int((t % 3600) // 60); s = t % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def chunk_words(words, max_chars=24, max_dur=2.5):
    """Group word-timestamps into short caption cues that fit a vertical frame."""
    cues, cur, start = [], [], None
    for w in words:
        if start is None:
            start = w.start
        cur.append(w.word)
        line = "".join(cur).strip()
        if len(line) >= max_chars or (w.end - start) >= max_dur or w.word.strip().endswith((".", "!", "?")):
            cues.append((start, w.end, line))
            cur, start = [], None
    if cur:
        cues.append((start, words[-1].end, "".join(cur).strip()))
    return cues


def write_srt(cues, path):
    with open(path, "w") as f:
        for i, (a, b, txt) in enumerate(cues, 1):
            f.write(f"{i}\n{fmt_ts(a)} --> {fmt_ts(b)}\n{txt}\n\n")


def transcribe(video, lang, translate, model_size):
    from faster_whisper import WhisperModel
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    task = "translate" if translate else "transcribe"
    segments, _ = model.transcribe(video, language=lang, task=task, word_timestamps=True)
    words = []
    for seg in segments:
        if seg.words:
            words.extend(seg.words)
    return words


def burn(video, srt, out, primary, outline, fontsize, margin_v):
    style = (f"FontName=Arial,Fontsize={fontsize},Bold=1,PrimaryColour={primary},"
             f"OutlineColour={outline},BorderStyle=1,Outline=3,Shadow=1,"
             f"Alignment=2,MarginV={margin_v}")
    # escape path for ffmpeg subtitles filter
    srt_esc = srt.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    vf = f"subtitles='{srt_esc}':force_style='{style}'"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-vf", vf,
                    "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                    "-pix_fmt", "yuv420p", "-c:a", "copy", out], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--lang", default=None, help="force language, e.g. en / es (auto if omitted)")
    ap.add_argument("--translate", action="store_true", help="translate audio to English")
    ap.add_argument("--model", default="small", help="whisper size: tiny/base/small/medium")
    ap.add_argument("--fontsize", type=int, default=15)
    ap.add_argument("--margin-v", type=int, default=90, help="distance from bottom")
    ap.add_argument("--primary", default="&H00FFFFFF", help="ASS BGR color, default white")
    ap.add_argument("--outline", default="&H00000000", help="ASS BGR color, default black")
    ap.add_argument("--max-chars", type=int, default=24)
    ap.add_argument("--srt-only", action="store_true", help="just emit .srt, don't burn")
    a = ap.parse_args()

    words = transcribe(a.input, a.lang, a.translate, a.model)
    if not words:
        print("NO_SPEECH_DETECTED")
        return
    cues = chunk_words(words, max_chars=a.max_chars)
    srt_path = os.path.splitext(a.output)[0] + ".srt"
    write_srt(cues, srt_path)
    if a.srt_only:
        print(srt_path)
        return
    burn(a.input, srt_path, a.output, a.primary, a.outline, a.fontsize, a.margin_v)
    print(a.output)


if __name__ == "__main__":
    main()
