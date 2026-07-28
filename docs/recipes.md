# reel-maker recipes

Extra recipes beyond the core reframe + caption workflow in the README.

## Auto-batch tutorials/screen-recordings (card look, source-scale quality)

Each clip is independent — the auto planner handles its own shots, so ENG+ESP just work
(no timing mapping). `--no-card-fill` forces EVERY shot to card (nothing slips into a
white edge-stretch); `--crf 14` for crisp UI.

```bash
for f in *.mp4; do
  reel-maker "$f" "out/${f%.mp4}_9x16.mp4" \
    --analyze-fps 5 --min-shot 0.6 --no-card-fill --fallback card --crf 14
done
```
Then force source-scale file size (true CBR — see the README quality note):
```bash
for f in out/*.mp4; do
  ffmpeg -y -i "$f" -c:v libx264 -preset medium -b:v 10M -maxrate 10M -minrate 10M \
    -bufsize 10M -x264-params "nal-hrd=cbr:force-cfr=1" -pix_fmt yuv420p -c:a copy "${f%.mp4}_big.mp4"
  mv "${f%.mp4}_big.mp4" "$f"
done
```

## Transcribe clips (blog / captions copy)

```bash
$PY - <<'PYEOF'
from faster_whisper import WhisperModel
m = WhisperModel("small", device="cpu", compute_type="int8")
segs,_ = m.transcribe("clip.mp4", language="en", vad_filter=True)
print(" ".join(s.text.strip() for s in segs))
PYEOF
```

## Same edit, two languages (hero video with a shot list)

If ENG and ESP are the SAME edit, the shot-list JSON often maps with only the back half
retimed (the longer-language VO shifts scenes after some point). Run `--dump-shots` on both,
find where the timings diverge, and shift only the overrides after that point. In practice the
front half is usually identical and only a handful of back-half overrides need new `t0`/`t1`.

## Batch a folder (e.g. ENG + ESP sets)

Process every mp4 in a folder, reframe + caption, into an `out/` dir. Set the
caption language per file (ENG vs ESP) — here by a filename convention.

```bash
mkdir -p out
for f in *.mp4; do
  base="${f%.mp4}"
  lang="en"; case "$f" in *_ESP*|*_es*|*ESP*) lang="es";; esac
  reel-maker  "$f" "out/${base}_v.mp4" --mode track --text-weight 3
  reel-maker-captions "out/${base}_v.mp4" "out/${base}_reel.mp4" --lang "$lang"
done
```

Reframe is CPU-bound and single-file; for many clips run a few in parallel with
`xargs -P` or just let it churn.

## Compare modes side by side

When unsure which mode suits a clip, render both and stack them for review:

```bash
$PY $S/reframe.py in.mp4 t_track.mp4 --mode track
$PY $S/reframe.py in.mp4 t_blur.mp4  --mode blur
ffmpeg -y -i t_track.mp4 -i t_blur.mp4 -filter_complex hstack compare.mp4
```

## Add a typed hook (first 1–2s)

Since this ffmpeg build may lack `drawtext`, render the hook text to a transparent
PNG with Python, then overlay it for the first 2 seconds.

```bash
$PY - <<'PY'
import cv2, numpy as np
img=np.zeros((300,1080,4),np.uint8)
cv2.putText(img,"Set up in 60 seconds",(40,180),cv2.FONT_HERSHEY_SIMPLEX,2.0,(255,255,255,255),5,cv2.LINE_AA)
cv2.imwrite("hook.png",img)
PY
ffmpeg -y -i reel.mp4 -i hook.png \
  -filter_complex "[0][1]overlay=x=0:y=180:enable='lt(t,2)'" -c:a copy reel_hook.mp4
```

## Background music with auto-duck

Mix a music bed under the original audio, ducking it when narration is present.

```bash
ffmpeg -y -i reel.mp4 -i music.mp3 -filter_complex \
 "[1:a]volume=0.25[m];[0:a][m]sidechaincompress=threshold=0.03:ratio=8:release=300[a]" \
 -map 0:v -map "[a]" -c:v copy -shortest reel_music.mp4
```

## Logo overlay (branded corner)

```bash
ffmpeg -y -i reel.mp4 -i logo.png -filter_complex \
 "[1]scale=180:-1[l];[0][l]overlay=W-w-40:40" -c:a copy reel_logo.mp4
```

## Branded frame layout (screen-on-top, brand panel + caption zone)

For UI tutorials, a "phone-in-frame" layout often beats any crop: place the full
16:9 recording in the top portion of a solid brand-color 9:16 canvas, leaving a
bottom band for captions/logo. Nothing is cropped and it looks designed.

```bash
# BRAND = brand hex background; SRC scaled to full width, pinned near the top
ffmpeg -y -i in.mp4 -filter_complex \
 "color=c=0xEC4899:s=1080x1920:d=1[bg]; \
  [0:v]scale=1080:-2[fg]; \
  [bg][fg]overlay=x=0:y=250:shortest=1" \
 -c:a copy framed.mp4
```

Then caption the result with `captions.py` (bump `--margin-v` so captions sit in the
lower band, clear of the video).

## ASS colour conversion

Caption `--primary` / `--outline` take ASS colours in `&HBBGGRR` (note: **BGR**, not
RGB). To convert a brand hex `#RRGGBB`: reverse the byte pairs and prefix `&H00`.
Example `#EC4899` → `&H0099 48 EC` → `&H009948EC`.
