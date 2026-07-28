#!/usr/bin/env python3
"""
reframe.py — convert landscape video to a vertical reel with content-aware framing.

Modes:
  track   Content-aware pan/zoom. Computes an "importance" signal per frame from
          spectral-residual saliency PLUS on-screen text (Tesseract OCR), and glides
          a virtual camera to keep the important region centered. This is what keeps
          headings / form fields / UI text in frame on a screen recording.
  blur    Full frame shrunk to fit width, blurred fill top/bottom. Nothing cropped.
          Best when there is no single focal region (dense dashboards).
  center  Static center crop. Fast, dumb, only good when the subject is dead-center.

The virtual camera path is smoothed (moving average + velocity clamp) so the motion
reads as an intentional glide, not a jitter. Tune with --smooth and --max-speed.
"""
import argparse, subprocess, sys, os, json
import numpy as np
import cv2


def probe(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate,duration",
         "-of", "json", path],
        capture_output=True, text=True).stdout
    st = json.loads(out)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    num, den = st["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 30.0
    dur = float(st.get("duration") or 0)
    return w, h, fps, dur


def has_audio(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries",
         "stream=index", "-of", "csv=p=0", path],
        capture_output=True, text=True).stdout
    return bool(out.strip())


def spectral_saliency(gray):
    """Spectral-residual saliency (Hou & Zhang). Returns HxW map in [0,1]."""
    small = cv2.resize(gray, (64, 64)).astype(np.float32)
    f = np.fft.fft2(small)
    log_amp = np.log(np.abs(f) + 1e-8)
    phase = np.angle(f)
    spectral_residual = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(spectral_residual + 1j * phase))) ** 2
    sal = cv2.GaussianBlur(sal, (0, 0), 2.5)
    sal = cv2.resize(sal, (gray.shape[1], gray.shape[0]))
    rng = sal.max() - sal.min()
    return (sal - sal.min()) / (rng + 1e-8)


class OcrUnavailable(Exception):
    """Raised when OCR was requested but Tesseract or pytesseract is missing."""


def text_mask(frame_bgr, min_conf):
    """Binary-ish mask of confident OCR word boxes, downscaled to speed OCR up."""
    try:
        import pytesseract
    except ImportError as e:
        raise OcrUnavailable(
            "pytesseract is not installed. Install the OCR extra with "
            "`pip install 'reel-maker[ocr]'` (and the tesseract binary: "
            "`brew install tesseract` / `apt install tesseract-ocr`)."
        ) from e
    H, W = frame_bgr.shape[:2]
    scale = 1280.0 / max(W, 1) if W > 1280 else 1.0
    img = cv2.resize(frame_bgr, (int(W * scale), int(H * scale))) if scale != 1.0 else frame_bgr
    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    except pytesseract.TesseractNotFoundError as e:
        raise OcrUnavailable(
            "the tesseract binary is not on PATH. Install it with "
            "`brew install tesseract` (macOS) or `apt install tesseract-ocr` (Debian/Ubuntu)."
        ) from e
    except Exception as e:
        # A tesseract that is installed but broken (bad TESSDATA_PREFIX, missing language
        # data, a stripped environment) fails in assorted ways -- including a
        # UnicodeDecodeError raised while pytesseract tries to decode tesseract's own
        # error output. OCR is an enhancement, never a reason to lose the render, so any
        # failure here degrades to saliency-only rather than propagating.
        raise OcrUnavailable(
            f"tesseract is installed but failed to run ({type(e).__name__}: {e}). "
            "Check TESSDATA_PREFIX and that language data is installed."
        ) from e
    mask = np.zeros((H, W), np.float32)
    for i in range(len(data["text"])):
        try:
            conf = float(data["conf"][i])
        except ValueError:
            conf = -1
        if conf >= min_conf and data["text"][i].strip():
            x = int(data["left"][i] / scale); y = int(data["top"][i] / scale)
            w = int(data["width"][i] / scale); h = int(data["height"][i] / scale)
            mask[y:y + h, x:x + w] += 1.0
    m = mask.max()
    return mask / m if m > 0 else mask


def weighted_center(importance):
    """Centroid of the importance map (fractions of W,H)."""
    H, W = importance.shape
    total = importance.sum()
    if total < 1e-6:
        return 0.5, 0.5
    ys, xs = np.mgrid[0:H, 0:W]
    cx = (xs * importance).sum() / total / W
    cy = (ys * importance).sum() / total / H
    return float(cx), float(cy)


NBINS = 240  # horizontal importance-profile resolution


def analyze(path, analyze_fps, text_weight, min_conf, use_ocr, cut_thresh=0.11):
    """Sample the clip and return, per sample: a horizontal importance PROFILE (how
    much important stuff sits in each vertical column) and a scene-cut flag. Working
    with a per-shot profile — rather than a per-frame 'best point' — is what lets us
    choose one stable framing per shot instead of chasing a jittery centroid."""
    cap = cv2.VideoCapture(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / analyze_fps)))
    profiles, times, cuts, flats, edges, motions, edgeflats = [], [], [], [], [], [], []
    prev_small = None
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(gray, (32, 18)).astype(np.float32)
            if prev_small is None:
                cuts.append(True)
                motions.append(0.0)
            else:
                d = float(np.abs(small - prev_small).mean() / 255.0)
                cuts.append(d > cut_thresh)
                motions.append(d)
            prev_small = small
            imp = spectral_saliency(gray)
            if use_ocr and text_weight > 0:
                try:
                    imp = imp + text_weight * text_mask(frame, min_conf)
                except OcrUnavailable as e:
                    # Degrade, don't crash: saliency-only tracking is still useful, it just
                    # won't preferentially hold on-screen text. Warn once, then stop trying.
                    print(f"reel-maker: OCR disabled — {e}\n"
                          f"reel-maker: falling back to saliency-only tracking.",
                          file=sys.stderr)
                    use_ocr = False
            col = imp.sum(axis=0)                       # importance per source column
            prof = cv2.resize(col.reshape(1, -1), (NBINS, 1)).ravel()
            profiles.append(prof)
            times.append(idx / src_fps)
            # "text-card" cues: flat background (most pixels near one tone) + few edges
            sc = cv2.resize(gray, (64, 36))
            mode = np.bincount(sc.ravel(), minlength=256).argmax()
            flats.append(float(np.mean(np.abs(sc.astype(int) - mode) < 18)))
            edges.append(float(cv2.Canny(cv2.resize(gray, (320, 180)), 80, 160).mean() / 255))
            # edge-band flatness: would extend/edge-stretch be seamless? measure how uniform
            # the top+bottom rows of a width-fit are (structure here = ugly streaks)
            band = cv2.resize(gray, (96, 54))
            eb = np.concatenate([band[:3].ravel(), band[-3:].ravel()])
            edgeflats.append(float(np.mean(np.abs(eb.astype(int) - np.median(eb)) < 14)))
        idx += 1
    cap.release()
    return (np.array(times), np.array(profiles), np.array(cuts), np.array(flats),
            np.array(edges), np.array(motions), np.array(edgeflats), idx, src_fps)


def shot_segments(times, cuts, min_shot_s):
    """Sample indices where each shot starts, merging cuts closer than min_shot_s so
    a brief false detection can't cause a mid-shot reframe."""
    starts = [0]
    for i in range(1, len(times)):
        if cuts[i] and (times[i] - times[starts[-1]]) >= min_shot_s:
            starts.append(i)
    return starts + [len(times)]


def best_center(profile, crop_w, W):
    """Fixed x-center (fraction) whose crop-width window captures the most importance."""
    w = max(1, int(round(crop_w / W * NBINS)))
    if w >= NBINS:
        return 0.5
    csum = np.concatenate([[0], np.cumsum(profile)])
    window = csum[w:] - csum[:-w]                       # importance inside each window
    start = int(np.argmax(window))
    return (start + w / 2) / NBINS


def compute_plan(times, profiles, cuts, flats, edges, motions, edgeflats, out_times,
                 crop_w, W, min_shot_s, flat_thr, edge_thr, motion_thr, band_thr,
                 fill_cards, overrides, fallback="crop"):
    """Build a per-shot plan. Each shot gets ONE treatment, held still:
      - extend : flat solid title card -> fit width + edge-stretch (wide titles read
                 in full). Only when the stretch band is seamless (band flatness high)
                 and the shot is static — otherwise stretching structure makes streaks.
      - fill   : scale up so the subject fills the frame (device/product hero shots).
      - crop   : native-scale 9:16 slice centered on the important region (default).
    `overrides` (list of {t0,t1,mode,zoom,cx}) lets specific shots be hand-directed.
    Returns per-shot dicts with the mean profile so render can pick the exact center."""
    shots = []
    if len(times) == 0:
        return [{"t0": 0, "t1": 1e9, "mode": "crop", "zoom": 1.0,
                 "cx": None, "prof": None}]
    bounds = shot_segments(times, cuts, min_shot_s)
    for s, e in zip(bounds[:-1], bounds[1:]):
        t0 = times[s]
        t1 = times[e] if e < len(times) else (out_times[-1] + 1 if len(out_times) else 1e9)
        prof = profiles[s:e].mean(axis=0) if e > s else None
        flat = np.median(flats[s:e]) if e > s else 0
        edge = np.median(edges[s:e]) if e > s else 1
        mot = np.median(motions[s + 1:e]) if e > s + 1 else 0   # skip the cut frame
        band = np.median(edgeflats[s:e]) if e > s else 0
        # auto treatment
        if (fill_cards and flat > flat_thr and edge < edge_thr
                and mot < motion_thr and band > band_thr):
            mode, zoom = "extend", 1.0
        else:
            mode, zoom = fallback, 1.0   # non-text shots: crop (default) or card
        shots.append({"t0": float(t0), "t1": float(t1), "mode": mode, "zoom": zoom,
                      "cx": None, "prof": prof, "_flat": float(flat), "_edge": float(edge),
                      "_mot": float(mot), "_band": float(band)})

    # apply hand-direction overrides by time overlap
    for ov in (overrides or []):
        o0, o1 = ov.get("t0", -1e9), ov.get("t1", 1e9)
        for sh in shots:
            if sh["t1"] > o0 and sh["t0"] < o1:
                for k in ("mode", "cx", "zoom", "hold", "ease", "cover_start", "cover_end"):
                    if k in ov:
                        sh[k] = ov[k]
    return shots


def shot_for_time(shots, t):
    for sh in shots:
        if sh["t0"] <= t < sh["t1"]:
            return sh
    return shots[-1] if shots else None


def smooth_path(times, centers, out_times, smooth_s, max_speed, cuts=None):
    """Smooth the camera path WITHIN each shot, but let it snap at scene cuts.

    Averaging across a hard cut would drag the camera from the old subject into the
    new scene — so we smooth each shot independently and reset the velocity clamp at
    cut boundaries, so a new shot frames its subject immediately instead of gliding in.
    """
    if len(times) == 0:
        return np.full((len(out_times), 2), 0.5)
    cx, cy = centers[:, 0].copy(), centers[:, 1].copy()
    dt = np.median(np.diff(times)) if len(times) > 1 else 0.5
    win = max(1, int(round(smooth_s / max(dt, 1e-3))))

    # shot boundaries (sample indices where a cut starts a new shot)
    seg_starts = [0] + [i for i in range(len(times)) if cuts is not None and cuts[i]]
    seg_starts = sorted(set(seg_starts))
    seg_bounds = seg_starts + [len(times)]
    for s, e in zip(seg_bounds[:-1], seg_bounds[1:]):
        n = e - s
        if n >= 2 and win > 1:
            k = np.ones(min(win, n)) / min(win, n)
            cx[s:e] = np.convolve(cx[s:e], k, mode="same")
            cy[s:e] = np.convolve(cy[s:e], k, mode="same")

    ox = np.interp(out_times, times, cx)
    oy = np.interp(out_times, times, cy)

    cut_times = [times[i] for i in seg_starts if i > 0]
    ofps = 1.0 / np.median(np.diff(out_times)) if len(out_times) > 1 else 30.0
    max_step = max_speed / ofps
    ci = 0
    for i in range(1, len(ox)):
        snap = False
        while ci < len(cut_times) and cut_times[ci] <= out_times[i]:
            if cut_times[ci] > out_times[i - 1]:
                snap = True
            ci += 1
        if snap:
            continue  # cut between these frames: allow instant reframe
        ox[i] = np.clip(ox[i], ox[i - 1] - max_step, ox[i - 1] + max_step)
        oy[i] = np.clip(oy[i], oy[i - 1] - max_step, oy[i - 1] + max_step)
    return np.stack([ox, oy], axis=1)


def fit_extend(frame, out_w, out_h):
    """Scale the whole frame to fit width, then extend the top/bottom edge pixels to
    fill height. On a flat-background text card this is invisible (no bars) and keeps
    a wide title fully readable."""
    H, W = frame.shape[:2]
    sh = int(round(out_w * H / W))
    scaled = cv2.resize(frame, (out_w, sh), interpolation=cv2.INTER_AREA)
    if sh >= out_h:
        y0 = (sh - out_h) // 2
        return scaled[y0:y0 + out_h]
    top = (out_h - sh) // 2
    bot = out_h - sh - top
    return cv2.copyMakeBorder(scaled, top, bot, 0, 0, cv2.BORDER_REPLICATE)


def render_scaled(frame, out_w, out_h, cover, cx_frac):
    """Unified framing by a single `cover` value = scaled height / output height.
      cover >= 1  -> frame fills (and overflows) the frame; crop to fit. (scale/fill/crop)
      cover <  1  -> frame is shorter than the frame; center it and edge-replicate the
                     top/bottom gap. (extend)
    Because it's one continuous parameter, animating `cover` morphs smoothly between
    scale-to-fill and edge-extend, with replication only ever filling a real gap."""
    H, W = frame.shape[:2]
    scaled_h = max(2, int(round(cover * out_h)))
    scaled_w = max(out_w, int(round(scaled_h * W / H)))
    s = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
    x0 = int(np.clip(cx_frac * scaled_w - out_w / 2, 0, scaled_w - out_w))
    s = s[:, x0:x0 + out_w]
    if scaled_h >= out_h:
        y0 = (scaled_h - out_h) // 2
        return s[y0:y0 + out_h]
    top = (out_h - scaled_h) // 2
    return cv2.copyMakeBorder(s, top, out_h - scaled_h - top, 0, 0, cv2.BORDER_REPLICATE)


def render_extend_frozen(frame, out_w, out_h, ref_top, ref_bot):
    """Edge-extend, but pad the top/bottom with FROZEN reference rows (captured from a
    clean earlier frame) instead of the current frame's own edges. Lets animated UI/
    content enter the middle band without being smeared into the padding."""
    H, W = frame.shape[:2]
    sh = int(round(out_w * H / W))
    band = cv2.resize(frame, (out_w, sh), interpolation=cv2.INTER_AREA)
    if sh >= out_h:
        y0 = (sh - out_h) // 2
        return band[y0:y0 + out_h]
    top = (out_h - sh) // 2
    bot = out_h - sh - top
    pt = np.repeat(ref_top[None, :, :], top, axis=0)
    pb = np.repeat(ref_bot[None, :, :], bot, axis=0)
    return np.vstack([pt, band, pb])


def _gradient_bg(out_w, out_h, top_bgr, bot_bgr):
    ys = np.linspace(0, 1, out_h)[:, None]
    col = np.array(top_bgr) * (1 - ys) + np.array(bot_bgr) * ys   # out_h x 3
    return np.repeat(col[:, None, :], out_w, axis=1).astype(np.uint8)


def _rounded_mask(h, w, r):
    r = int(min(r, h // 2, w // 2))
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(m, (0, r), (w, h - r), 255, -1)
    for cx, cy in [(r, r), (w - r, r), (r, h - r), (w - r, h - r)]:
        cv2.circle(m, (cx, cy), r, 255, -1)
    return cv2.GaussianBlur(m, (0, 0), 1.1)   # antialias the edge


def render_card(frame, out_w, out_h, top_bgr, bot_bgr, radius, margin):
    """Composite a full-bleed frame into a rounded-corner card centered on a brand
    gradient — makes a plain photo match the designed 'photo card on gradient' scenes."""
    bg = _gradient_bg(out_w, out_h, top_bgr, bot_bgr)
    H, W = frame.shape[:2]
    cw = int(round(margin * out_w))
    ch = int(round(cw * H / W))
    if ch > out_h:
        ch = int(round(margin * out_h)); cw = int(round(ch * W / H))
    card = cv2.resize(frame, (cw, ch), interpolation=cv2.INTER_AREA)
    mask = (_rounded_mask(ch, cw, radius).astype(np.float32) / 255)[:, :, None]
    x0, y0 = (out_w - cw) // 2, (out_h - ch) // 2
    roi = bg[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
    bg[y0:y0 + ch, x0:x0 + cw] = (card.astype(np.float32) * mask + roi * (1 - mask)).astype(np.uint8)
    return bg


def _ease_out_cubic(u):
    return 1 - (1 - u) ** 3


def interp_keys(keys, t, field, default):
    """Smoothstep-interpolate a field across a shot's keyframes (list of {t, <field>...})."""
    ks = [k for k in keys if field in k]
    if not ks:
        return default
    if t <= ks[0]["t"]:
        return ks[0][field]
    if t >= ks[-1]["t"]:
        return ks[-1][field]
    for a, b in zip(ks[:-1], ks[1:]):
        if a["t"] <= t <= b["t"]:
            u = (t - a["t"]) / max(1e-6, b["t"] - a["t"])
            u = u * u * (3 - 2 * u)                 # smoothstep ease
            return a[field] + (b[field] - a[field]) * u
    return ks[-1][field]


def shot_cover(sh, t, fit_cover):
    """Cover value for a shot at time t. `reveal` holds a fill, then eases down to the
    extend cover so a wide title/logo reveals in full once its background settles."""
    m = sh["mode"]
    if m == "extend":
        return fit_cover
    if m == "fill":
        return max(1.0, sh.get("zoom", 1.0))
    if m == "reveal":
        cs = sh.get("cover_start", 1.0)
        ce = sh.get("cover_end", fit_cover)
        hold = sh.get("hold", sh["t0"])
        ease = max(1e-3, sh.get("ease", 0.25))
        if t < hold:
            return cs
        if t < hold + ease:
            return cs + (ce - cs) * _ease_out_cubic((t - hold) / ease)
        return ce
    return 1.0  # crop (native full-height slice)


def render(path, out_path, mode, aspect, zoom, analyze_fps, text_weight,
           min_conf, min_shot_s, flat_thr, edge_thr, motion_thr, band_thr,
           fill_cards, fill_zoom, overrides, dump_shots, crf, fallback="crop",
           cut_thresh=0.11):
    W, H, fps, dur = probe(path)
    aw, ah = (int(x) for x in aspect.split(":"))
    out_w, out_h = 1080, int(round(1080 * ah / aw))
    if out_h % 2:
        out_h += 1
    base_crop_w = int(round((H * out_w / out_h)))  # zoom=1 slice width, for centering
    fit_cover = (out_w * H / W) / out_h            # cover value that == pure edge-extend

    total_frames = int(round(dur * fps)) if dur else 0
    out_times = (np.arange(total_frames) / fps) if total_frames else np.array([])

    if mode == "center":
        shots = [{"t0": 0, "t1": 1e9, "mode": "crop", "zoom": 1.0, "cx": 0.5, "prof": None}]
        if total_frames == 0:
            total_frames = int(round((dur or 0) * fps))
    else:
        (t, profiles, cuts, flats, edges, motions, edgeflats,
         nframes, sfps) = analyze(path, analyze_fps, text_weight, min_conf, use_ocr=True,
                                 cut_thresh=cut_thresh)
        if total_frames == 0:
            total_frames = nframes
            out_times = np.arange(total_frames) / fps
        shots = compute_plan(t, profiles, cuts, flats, edges, motions, edgeflats,
                             out_times, base_crop_w, W, min_shot_s, flat_thr, edge_thr,
                             motion_thr, band_thr, fill_cards, None, fallback)
        # resolve a fixed center per shot (from its profile) for crop/fill
        for sh in shots:
            if sh["mode"] == "fill" and sh["zoom"] <= 1.0:
                sh["zoom"] = fill_zoom
            z = sh["zoom"] if sh["mode"] == "fill" else 1.0
            cw = min(W, (H / max(z, 1e-3)) * out_w / out_h)
            if sh["cx"] is None:
                sh["cx"] = (0.5 if sh["mode"] == "reveal" else
                            (best_center(sh["prof"], cw, W) if sh["prof"] is not None else 0.5))

    if dump_shots:
        for k, sh in enumerate(shots):
            print(f"shot {k:2d}  {sh['t0']:5.2f}-{sh['t1']:5.2f}s  {sh['mode']:6s}  "
                  f"zoom={sh['zoom']:.2f}  cx={sh['cx']:.2f}  "
                  f"flat={sh.get('_flat',0):.2f} edge={sh.get('_edge',0):.3f} "
                  f"mot={sh.get('_mot',0):.3f} band={sh.get('_band',0):.2f}")
        return out_path

    amap = ["-map", "1:a:0?"] if has_audio(path) else []
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{out_w}x{out_h}", "-r", f"{fps}",
         "-i", "-", "-i", path,
         "-map", "0:v:0", *amap,
         "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", out_path],
        stdin=subprocess.PIPE)

    cap = cv2.VideoCapture(path)
    i = 0
    frozen = {}   # per-freeze-shot captured clean edge rows
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        sh = dict(shot_for_time(shots, t))          # base auto treatment (copy)
        for ov in (overrides or []):                 # per-frame overrides take priority
            if ov.get("t0", -1e9) <= t < ov.get("t1", 1e9):
                sh.update({k: ov[k] for k in ov
                           if k in ("mode", "cx", "zoom", "hold", "ease", "keys",
                                    "cover_start", "cover_end", "t0", "t1", "freeze", "pad",
                                    "radius", "margin", "bg_top", "bg_bot")})
                if ov.get("mode") == "reveal" and "cx" not in ov:
                    sh["cx"] = 0.5
                break
        if sh.get("mode") == "card":
            out = render_card(frame, out_w, out_h,
                              sh.get("bg_top", [121, 0, 210]), sh.get("bg_bot", [105, 33, 221]),
                              sh.get("radius", 48), sh.get("margin", 0.955))
        elif sh.get("mode") == "anim":
            keys = sh.get("keys", [])
            cover = interp_keys(keys, t, "cover", 1.0)
            cx = interp_keys(keys, t, "cx", 0.5)
            op = interp_keys(keys, t, "opacity", 1.0)
            out = render_scaled(frame, out_w, out_h, cover, cx)
            if op < 0.999:
                out = (out.astype(np.float32) * op).astype(np.uint8)
        elif sh.get("freeze") and sh.get("mode") == "extend":
            key = sh.get("t0", 0)
            if key not in frozen:
                shh = int(round(out_w * H / W))
                bandref = cv2.resize(frame, (out_w, shh), interpolation=cv2.INTER_AREA)
                if sh.get("pad") == "auto":
                    # solid pad in the shot's dominant bg color — right for flat
                    # backgrounds where edge rows may contain animating cards/UI
                    q = (bandref // 16 * 16).reshape(-1, 3)
                    vals, cnts = np.unique(q, axis=0, return_counts=True)
                    row = np.tile(vals[cnts.argmax()].astype(np.uint8), (out_w, 1))
                    frozen[key] = (row, row)
                else:                                # clean edge rows (gradient bgs)
                    frozen[key] = (bandref[0].copy(), bandref[-1].copy())
            out = render_extend_frozen(frame, out_w, out_h, *frozen[key])
        else:
            cover = shot_cover(sh, t, fit_cover)
            out = render_scaled(frame, out_w, out_h, cover, sh["cx"])
        ff.stdin.write(out.tobytes())
        i += 1
    cap.release()
    ff.stdin.close()
    ff.wait()
    return out_path


def render_blur(path, out_path, aspect, crf):
    aw, ah = (int(x) for x in aspect.split(":"))
    out_w, out_h = 1080, int(round(1080 * ah / aw))
    if out_h % 2:
        out_h += 1
    vf = (f"split[a][b];"
          f"[a]scale={out_w}:{out_h}:force_original_aspect_ratio=increase,"
          f"crop={out_w}:{out_h},gblur=sigma=22[bg];"
          f"[b]scale={out_w}:-2[fg];"
          f"[bg][fg]overlay=(W-w)/2:(H-h)/2")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", path, "-vf", vf,
                    "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", "-c:a", "copy", out_path], check=True)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Landscape -> vertical reel reframer")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--mode", default="track", choices=["track", "blur", "center"])
    ap.add_argument("--aspect", default="9:16")
    ap.add_argument("--zoom", type=float, default=1.0,
                    help="1.0 = full-height slice; >1 punches in (e.g. 1.3)")
    ap.add_argument("--analyze-fps", type=float, default=3.0)
    ap.add_argument("--text-weight", type=float, default=2.5,
                    help="How hard to bias framing toward on-screen text (0 = pure saliency)")
    ap.add_argument("--min-conf", type=float, default=45)
    ap.add_argument("--min-shot", type=float, default=0.6,
                    help="Merge scene cuts closer than this (seconds) to avoid twitchy reframes")
    ap.add_argument("--cut-thresh", type=float, default=0.11,
                    help="Scene-cut sensitivity (0-1). Cuts are found by comparing 32x18 "
                         "GRAYSCALE thumbnails, so scenes differing in colour or layout but "
                         "not overall brightness can be missed. Lower to 0.03-0.06 if several "
                         "distinct scenes merge into one shot")
    ap.add_argument("--no-card-fill", action="store_true",
                    help="Disable text-card fit+extend; crop everything instead")
    ap.add_argument("--flat-thr", type=float, default=0.62,
                    help="extend detector: min background flatness (0-1)")
    ap.add_argument("--edge-thr", type=float, default=0.045,
                    help="extend detector: max edge density (0-1)")
    ap.add_argument("--motion-thr", type=float, default=0.02,
                    help="extend detector: max intra-shot motion (animated shots crop/fill instead)")
    ap.add_argument("--band-thr", type=float, default=0.9,
                    help="extend detector: min top/bottom-edge flatness (low = would streak, so crop)")
    ap.add_argument("--fill-zoom", type=float, default=1.5,
                    help="Default scale-up factor for shots directed to 'fill'")
    ap.add_argument("--shots", default=None,
                    help="Path to JSON list of per-shot overrides: "
                         '[{"t0":16.3,"t1":19,"mode":"fill","zoom":1.6}, ...]')
    ap.add_argument("--fallback", default="crop", choices=["crop", "card"],
                    help="Treatment for non-text shots: crop (default) or card (rounded on gradient)")
    ap.add_argument("--dump-shots", action="store_true",
                    help="Print the detected shot plan (times, mode, zoom, center) and exit")
    ap.add_argument("--crf", type=int, default=20)
    a = ap.parse_args()

    overrides = None
    if a.shots:
        import json
        with open(a.shots) as fh:
            overrides = json.load(fh)

    if a.mode == "blur":
        render_blur(a.input, a.output, a.aspect, a.crf)
    else:
        render(a.input, a.output, a.mode, a.aspect, a.zoom, a.analyze_fps,
               a.text_weight, a.min_conf, a.min_shot, a.flat_thr, a.edge_thr,
               a.motion_thr, a.band_thr, not a.no_card_fill, a.fill_zoom,
               overrides, a.dump_shots, a.crf, a.fallback, a.cut_thresh)
    print(a.output)


if __name__ == "__main__":
    main()
