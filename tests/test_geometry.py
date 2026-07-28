"""Unit tests for the pure geometry/timing core of the reframer.

These cover the numeric decisions that determine whether a reel looks intentional or
jittery: where the virtual camera points, how fast it is allowed to move, and how
keyframed moves interpolate. No video fixtures required — every function here is pure.
"""
import numpy as np
import pytest

from reel_maker.reframe import (
    NBINS,
    _ease_out_cubic,
    best_center,
    fit_extend,
    interp_keys,
    smooth_path,
    weighted_center,
)


# --- weighted_center: centroid of the importance map -------------------------------

def test_weighted_center_finds_offcenter_mass():
    """Mass concentrated on the right must pull the center right of frame."""
    imp = np.zeros((18, 32), np.float32)
    imp[:, 24:] = 1.0  # right-hand quarter only
    cx, cy = weighted_center(imp)
    assert cx > 0.7, f"expected center pulled right, got cx={cx}"
    assert cy == pytest.approx(0.5, abs=0.05), "uniform vertical mass should sit mid-frame"


def test_weighted_center_empty_map_defaults_to_middle():
    """An all-zero importance map must not divide by zero — it falls back to center."""
    assert weighted_center(np.zeros((10, 10), np.float32)) == (0.5, 0.5)


# --- best_center: the fixed window that captures the most importance ---------------

def test_best_center_locks_onto_an_offcenter_content_bump():
    """A realistic importance bump should centre the crop window on it, not on the frame.

    Uses a Gaussian rather than a single-bin spike because that is what saliency+OCR
    actually produces, and because a delta creates a plateau of tied windows (see the
    tie-break test below).
    """
    bins = np.arange(NBINS)
    profile = np.exp(-((bins - 200) ** 2) / (2 * 8.0 ** 2)).astype(np.float32)
    cx = best_center(profile, crop_w=270, W=1080)  # window = a quarter of source width
    assert cx == pytest.approx(200 / NBINS, abs=0.02)


def test_best_center_breaks_ties_toward_the_left():
    """Documented behaviour, not an accident: when several windows capture identical
    importance (a single isolated spike), np.argmax takes the leftmost, so the content
    ends up at the window's trailing edge. Harmless on real profiles, which have a
    unique maximum — but pinned here so a refactor can't change it silently."""
    profile = np.zeros(NBINS, np.float32)
    profile[200] = 100.0
    w_bins = 60  # 270/1080 * 240
    cx = best_center(profile, crop_w=270, W=1080)
    expected_start = 200 - w_bins + 1
    assert cx == pytest.approx((expected_start + w_bins / 2) / NBINS, abs=1e-6)


def test_best_center_returns_middle_when_window_covers_everything():
    """If the crop is as wide as the source there is nothing to choose — stay centered."""
    profile = np.random.default_rng(0).random(NBINS).astype(np.float32)
    assert best_center(profile, crop_w=1080, W=1080) == 0.5


# --- smooth_path: velocity clamp within a shot, snap at cuts -----------------------

def _step_inputs():
    """A hard left→right jump in the detected centers, sampled at 2 Hz, output at 30 fps."""
    times = np.array([0.0, 0.5, 1.0, 1.5])
    centers = np.array([[0.2, 0.5], [0.2, 0.5], [0.8, 0.5], [0.8, 0.5]])
    out_times = np.arange(0.0, 1.5, 1 / 30)
    return times, centers, out_times


def test_smooth_path_clamps_velocity_within_a_shot():
    """Without a cut, the camera must never exceed max_speed — this is what stops jitter."""
    times, centers, out_times = _step_inputs()
    max_speed = 0.1  # fraction of frame width per second
    path = smooth_path(times, centers, out_times, smooth_s=0.1,
                       max_speed=max_speed, cuts=np.array([True, False, False, False]))
    steps = np.abs(np.diff(path[:, 0]))
    max_allowed = max_speed / 30.0
    assert steps.max() <= max_allowed + 1e-9, (
        f"camera moved {steps.max():.5f}/frame, clamp was {max_allowed:.5f}")


def test_smooth_path_snaps_across_a_scene_cut():
    """At a cut the clamp must release, so a new shot frames its subject immediately
    instead of gliding in from the previous scene's composition."""
    times, centers, out_times = _step_inputs()
    max_speed = 0.1
    path = smooth_path(times, centers, out_times, smooth_s=0.1, max_speed=max_speed,
                       cuts=np.array([True, False, True, False]))  # cut at t=1.0
    steps = np.abs(np.diff(path[:, 0]))
    assert steps.max() > max_speed / 30.0, "expected an instant reframe at the cut"


# --- fit_extend: width-fit plus edge replication -----------------------------------

def test_fit_extend_produces_exact_output_dimensions():
    """Every treatment must return precisely the output canvas — off-by-one here
    corrupts the ffmpeg pipe."""
    frame = np.zeros((1080, 1920, 3), np.uint8)
    out = fit_extend(frame, out_w=1080, out_h=1920)
    assert out.shape == (1920, 1080, 3)


def test_fit_extend_replicates_edge_rows_rather_than_padding_black():
    """The pad must copy the frame's own edge pixels — black bars are the failure mode
    this whole treatment exists to avoid."""
    frame = np.full((1080, 1920, 3), 200, np.uint8)
    out = fit_extend(frame, out_w=1080, out_h=1920)
    assert out[0].mean() > 190, "top pad should carry the frame's edge tone, not black"
    assert out[-1].mean() > 190, "bottom pad should carry the frame's edge tone, not black"


# --- interp_keys / easing: keyframed moves ----------------------------------------

def test_interp_keys_smoothsteps_between_keyframes():
    """Smoothstep is symmetric, so the midpoint between two keys is the mean value."""
    keys = [{"t": 0.0, "cover": 1.0}, {"t": 2.0, "cover": 2.0}]
    assert interp_keys(keys, 1.0, "cover", 0) == pytest.approx(1.5)


def test_interp_keys_clamps_outside_the_keyframe_range():
    keys = [{"t": 1.0, "cover": 1.0}, {"t": 2.0, "cover": 2.0}]
    assert interp_keys(keys, 0.0, "cover", 99) == 1.0   # before first key
    assert interp_keys(keys, 5.0, "cover", 99) == 2.0   # after last key


def test_interp_keys_falls_back_when_field_absent():
    """A shot may keyframe `cover` but not `cx` — the un-keyed field keeps its default."""
    assert interp_keys([{"t": 0.0, "cover": 1.0}], 0.5, "cx", 0.42) == 0.42


def test_ease_out_cubic_is_monotonic_and_front_loaded():
    assert _ease_out_cubic(0.0) == 0.0
    assert _ease_out_cubic(1.0) == 1.0
    us = np.linspace(0, 1, 50)
    vals = np.array([_ease_out_cubic(u) for u in us])
    assert np.all(np.diff(vals) >= 0), "easing must never move backwards"
    assert _ease_out_cubic(0.5) > 0.5, "ease-OUT decelerates, so it leads the linear ramp"
