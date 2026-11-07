#!/usr/bin/env python3
"""Heartbeat - always-on structural health monitor for the shear-frame tower.

Three accelerometers stream over USB. The tower is tapped; every tap is measured and the
recent ones are pooled into a fingerprint. A fingerprint recorded once, while every nut is
tight, is the baseline, and everything afterwards is judged against it. There is no
recalibration before a demo.

    python monitor.py                      live, reading COM17
    python monitor.py --port COM5          live, another port
    python monitor.py --replay a.csv       replay a recording instead of the hardware
    python monitor.py --list               list serial ports

Then open http://localhost:8000 . Everything else is buttons on that page.

Every training capture is also written to data/ as a plain CSV, so the tower never has to
be taken apart twice: the analysis can be redone from those files afterwards.

Needs: pip install numpy scipy pyserial
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import threading
import time
import traceback
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

import numpy as np
from numpy.fft import rfft, rfftfreq
from scipy.signal import butter, filtfilt, find_peaks, hilbert, welch

HERE = os.path.dirname(os.path.abspath(__file__))
LIBRARY_PATH = os.path.join(HERE, "tower_library.json")
DATA_DIR = os.path.join(HERE, "data")


# ======================================================================================
# 1. Signal processing
#
# Each tap makes the tower ring and die away. Every tap gives:
#
#   f1      first bending frequency       falls when the frame softens
#   f2      second bending frequency      falls most when a middle floor softens
#   zeta    damping ratio of mode 1       rises when a joint rubs instead of gripping
#   slip    how much damping grows with   near zero on a tight frame, clearly positive
#           how hard the tower is hit     on a loose one, whatever strength the taps were
#   share1  floor-1 motion / roof motion  mode shape: says where the softness is
#   share2  floor-2 motion / roof motion
#
# Frequencies come from averaging the spectra of individual tap windows, not one long FFT.
# Repeated taps smear a long FFT into a comb of sidebands whose tallest tooth hops by a few
# Hz between recordings; averaging short windows gives one smooth resonance instead.
# ======================================================================================

FS = 200.0                  # samples per second the firmware is asked for
MODE1_BAND = (5.0, 12.0)    # first bending mode, around 7.8 Hz on this tower
MODE3_BAND = (27.0, 42.0)   # the strong upper mode, around 34 Hz
TAP_BAND = (4.0, 14.0)      # used to find taps and to measure the first mode's shape

# The mode shape is measured in three bands, not one. The first mode bends mostly in the
# bottom storey, so on its own it is nearly blind to a loose joint higher up; the upper
# modes bend most in the upper storeys. Measuring the same ratio in each band is what
# tells floor 2 apart from the roof.
SHAPE_BANDS = {"m1": TAP_BAND, "mid": (14.0, 22.0), "m3": MODE3_BAND}

WIN = 128                   # samples per tap used for the spectrum   (0.64 s)
DECAY = 120                 # samples per tap used for the damping fit (0.60 s)
SHAPE = 40                  # samples per tap for the first mode's shape  (0.20 s)
SHAPE_FAST = 30             # shorter window for the faster upper modes   (0.15 s)
NFFT = 4096
TAP_MIN_G = 0.25            # a tap must swing the roof at least this hard
TAP_GAP = 120               # samples two taps must be apart

AMP_LOW, AMP_HIGH = 0.25, 4.0   # discard only wild outliers; the spread between is wanted
AMP_FLOOR = 0.05                # taps weaker than this are too quiet to fit
VARIETY_MIN = 0.5               # spread of tap strength needed before the slope means anything
SLIP_MIN, SLIP_MAX = -1.0, 1.5  # the ratio is unstable at the edges; a loose floor read 1.24

_FILTERS = {name: butter(2, [lo / (FS / 2), hi / (FS / 2)], "band")
            for name, (lo, hi) in SHAPE_BANDS.items()}


def bandpass(y, band="m1"):
    b, a = _FILTERS[band]
    return filtfilt(b, a, y, axis=1)


def filter_bands(y):
    """The signal split into the bands each mode lives in."""
    return {name: bandpass(y, name) for name in SHAPE_BANDS}


def _shape(filtered, i, width):
    """How far floors 1 and 2 move per unit of roof motion, just after a tap.

    Least squares over a short window, so it reads the shape while the tap's energy is
    still in this band rather than after it has leaked away.
    """
    w = filtered[:, i:i + width]
    power = float(w[2] @ w[2])
    if power <= 0:
        return float("nan"), float("nan")
    return abs(float(w[0] @ w[2]) / power), abs(float(w[1] @ w[2]) / power)


def find_taps(yb, guard=40):
    """Indices of tap peaks in the band-passed roof signal, clear of the buffer edges.

    While the tower is being tapped it is ringing most of the time, so the average level
    says nothing about where a tap is. What marks a tap is a sharp rise standing clear of
    the ring around it, which is what prominence measures. The floor on height stops
    ordinary noise counting as a tap once the tapping has stopped.
    """
    env = np.abs(hilbert(yb[2]))
    loud = float(np.percentile(env, 98))
    peaks, _ = find_peaks(env, height=max(TAP_MIN_G, 0.45 * loud),
                          prominence=0.25 * loud, distance=TAP_GAP)
    n = yb.shape[1]
    return [int(i) for i in peaks if i >= guard and i + max(WIN, DECAY) + guard < n]


def tap_record(y, bands, i):
    """Everything measurable from one tap, except what needs the pooled spectrum."""
    yb = bands["m1"]
    share1, share2 = _shape(yb, i, SHAPE)
    if not np.isfinite(share1):
        return None

    # The same ratio in the two upper bands. These are what separate a loose floor 2 from
    # a loose roof: the first mode barely notices the difference, the upper modes do.
    shapes = {}
    for name in ("mid", "m3"):
        a, b = _shape(bands[name], i, SHAPE_FAST)
        shapes["s1_" + name], shapes["s2_" + name] = a, b

    # Decay rate: slope of log(envelope) while the ring is above a quarter of its peak.
    env = np.abs(hilbert(yb[2, i:i + DECAY]))
    peak = float(env.max())
    sigma = float("nan")
    if peak > 0:
        e = env / peak
        keep = e > 0.25
        if keep.any() and (~keep).any():
            keep[np.argmax(~keep):] = False     # stop at the first dip, before any later tap
        if keep.sum() >= 15:
            t = np.arange(len(e))[keep] / FS
            sigma = -float(np.polyfit(t, np.log(e[keep]), 1)[0])

    return dict(shapes, share1=share1, share2=share2, sigma=sigma,
                amplitude=float(abs(yb[2, i])),
                window=np.asarray(y[2, i:i + WIN], dtype=float))


def _centre_of_peak(freqs, mags, band):
    """Centre of mass of the upper half of the tallest peak in a band.

    Steadier than the single tallest bin, which hops between neighbours when two taps
    land slightly differently.
    """
    sel = (freqs > band[0]) & (freqs < band[1])
    f, m = freqs[sel], mags[sel]
    if not len(f) or m.max() <= 0:
        return float("nan")
    top = m >= 0.5 * m.max()
    return float((f[top] * m[top]).sum() / m[top].sum())


def fit_damping(amps, sigmas):
    """Straight-line fit of decay rate against tap strength: sigma = a + b * amplitude.

    A tight bolted joint behaves the same however hard it is hit, so b is about zero. A
    loose one slips, and the slip grows with the force, so b is clearly positive. That
    makes b a damage indicator that does not care how hard the tower was tapped, which
    matters because nothing outside a laboratory taps a building the same way twice.
    """
    keep = amps > AMP_FLOOR
    a, s = amps[keep], sigmas[keep]
    if len(a) < 6:
        return float("nan"), float("nan")

    # A tap that lands while the tower is still ringing from the last one gets its decay
    # measured across two overlapping rings, which reports a wildly wrong rate. One such
    # point is enough to drag the whole straight line, so they go before anything is fitted.
    mid = float(np.median(s))
    if mid > 0:
        sane = (s > 0.15 * mid) & (s < 3.0 * mid)
        if sane.sum() >= 6:
            a, s = a[sane], s[sane]

    b, a0 = np.polyfit(a, s, 1)
    resid = s - (a0 + b * a)
    # Median absolute deviation rather than the standard deviation: an outlier inflates the
    # standard deviation enough to keep itself inside the cut it is supposed to fail.
    scale = 1.4826 * float(np.median(np.abs(resid - np.median(resid))))
    if scale > 0:
        good = np.abs(resid) < 2.5 * scale
        if good.sum() >= 6:
            b, a0 = np.polyfit(a[good], s[good], 1)
    return float(a0), float(b)


def fingerprint(taps):
    """Pool tap records into one fingerprint, or None if there are too few taps."""
    taps = [t for t in taps if t is not None and len(t["window"]) == WIN]
    if len(taps) < 4:
        return None

    amps = np.array([t["amplitude"] for t in taps])
    middle = float(np.median(amps))
    if middle > 0:
        kept = [t for t, a in zip(taps, amps) if AMP_LOW * middle <= a <= AMP_HIGH * middle]
        if len(kept) >= 4:
            taps = kept
            amps = np.array([t["amplitude"] for t in taps])
            middle = float(np.median(amps))

    window = np.hanning(WIN)
    spectrum = sum(np.abs(rfft(t["window"] * window, NFFT)) for t in taps)
    freqs = rfftfreq(NFFT, 1.0 / FS)
    f1 = _centre_of_peak(freqs, spectrum, MODE1_BAND)
    f2 = _centre_of_peak(freqs, spectrum, MODE3_BAND)

    sigmas = np.array([t["sigma"] for t in taps])
    usable = np.isfinite(sigmas)
    damp_a, damp_b = fit_damping(amps[usable], sigmas[usable])
    variety = (float(np.percentile(amps, 90) - np.percentile(amps, 10)) / middle) if middle > 0 else 0.0

    def middle_of(key):
        vals = [t[key] for t in taps if np.isfinite(t.get(key, np.nan))]
        return float(np.median(vals)) if len(vals) >= 4 else float("nan")

    return {"f1": f1, "f2": f2,
            "share1": middle_of("share1"), "share2": middle_of("share2"),
            "s1_mid": middle_of("s1_mid"), "s2_mid": middle_of("s2_mid"),
            "s1_m3": middle_of("s1_m3"), "s2_m3": middle_of("s2_m3"),
            "damp_a": damp_a, "damp_b": damp_b,
            "sigma_med": float(np.median(sigmas[usable])) if usable.any() else float("nan"),
            "amp_med": middle, "amp_lo": float(amps.min()), "amp_hi": float(amps.max()),
            "variety": variety, "taps": len(taps)}


def rescale(fp, measured_rate):
    """Correct for the sample rate actually received.

    Everything above counts samples and assumes FS of them per second. If the laptop only
    took in, say, 175 of the 200 the board sent, every frequency and every decay rate comes
    out low by that same fraction and the baseline is no longer comparable. Ratios such as
    damping and mode shape look after themselves, since both halves scale alike.
    """
    if fp is None:
        return None
    if not np.isfinite(measured_rate) or not (0.5 * FS < measured_rate < 2.0 * FS):
        return fp
    k = measured_rate / FS
    out = dict(fp)
    for key in ("f1", "f2", "sigma_med", "damp_a", "damp_b"):
        value = out.get(key)
        if value is not None and np.isfinite(value):
            out[key] = value * k
    out["rate"] = float(measured_rate)
    return out


def derive(fp):
    """The six comparable numbers.

    Damping is the plain median over the taps. Normalising it to one fixed tap strength was
    tried and measured worse: predicting the fitted line away from the strengths actually
    used adds more error than the amplitude dependence it removes. That dependence is
    carried by `slip` instead, where it is the signal rather than a nuisance.
    """
    f1 = fp["f1"]
    a0, b = fp.get("damp_a"), fp.get("damp_b")
    # The slope is divided by the intercept so the number means "fractional rise in damping
    # per g of tap" and does not depend on the units. The intercept is an extrapolation to
    # zero tap strength, though, and over a short window it can land near zero and send the
    # ratio to absurd values - a sound tower once read 28, against 0.5 for a genuinely loose
    # joint, which drove the alarm to 95. Require a positive intercept and hold the ratio
    # inside the range physically reachable on this tower; the widest reading on a loosened
    # floor was 1.24, so the bound discards nothing real.
    fitted = (a0 is not None and b is not None and np.isfinite(a0) and np.isfinite(b)
              and a0 > 0 and fp["variety"] >= VARIETY_MIN)
    slip = float(np.clip(b / a0, SLIP_MIN, SLIP_MAX)) if fitted else float("nan")

    sigma = fp.get("sigma_med", float("nan"))
    zeta = sigma / (2 * np.pi * f1) if np.isfinite(sigma) and np.isfinite(f1) and f1 else float("nan")
    out = {"f1": f1, "f2": fp["f2"], "zeta": zeta, "slip": slip}
    for key in ("share1", "share2", "s1_mid", "s2_mid", "s1_m3", "s2_m3"):
        out[key] = fp.get(key, float("nan"))
    return out


SPECTRUM_SECONDS = 6.0
SPECTRUM_TOP = 45.0
LEVELS = ("Level 1", "Level 2", "Roof")


def live_spectra(data, rate):
    """A running spectrum for each sensor, for the display only.

    This is deliberately separate from everything the verdict uses. It is not tied to taps,
    so it keeps moving between them and shows the tower's response to whatever is shaking
    it - including nothing but the room. The verdict still comes from the tap fingerprint.
    """
    width = min(data.shape[1], int(SPECTRUM_SECONDS * FS))
    if width < 512:
        return None
    segment = data[:, -width:]
    scale = rate / FS if (np.isfinite(rate) and 0.5 * FS < rate < 2.0 * FS) else 1.0

    freqs = None
    sensors = []
    for i in range(3):
        f, power = welch(segment[i], FS, nperseg=512)
        f = f * scale
        keep = f <= SPECTRUM_TOP
        f, power = f[keep], power[keep]
        band = (f > MODE1_BAND[0]) & (f < MODE1_BAND[1])
        peak, height = float("nan"), float("nan")
        if band.any():
            where = int(np.flatnonzero(band)[int(np.argmax(power[band]))])
            peak, height = float(f[where]), float(np.sqrt(power[where]))
            # The bins are about 0.4 Hz apart, which is coarser than the differences worth
            # seeing between sensors. Fitting a parabola to the peak and its two neighbours
            # recovers where the true peak sits between them.
            if 0 < where < len(f) - 1:
                a, b, c = power[where - 1], power[where], power[where + 1]
                curve = a - 2 * b + c
                if curve != 0:
                    shift = float(np.clip(0.5 * (a - c) / curve, -0.5, 0.5))
                    peak += shift * float(f[1] - f[0])
        if freqs is None:
            freqs = [round(float(v), 2) for v in f]
        sensors.append({
            "name": LEVELS[i],
            "peak": peak,
            "height": height,
            "rms": float(segment[i].std()),
            "curve": [round(float(v), 7) for v in np.sqrt(power)],
        })
    return {"freqs": freqs, "sensors": sensors}


def true_rate(times):
    """Samples per second according to the board's own clock, and the fraction missing.

    The firmware stamps every reading, so if the laptop cannot keep up and loses some, the
    gap shows here. That matters: a frequency worked out by counting samples is wrong by
    exactly the fraction that went missing.
    """
    gaps = np.diff(np.asarray(times, dtype=float))
    gaps = gaps[(gaps > 0) & (gaps < 1e6)]        # drop the wrap of the microsecond clock
    if len(gaps) < 50:
        return float("nan"), 0.0
    step = float(np.median(gaps))
    return 1e6 / step, float(np.mean(gaps > 1.5 * step))


def load_csv(path):
    """A recording from record.py or a saved capture, as a 3 x N array."""
    d = np.genfromtxt(path, delimiter=",", names=True)
    return np.vstack([d[k + "y"] for k in ("f1", "f2", "f3")])


def load_csv_timed(path):
    """Same, with the board's microsecond clock as a fourth row on top."""
    d = np.genfromtxt(path, delimiter=",", names=True)
    return np.vstack([d["t_us"]] + [d[k + "y"] for k in ("f1", "f2", "f3")])


def analyse_block(y):
    """Fingerprint a whole 3 x N block of side-to-side acceleration."""
    y = np.asarray(y, dtype=float)
    y = y - y.mean(axis=1, keepdims=True)
    bands = filter_bands(y)
    return fingerprint([tap_record(y, bands, i) for i in find_taps(bands["m1"])])


# ======================================================================================
# 2. Diagnosis
#
# Two questions, answered separately on purpose.
#   Is anything loose?  One number, the alarm index, from the drop in frequency, the rise
#                       in damping, and the appearance of slip. All three move the same way
#                       for any loosened joint, so this works on a joint never seen loose.
#   Which floor?        The pattern across the steady numbers, matched against examples
#                       taken on this tower. Needs examples, and says so when it has none.
# ======================================================================================

FEATURES = ("f1", "f2", "zeta", "slip",
            "share1", "share2", "s1_mid", "s2_mid", "s1_m3", "s2_m3")
RELATIVE = {"f1", "f2", "zeta"}        # compared as a percent change; the rest as a difference
LABELS = {"healthy": "All clear", "floor1": "Level 1 loose",
          "floor2": "Level 2 loose", "roof": "Roof loose"}
TRAINABLE = ("floor1", "floor2", "roof")

# How far each number drifts by itself between two readings of the tower in the same state,
# measured across repeated captures. One unit of distance is one of these.
TOLERANCE = np.array([0.012, 0.016, 0.070, 0.300,
                      0.015, 0.015, 0.250, 0.120, 0.050, 0.050])

# Slip is a strong alarm but a poor pointer: fitting a slope needs a spread of tap
# strengths, and over a short window that slope is noisy. The frequencies and the mode
# shapes, which are steady, are what name the floor.
LOCATE = [i for i, name in enumerate(FEATURES) if name != "slip"]

# Which numbers, and in which direction, say the frame has softened. A loosened joint
# anywhere drops the first frequency; one high up barely touches it but drops the upper
# mode sharply, so both count.
ALARM_TERMS = ((0, -1.0), (1, -1.0), (2, 1.0), (3, 1.0))

# Measured on this tower over fourteen-tap readings across eight captures: every sound
# state, including three separate retightenings, stayed under 4.1, and every loosened floor
# stayed above 6.5. The thresholds sit in that gap, so neither a false alarm nor a missed
# floor occurred. They were re-derived after the damping fit was made robust; the earlier
# pair, 4.0 and 5.0, came from a narrower set of captures.
ALARM_WARN = 4.5            # above this, something has changed
ALARM_TRIP = 5.5            # above this, call it a loose joint
# Which floor is read from the DIRECTION the fingerprint moved, not the distance. Loosening
# a nut further pushes the same pattern further without changing its shape, so a half-turn
# and a full turn point the same way. Matching on distance instead meant a gentle loosening
# of the roof landed nearer the recorded floor 2 than the recorded roof.
# Measured on this tower: the right floor sits within 0.27, the next-best never closer than
# 0.70, and a sound tower is at least 1.27 from all of them.
MATCH_LIMIT = 0.90          # further than this from every example: say so, do not guess
MATCH_MARGIN = 0.15         # two floors this close together: say it is between them


def delta(current, base):
    out = []
    for name in FEATURES:
        a, b = current.get(name), base.get(name)
        if a is None or b is None or not np.isfinite(a) or not np.isfinite(b) or b == 0:
            out.append(np.nan)
        elif name in RELATIVE:
            out.append(a / b - 1.0)
        else:
            out.append(a - b)
    return np.array(out, dtype=float)


def alarm_index(dv):
    """Distance from healthy in units of normal drift.

    Softening means falling frequencies, rising damping, and damping starting to depend on
    how hard the tower is hit. Only those directions count: a frequency that rises, or
    damping that falls, is not damage.
    """
    parts = []
    for i, sign in ALARM_TERMS:
        if np.isfinite(dv[i]):
            parts.append(max(0.0, sign * dv[i]) / TOLERANCE[i])
    return float(np.linalg.norm(parts)) if parts else 0.0


def _unit(vector):
    """A delta in tolerance units, scaled to length one. None if too little is measurable."""
    v = (np.asarray(vector, dtype=float) / TOLERANCE)[LOCATE]
    good = np.isfinite(v)
    if good.sum() < 3:
        return None
    v = np.where(good, v, 0.0)
    length = float(np.linalg.norm(v))
    return (v / length, good) if length > 0 else None


def direction_gap(dv, examples):
    """How far the current change points from a set of recorded ones, ignoring size.

    Each example is scaled to length one before averaging, so an example recorded from a
    hard loosening carries no more weight than one from a gentle one. The answer is the
    chord between the two directions: 0 means identical, 2 means opposite.
    """
    here = _unit(dv)
    if here is None:
        return None
    pointing = [u for u in (_unit(e) for e in examples) if u is not None]
    if not pointing:
        return None
    centre = np.mean([u for u, _ in pointing], axis=0)
    length = float(np.linalg.norm(centre))
    if length == 0:
        return None
    return float(np.sqrt(max(0.0, 2.0 * (1.0 - float(here[0] @ (centre / length))))))


class Library:
    """Stored baseline fingerprint plus learned examples, saved beside this script."""

    def __init__(self, path):
        self.path = path
        self.baseline = None
        self.examples = {}
        self.load()

    def load(self):
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path) as fh:
                blob = json.load(fh)
        except (OSError, ValueError):
            print("Could not read the saved library; starting fresh.")
            return
        baseline = blob.get("baseline")
        if isinstance(baseline, dict) and all(k in baseline for k in FEATURES):
            self.baseline = {k: float(baseline[k]) for k in FEATURES}
        elif baseline:
            print("Ignoring a reference saved by an older version. Press Set reference again.")
            return
        examples = {}
        for label, rows in (blob.get("examples") or {}).items():
            kept = [np.array(r, dtype=float) for r in rows if len(r) == len(FEATURES)]
            if kept:
                examples[label] = kept
        self.examples = examples

    def save(self):
        blob = {"baseline": self.baseline,
                "examples": {k: [list(map(float, v)) for v in vs]
                             for k, vs in self.examples.items()}}
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(blob, fh, indent=1)
        os.replace(tmp, self.path)

    def set_baseline(self, fp):
        self.baseline = {k: float(v) for k, v in derive(fp).items()}
        self.examples.clear()          # old examples are offsets from the old baseline
        self.save()

    def add_example(self, label, dv):
        finite = np.isfinite(dv)
        if finite[LOCATE].sum() < 3:   # too little to point at a floor with
            return False
        self.examples.setdefault(label, []).append(np.asarray(dv, dtype=float))
        self.save()
        return True

    def forget(self, label):
        self.examples.pop(label, None)
        self.save()

    def counts(self):
        return {k: len(v) for k, v in self.examples.items()}


def diagnose(fp, lib):
    if lib.baseline is None:
        return {"state": "no_baseline", "headline": "No reference",
                "detail": "Tighten every nut, press Set reference, then tap for 45 seconds.",
                "alarm": 0.0, "delta": None, "scores": {}, "floor": None, "values": None}

    values = derive(fp)
    dv = delta(values, lib.baseline)
    alarm = alarm_index(dv)
    out = {"alarm": alarm, "floor": None, "values": values,
           "delta": {n: float(v) for n, v in zip(FEATURES, dv)}}

    if alarm < ALARM_WARN:
        out.update(state="healthy", headline=LABELS["healthy"],
                   detail="Frequency and damping match the reference.", scores={})
        return out

    scores = {}
    for label, examples in lib.examples.items():
        if label == "healthy" or not examples:
            continue
        gap = direction_gap(dv, examples)
        if gap is not None:
            scores[label] = gap

    severity = "Connection loose" if alarm >= ALARM_TRIP else "Change detected"
    out["scores"] = scores

    if not scores:
        out.update(state="unlearned", headline=severity,
                   detail="No examples recorded yet, so the floor cannot be named. "
                          "Use the Learn buttons once per floor.")
        return out

    ranked = sorted(scores, key=scores.get)
    best = ranked[0]
    margin = (scores[ranked[1]] - scores[best]) if len(ranked) > 1 else float("inf")

    # Far from every example, and no clear winner among them: do not guess. A badly
    # loosened joint lands far from its own example too, so a big lead over the runners-up
    # still counts as a match even when the absolute distance is large.
    if scores[best] > MATCH_LIMIT and margin < MATCH_LIMIT:
        out.update(state="unknown", headline=severity,
                   detail="The pattern matches no recorded floor. The tower may have been "
                          "moved, or a joint we have not learned is loose.")
    elif margin < MATCH_MARGIN:
        out.update(state="ambiguous", headline=severity, floor=best,
                   detail=f"Closest to {LABELS[best]}, but {LABELS[ranked[1]]} is nearly as "
                          "close. Keep tapping.")
    else:
        out.update(state="located", headline=LABELS[best], floor=best,
                   detail=f"Matches the recorded pattern for {LABELS[best]}.")
    return out


# ======================================================================================
# 3. Making values safe to send
# ======================================================================================

def jsonable(value):
    """Convert to something a browser can parse.

    Python writes a bare NaN into JSON and every browser refuses to parse it, which kills
    the whole page rather than one number. Anything not finite becomes null here.
    """
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    return str(value)


# ======================================================================================
# 4. Where the readings come from
# ======================================================================================

class SerialSource:
    """Rows of [clock, floor1, floor2, roof] streaming from the Feather."""

    name = "serial"

    def __init__(self, port, baud=115200):
        self.port = port
        self.baud = baud
        self.status = "connecting"
        self._serial = None

    def rows(self, stop):
        try:
            import serial
        except ImportError:
            self.status = "pyserial is not installed - run: pip install pyserial"
            return
        while not stop.is_set():
            if self._serial is None:
                try:
                    self._serial = serial.Serial(self.port, self.baud, timeout=1)
                    self._serial.reset_input_buffer()
                    self.status = f"connected on {self.port}"
                except Exception as exc:
                    self.status = f"cannot open {self.port} ({exc.__class__.__name__}) - "\
                                  "is the Arduino Serial Monitor still open?"
                    time.sleep(2.0)
                    continue
            try:
                line = self._serial.readline().decode("ascii", "ignore").strip()
            except Exception:
                self.status = f"lost {self.port}, retrying"
                try:
                    self._serial.close()
                except Exception:
                    pass
                self._serial = None
                time.sleep(1.0)
                continue
            if not line or line[0] in "#t":
                continue
            parts = line.split(",")
            if len(parts) != 11:
                continue
            try:
                yield (float(parts[0]), float(parts[2]), float(parts[5]), float(parts[8]))
            except ValueError:
                continue

    def close(self):
        if self._serial is not None:
            try:
                self._serial.close()
            except Exception:
                pass


class ReplaySource:
    """A saved recording, played back at real speed. Doubles as a backup demo."""

    name = "replay"

    def __init__(self, path, loop=True):
        self.path = path
        self.loop = loop
        self.status = f"replaying {os.path.basename(path)}"
        self.data = load_csv_timed(path).T          # N x 4

    def rows(self, stop):
        step = 1.0 / FS
        while not stop.is_set():
            started = time.time()
            for n, row in enumerate(self.data):
                if stop.is_set():
                    return
                yield (float(row[0]), float(row[1]), float(row[2]), float(row[3]))
                ahead = started + (n + 1) * step - time.time()
                if ahead > 0.002:
                    time.sleep(ahead)
            if not self.loop:
                self.status = "replay finished"
                return

    def close(self):
        pass


# ======================================================================================
# 5. The monitor
#
# Two threads and no shared mutable data between them. The reader only parses lines and
# posts them to a queue; the analyser owns every buffer it touches and publishes a finished,
# immutable snapshot for the web page to read. An earlier version had both threads fighting
# over one lock, which made the reader fall behind and lose readings.
# ======================================================================================

BUFFER_SECONDS = 25
MIN_SECONDS = 6
ANALYSE_EVERY = 0.3        # the live spectrum updates this often
WAVE_SECONDS = 3.2         # how much raw signal the trace keeps on screen
WAVE_STEP = 1              # full rate: the page runs its own spectrum off this,
                           # and decimating first would fold higher noise into it
TAP_MEMORY = 14             # taps pooled into the live verdict, for a steady reading
TAP_MAX_AGE = 30.0          # older taps are dropped so a stale verdict cannot stick
STALE_AFTER = 6.0           # no taps for this long and the verdict is marked as old
TRAIN_SECONDS = 45.0        # a baseline or an example is pooled over this much tapping

# Pooling 14 taps makes a steady verdict but a slow one: after a nut is tightened the old
# taps have to be tapped out one by one. So the newest few are also checked on their own,
# and when they disagree with the pool about whether anything is loose at all, the older
# taps are thrown away and the verdict catches up at once. On this tower four taps put a
# tight frame under 2.5 and any loose floor above 5.9, so the two are easy to tell apart.
RECENT = 4
FLUSH_COOLDOWN = 3.0

# The two directions are not equally risky, so they are not treated alike. Over every
# four-tap reading in eight captures, a sound tower never passed 4.98 and a loosened floor
# never fell below 5.93, so 5.4 sits in that gap: under it means the newest taps say the
# frame is sound, and that acts at once. The other way round is noisier, so declaring new
# damage waits for a clearly higher reading, twice in a row.
# The earlier value here was 2.0, derived before a joint worked loose and was retightened;
# the retightened tower reads 2.2 to 4.3 on four taps, which that threshold never cleared,
# and recovery had to wait for the old taps to age out instead.
FLUSH_TIGHT = 5.4
FLUSH_LOOSE = 5.5
FLUSH_REPEATS = 2

WAITING = {"state": "waiting", "headline": "Waiting for taps",
           "detail": "Tap the roof floor about once a second.",
           "alarm": 0.0, "delta": None, "scores": {}, "floor": None, "values": None}

# What a held verdict keeps while a new one is being confirmed: the words and the floor it
# points at. Everything else - the alarm, the ten numbers, the deltas - stays live.
CARRIED = ("state", "headline", "detail", "floor")

RECHECK = {"state": "recheck", "headline": "Tower changed",
           "detail": "Something was tightened or loosened. Keep tapping; "
                     "the new reading needs four taps.",
           "alarm": 0.0, "delta": None, "scores": {}, "floor": None, "values": None}


class Monitor:
    def __init__(self, source, library, train_seconds=TRAIN_SECONDS, save_captures=True):
        self.source = source
        self.lib = library
        self.train_seconds = train_seconds
        self.save_captures = save_captures

        self.incoming = queue.SimpleQueue()     # reader -> analyser
        self.commands = queue.SimpleQueue()     # web page -> analyser
        self.state = {"starting": True}         # analyser -> web page (rebound, never edited)
        self.stop_flag = threading.Event()

        self.received = 0                       # reader thread only
        self.arrival_rate = 0.0

        self.taps = deque(maxlen=TAP_MEMORY)    # analyser thread only, below here
        self.seen = deque(maxlen=600)
        self.history = deque(maxlen=600)
        self.job = None
        self.notice = ""
        self.last_error = ""
        self.last_flush = 0.0
        self.rechecking = False
        self.last_judged = None
        self.agreeing = 0
        self.held = None                        # the verdict currently on screen
        self.candidate = None                   # a different one, waiting to be confirmed
        self.candidate_taps = 0
        self.spectra = None
        self.wave = None

    # -- reader -------------------------------------------------------------------
    def read_loop(self):
        count, mark = 0, time.time()
        try:
            for row in self.source.rows(self.stop_flag):
                self.incoming.put(row)
                self.received += 1
                count += 1
                if count >= 200:
                    now = time.time()
                    self.arrival_rate = count / max(now - mark, 1e-6)
                    count, mark = 0, now
        except Exception:
            self.source.status = "reader stopped: " + traceback.format_exc(limit=1).strip()

    # -- analyser -----------------------------------------------------------------
    def analyse_loop(self):
        buffer = deque(maxlen=int(BUFFER_SECONDS * FS))
        total = 0
        while not self.stop_flag.is_set():
            deadline = time.time() + ANALYSE_EVERY
            while True:                                  # drain whatever has arrived
                left = deadline - time.time()
                if left <= 0:
                    break
                try:
                    row = self.incoming.get(timeout=min(left, 0.2))
                except queue.Empty:
                    break
                buffer.append(row)
                total += 1
                if self.job is not None:
                    self.job["rows"].append(row)
            self.publish_wave(buffer, total)
            try:
                self.cycle(buffer, total)
            except Exception:
                self.last_error = traceback.format_exc(limit=2).strip().splitlines()[-1]
                self.publish(buffer, None, WAITING, float("nan"), 0.0)

    def publish_wave(self, buffer, total):
        """The newest slice of raw signal, published on its own.

        Separate from the verdict, and sent as soon as the readings are drained rather than
        after the analysis, so the trace on screen is as close to live as the link allows.
        Each block carries the absolute sample number of its first reading, which lets the
        page stitch blocks together and scroll smoothly between them.
        """
        if not buffer:
            return
        span = min(len(buffer), int(WAVE_SECONDS * FS))
        rows = list(buffer)[-span:][::WAVE_STEP]
        first = total - span
        self.wave = {
            "first": first,
            "step": WAVE_STEP,
            "rate": FS / WAVE_STEP,
            "levels": [[round(r[1], 3) for r in rows],
                       [round(r[2], 3) for r in rows],
                       [round(r[3], 3) for r in rows]],
        }

    def cycle(self, buffer, total):
        while True:
            try:
                kind, label = self.commands.get_nowait()
            except queue.Empty:
                break
            self.begin(kind, label)

        if len(buffer) < int(MIN_SECONDS * FS):
            self.publish(buffer, None, WAITING, float("nan"), 0.0)
            return

        block = np.array(buffer, dtype=float).T          # private to this thread
        clock, data = block[0], block[1:]
        rate, lost = true_rate(clock)
        data = data - data.mean(axis=1, keepdims=True)
        self.spectra = live_spectra(data, rate)
        bands = filter_bands(data)

        now = time.time()
        start = total - block.shape[1]
        for i in find_taps(bands["m1"]):
            position = start + i
            if any(abs(position - s) < TAP_GAP for s in self.seen):
                continue
            record = tap_record(data, bands, i)
            if record is None:
                continue
            record["at"] = now
            self.seen.append(position)
            self.taps.append(record)
            if self.job is not None:
                self.job["taps"].append(record)

        if self.job is not None and now >= self.job["deadline"]:
            self.finish(rate)

        live = [t for t in self.taps if now - t["at"] < TAP_MAX_AGE]
        fp = rescale(fingerprint(live), rate)
        if fp is not None:
            fp, live = self.catch_up(fp, live, rate, now)
        if fp is None:
            self.publish(buffer, None, RECHECK if self.rechecking else WAITING, rate, lost)
            return
        self.rechecking = False

        verdict = dict(diagnose(fp, self.lib), age=round(now - live[-1]["at"], 1))
        verdict = self.settle(verdict, fp["taps"])
        shown = verdict.get("values") or derive(fp)
        self.history.append({"f1": shown["f1"], "alarm": verdict["alarm"]})
        self.publish(buffer, dict(shown, taps=fp["taps"], variety=fp["variety"]),
                     verdict, rate, lost, len(live))

    def settle(self, verdict, taps):
        """Hold the headline until a different one survives a further tap.

        One unlucky tap can shift a fourteen-tap pooling far enough to change the answer for
        a moment, and the next tap puts it back. The numbers below keep updating either way;
        only the words wait. A real change still shows within one extra tap, because every
        tap after it agrees, and the flush that clears stale taps is untouched.
        """
        word = (verdict.get("state"), verdict.get("floor"))
        if self.held is None or word == self.held[0]:
            self.held = (word, {k: verdict.get(k) for k in CARRIED})
            self.candidate = None
            return verdict

        if self.candidate != word:
            self.candidate, self.candidate_taps = word, taps
        elif taps != self.candidate_taps:          # a further tap, same answer: accept it
            self.held = (word, {k: verdict.get(k) for k in CARRIED})
            self.candidate = None
            return verdict
        return dict(verdict, **self.held[1], settling=True)

    def catch_up(self, fp, live, rate, now):
        """Throw away stale taps when the tower has clearly just changed state.

        The newest few taps are judged on their own. If they say the frame is sound while
        the pooled reading still says a joint is loose - or the other way round - then
        something was tightened or loosened a moment ago, and the older taps are describing
        a tower that no longer exists.
        """
        if (self.lib.baseline is None or len(live) <= RECENT
                or now - self.last_flush < FLUSH_COOLDOWN):
            return fp, live

        recent = live[-RECENT:]
        fp_recent = rescale(fingerprint(recent), rate)
        if fp_recent is None:
            return fp, live

        newest = alarm_index(delta(derive(fp_recent), self.lib.baseline))
        pooled = alarm_index(delta(derive(fp), self.lib.baseline))
        tightened = newest < FLUSH_TIGHT and pooled >= ALARM_TRIP
        loosened = newest >= FLUSH_LOOSE and pooled < ALARM_WARN

        # Only count a fresh opinion once per tap; the analysis runs twice a second and
        # would otherwise confirm the same four taps over and over.
        stamp = id(recent[-1])
        if stamp != self.last_judged:
            self.last_judged = stamp
            self.agreeing = self.agreeing + 1 if (tightened or loosened) else 0

        if not (tightened or (loosened and self.agreeing >= FLUSH_REPEATS)):
            return fp, live
        self.agreeing = 0

        # Keep only the newest couple of taps. A pool that mixes the old state with the new
        # one reads as some third floor, which would flash a wrong answer on screen, so
        # almost everything goes; the two newest are the ones most likely to be past the
        # change. Until the pool refills the banner says the tower changed rather than
        # naming a floor. Better to show nothing for a few taps than to show a lie.
        newest = live[-2:]
        self.taps.clear()
        self.taps.extend(newest)
        self.last_flush = now
        self.rechecking = True
        self.held = None            # the flush is itself the confirmation; do not hold again
        self.candidate = None
        self.notice = "Tower changed - re-reading from fresh taps."
        return None, []

    # -- captures -----------------------------------------------------------------
    def begin(self, kind, label):
        self.job = {"kind": kind, "label": label, "taps": [], "rows": [],
                    "deadline": time.time() + self.train_seconds}
        self.notice = (f"Recording for {self.train_seconds:.0f} s - keep tapping, "
                       "and vary how hard you tap.")

    def finish(self, rate):
        job, self.job = self.job, None
        saved = self.write_capture(job) if self.save_captures else None
        tail = f" Raw data saved to {os.path.basename(saved)}." if saved else ""

        fp = rescale(fingerprint(job["taps"]), rate)
        if fp is None:
            self.notice = (f"Only {len(job['taps'])} taps in that capture, which is too few. "
                           f"Tap about once a second and try again.{tail}")
            return

        note = ""
        if fp["variety"] < VARIETY_MIN:
            note = (" Your taps were all much alike, so slip sensitivity was not measured - "
                    "mix in harder and softer ones.")

        if job["kind"] == "baseline":
            self.lib.set_baseline(fp)
            self.notice = f"Reference set from {fp['taps']} taps: {fp['f1']:.2f} Hz.{note}{tail}"
        elif self.lib.baseline is None:
            self.notice = f"Set a reference first.{tail}"
        elif self.lib.add_example(job["label"], delta(derive(fp), self.lib.baseline)):
            kept = len(self.lib.examples[job["label"]])
            self.notice = (f"Learned {LABELS[job['label']]} from {fp['taps']} taps "
                           f"({kept} example{'s' if kept > 1 else ''} now).{note}{tail}")
        else:
            self.notice = f"That capture was incomplete, so nothing was learned.{tail}"

    def write_capture(self, job):
        """Save the raw readings so the tower never has to be taken apart twice."""
        if not job["rows"]:
            return None
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            name = f"train_{job['label'] or 'baseline'}_{time.strftime('%H%M%S')}.csv"
            path = os.path.join(DATA_DIR, name)
            with open(path, "w") as fh:
                fh.write("t_us,f1y,f2y,f3y\n")
                for r in job["rows"]:
                    fh.write(f"{int(r[0])},{r[1]:.4f},{r[2]:.4f},{r[3]:.4f}\n")
            return path
        except OSError:
            return None

    # -- publishing ---------------------------------------------------------------
    def publish(self, buffer, shown, verdict, rate, lost, live_taps=0):
        tail = list(buffer)[-int(5 * FS):]
        job = self.job
        notice, self.notice = self.notice, ""
        self.state = jsonable({
            "source": self.source.name,
            "status": getattr(self.source, "status", ""),
            "arrival_rate": round(self.arrival_rate, 1),
            "device_rate": rate,
            "expected_rate": FS,
            "lost": round(100 * lost, 1),
            "samples": self.received,
            "taps": live_taps,
            "fingerprint": shown,
            "baseline": self.lib.baseline,
            "examples": self.lib.counts(),
            "verdict": verdict,
            "history": [h["f1"] for h in list(self.history)[-120:]],
            "wave": [round(r[3], 3) for r in tail[::3]],
            "spectra": self.spectra,
            "notice": notice,
            "error": self.last_error,
            "collecting": None if job is None else {
                "kind": job["kind"], "label": job["label"],
                "left": max(0.0, round(job["deadline"] - time.time(), 1)),
                "taps": len(job["taps"])},
            "busy": job is not None,
            "labels": LABELS,
            "thresholds": {"warn": ALARM_WARN, "trip": ALARM_TRIP},
            "train_seconds": self.train_seconds,
        })

    def request(self, kind, label=None):
        self.commands.put((kind, label))

    def stop(self):
        self.stop_flag.set()
        self.source.close()


# ======================================================================================
# 6. Web dashboard
# ======================================================================================

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Heartbeat</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<style>
:root{
  --paper:#FFFFFF; --panel:#FFFFFF; --ink:#16202B; --ink-soft:#5C6875; --ink-faint:#8D97A2;
  --rule:#E4E8EC; --rule-strong:#C8D0D7; --field:#F4F6F8;
  --s1:#74A3CC; --s2:#2E6FA7; --s3:#0F3D6B;
  --ok:#1E7A4D; --watch:#96670F; --alert:#B3261E;
  --sans:"IBM Plex Sans",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  color-scheme:light;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%}
body{background:var(--paper);color:var(--ink);font:400 15px/1.5 var(--sans);
     -webkit-font-smoothing:antialiased;overflow:hidden}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums}

/* The console is sized to the window and never scrolls: the panels that hold charts take
   whatever height is left over, so it fits a laptop screen and a large monitor alike. */
.console{height:100dvh;display:grid;
         grid-template-rows:auto auto minmax(0,1fr) auto auto;
         border-left:1px solid var(--rule-strong);border-right:1px solid var(--rule-strong);
         max-width:1680px;margin:0 auto}

.masthead{display:flex;align-items:baseline;gap:18px;flex-wrap:wrap;
          padding:11px 20px;border-bottom:1px solid var(--rule-strong)}
.mark{font-weight:600;font-size:17px;letter-spacing:-.01em;display:inline-flex;align-items:center;gap:9px}
.logo{width:23px;height:23px;flex:none}
.subject{color:var(--ink-soft);font-size:14px}
.link{margin-left:auto;display:flex;align-items:center;gap:9px;color:var(--ink-soft);font-size:13px}
.lamp{width:7px;height:7px;border-radius:50%;background:var(--alert);flex:none}
.lamp.on{background:var(--ok)}
.link.fault{color:var(--watch)}

.state{display:flex;gap:18px;align-items:center;padding:13px 20px;
       border-bottom:1px solid var(--rule);border-left:4px solid var(--rule-strong)}
.state.ok{border-left-color:var(--ok)}
.state.watch{border-left-color:var(--watch)}
.state.alert{border-left-color:var(--alert)}
.state h1{margin:0;font-size:25px;font-weight:600;letter-spacing:-.02em;line-height:1.1}
.state.ok h1{color:var(--ok)} .state.watch h1{color:var(--watch)} .state.alert h1{color:var(--alert)}
.state p{margin:3px 0 0;color:var(--ink-soft);max-width:62ch;font-size:13.5px}
.index{margin-left:auto;text-align:right;flex:none;padding-left:14px}
.index .v{font-size:25px;font-weight:500;line-height:1.1}
.index .cap{font-size:11.5px;color:var(--ink-soft)}
.gauge{margin-top:6px;width:128px;height:3px;background:var(--field);position:relative}
.gauge i{position:absolute;inset:0 auto 0 0;background:var(--ok)}
.gauge b{position:absolute;top:-3px;bottom:-3px;width:1px;background:var(--rule-strong)}

.main{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.05fr);min-height:0}
.structure{display:grid;grid-template-columns:168px minmax(0,1fr);min-height:0;
           border-right:1px solid var(--rule-strong)}
.elevation{border-right:1px solid var(--rule);padding:14px 8px;display:flex;
           align-items:center;justify-content:center;min-height:0}
.elevation svg{max-height:100%;width:auto}
.levels{display:grid;grid-template-rows:1fr 1fr 1fr auto;min-height:0}
.level{display:grid;grid-template-columns:76px 96px minmax(0,1fr) 72px;gap:12px;
       align-items:center;padding:0 16px;min-height:0;border-bottom:1px solid var(--rule)}
.level:last-child{border-bottom:0}
.level .name{font-weight:500;font-size:14px}
.level .hz{font-size:19px;font-weight:500}
.level .hz small{font-size:11px;color:var(--ink-faint);font-weight:400;margin-left:3px}
.level .mg{text-align:right;color:var(--ink-soft);font-size:12.5px}
.level.flagged{background:var(--field)}
.level .swatch{display:inline-block;width:9px;height:2px;margin-right:7px;vertical-align:middle}
.level.quiet{padding-top:9px;padding-bottom:9px;color:var(--ink-faint)}
.level canvas{width:100%;height:100%;max-height:34px}

.charts{display:grid;grid-template-rows:1fr 1fr;min-height:0}
.panel{padding:9px 16px 12px;min-height:0;display:flex;flex-direction:column;
       border-bottom:1px solid var(--rule)}
.panel:last-child{border-bottom:0}
.panel h2{margin:0 0 7px;font-size:12.5px;font-weight:500;color:var(--ink-soft);flex:none}
.panel h2 span{color:var(--ink-faint);font-weight:400}
.panel canvas{flex:1 1 auto;min-height:0;width:100%}
.axis{display:flex;justify-content:space-between;color:var(--ink-faint);font-size:10.5px;
      margin-top:3px;flex:none}

.readouts{display:grid;grid-template-columns:repeat(6,1fr);gap:1px;
          background:var(--rule);border-top:1px solid var(--rule-strong)}
.readout{background:var(--panel);padding:9px 16px}
.readout .k{font-size:12px;color:var(--ink-soft)}
.readout .v{font-size:18px;font-weight:500;margin-top:1px}
.readout .v small{font-size:11px;color:var(--ink-faint);font-weight:400}
.readout .d{font-size:11.5px;color:var(--ink-faint)}
.readout .d.up{color:var(--watch)} .readout .d.down{color:var(--s2)}

.controls{border-top:1px solid var(--rule-strong);padding:11px 20px;
          display:flex;gap:9px;flex-wrap:wrap;align-items:center}
button{font:500 13px var(--sans);color:var(--ink);background:var(--panel);
       border:1px solid var(--rule-strong);padding:7px 12px;cursor:pointer}
button:hover:not(:disabled){border-color:var(--ink-soft)}
button:focus-visible{outline:2px solid var(--s2);outline-offset:1px}
button.key{background:var(--ink);color:var(--panel);border-color:var(--ink)}
button:disabled{opacity:.4;cursor:default}
.tally{margin-left:auto;display:flex;gap:14px;color:var(--ink-faint);font-size:11.5px}
.message{color:var(--s2);font-size:12.5px;flex-basis:100%;min-height:17px}

/* Below this width the panels cannot all fit; let the page scroll instead of squeezing. */
@media(max-width:1000px){
  body{overflow:auto}
  .console{height:auto;max-width:none;border:0}
  .main{grid-template-columns:1fr}
  .structure{border-right:0;border-bottom:1px solid var(--rule-strong)}
  .levels{grid-template-rows:none}
  .level{min-height:68px}
  .panel canvas{min-height:130px}
  .readouts{grid-template-columns:repeat(3,1fr)}
}
@media(max-width:620px){
  .structure{grid-template-columns:1fr}
  .elevation{border-right:0;border-bottom:1px solid var(--rule);max-height:230px}
  .readouts{grid-template-columns:repeat(2,1fr)}
}
</style></head><body>

<div class="console">
  <div class="masthead">
    <span class="mark"><svg class="logo" viewBox="0 0 64 64" fill="none" stroke="currentColor"
      stroke-width="4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
      <path d="M15 13 V51 M49 13 V51"/><path d="M15 17 H49"/>
      <path d="M15 32 H23 L26 27 L29 16 L32 46 L35 28 L37.5 33 H49"/>
      <path d="M15 47 H49"/></svg>Heartbeat</span>
    <span class="subject">Shear-frame tower, three levels</span>
    <span class="link" id="link"><span class="lamp" id="lamp"></span><span id="linktext">connecting</span></span>
  </div>

  <div class="state" id="state">
    <div>
      <h1 id="headline">Starting</h1>
      <p id="detail">Reading the sensors.</p>
    </div>
    <div class="index">
      <div class="v num" id="index">—</div>
      <div class="cap">Damage index</div>
      <div class="gauge"><i id="gauge" style="width:0"></i><b id="mark" style="left:0"></b></div>
    </div>
  </div>

  <div class="main">
    <div class="structure">
      <div class="elevation">
        <svg id="frame" viewBox="0 0 214 272" aria-label="Tower elevation">
          <g stroke="var(--rule-strong)" stroke-width="1" fill="none">
            <path d="M44 54 h-7 M44 118 h-7 M44 182 h-7 M44 248 h-7"/>
          </g>
          <g fill="var(--ink-faint)" font-size="9.5" text-anchor="end">
            <text x="33" y="57">15.75</text><text x="33" y="121">10.75</text>
            <text x="33" y="185">5.75</text><text x="33" y="251">0</text>
            <text x="33" y="268" font-size="8.5">inches</text>
          </g>
          <g stroke="var(--ink-soft)" stroke-width="3" fill="none">
            <path d="M76 52 V248"/><path d="M182 52 V248"/>
          </g>
          <rect id="m-base" x="48" y="245" width="162" height="7" fill="var(--ink-soft)"/>
          <rect id="m-floor1" x="62" y="179" width="134" height="6" fill="var(--ink-soft)"/>
          <rect id="m-floor2" x="62" y="115" width="134" height="6" fill="var(--ink-soft)"/>
          <rect id="m-roof"   x="62" y="51"  width="134" height="6" fill="var(--ink-soft)"/>
        </svg>
      </div>
      <div class="levels" id="levels"></div>
    </div>

    <div class="charts">
      <div class="panel">
        <h2>Live signal <span id="tracenote">three sensors</span></h2>
        <canvas id="trace"></canvas>
      </div>
      <div class="panel">
        <h2>Response spectrum <span id="specnote">three sensors</span></h2>
        <canvas id="spectrum"></canvas>
        <div class="axis"><span class="num">0 Hz</span><span class="num">15</span><span class="num">30</span><span class="num">45 Hz</span></div>
      </div>
    </div>
  </div>

  <div class="readouts" id="readouts"></div>

  <div class="controls">
    <button class="key" data-cmd="/api/baseline">Set reference</button>
    <button data-cmd="/api/learn?label=floor1">Teach level 1 loose</button>
    <button data-cmd="/api/learn?label=floor2">Teach level 2 loose</button>
    <button data-cmd="/api/learn?label=roof">Teach roof loose</button>
    <span class="tally" id="tally"></span>
    <span class="message" id="message"></span>
  </div>
</div>

<script>
"use strict";
const $ = id => document.getElementById(id);
const SENSOR = ["--s1", "--s2", "--s3"];          // level 1, level 2, roof
const ROWS = [
  {key:"roof",   name:"Roof",    sensor:2},
  {key:"floor2", name:"Level 2", sensor:1},
  {key:"floor1", name:"Level 1", sensor:0},
  {key:"base",   name:"Base",    sensor:null},
];
const READOUTS = [
  {k:"f1",    label:"First mode",       fmt:v => v.toFixed(2),       unit:"Hz",     pct:true},
  {k:"f2",    label:"Upper mode",       fmt:v => v.toFixed(1),       unit:"Hz",     pct:true},
  {k:"zeta",  label:"Damping",          fmt:v => (v*100).toFixed(2), unit:"%",      pct:true},
  {k:"slip",  label:"Slip sensitivity", fmt:v => v.toFixed(2),       unit:"",       pct:false},
  {k:"s1_m3", label:"Level 1 motion",   fmt:v => v.toFixed(2),       unit:"× roof", pct:false},
  {k:"s2_m3", label:"Level 2 motion",   fmt:v => v.toFixed(2),       unit:"× roof", pct:false},
];
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const trail = [[], [], []];

document.querySelectorAll("button[data-cmd]").forEach(b =>
  b.addEventListener("click", () => fetch(b.dataset.cmd, {method:"POST"}).then(poll).catch(()=>{})));

function fitCanvas(c){
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (c.width !== Math.round(w*dpr) || c.height !== Math.round(h*dpr)){
    c.width = Math.round(w*dpr); c.height = Math.round(h*dpr);
  }
  const g = c.getContext("2d");
  g.setTransform(dpr,0,0,dpr,0,0);
  g.clearRect(0,0,w,h);
  return [g, w, h];
}

/* ---- the page runs its own transform, so the spectrum moves as fast as it is drawn ---- */
const NFFT = 512;
const HANN = new Float64Array(NFFT);
for (let i = 0; i < NFFT; i++) HANN[i] = 0.5 - 0.5*Math.cos(2*Math.PI*i/(NFFT-1));

function fft(re, im){
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++){
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j){ let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1){
    const ang = -2*Math.PI/len, wr = Math.cos(ang), wi = Math.sin(ang), half = len >> 1;
    for (let i = 0; i < n; i += len){
      let cr = 1, ci = 0;
      for (let k = 0; k < half; k++){
        const ur = re[i+k], ui = im[i+k];
        const vr = re[i+k+half]*cr - im[i+k+half]*ci;
        const vi = re[i+k+half]*ci + im[i+k+half]*cr;
        re[i+k] = ur + vr; im[i+k] = ui + vi;
        re[i+k+half] = ur - vr; im[i+k+half] = ui - vi;
        const nr = cr*wr - ci*wi; ci = cr*wi + ci*wr; cr = nr;
      }
    }
  }
}

const SPEC = {smooth:[null,null,null], re:new Float64Array(NFFT), im:new Float64Array(NFFT)};

function spectrumOf(channel){
  const n = channel.length;
  if (n < NFFT) return null;
  const re = SPEC.re, im = SPEC.im;
  for (let i = 0; i < NFFT; i++){ re[i] = channel[n - NFFT + i]*HANN[i]; im[i] = 0; }
  fft(re, im);
  const bins = NFFT >> 1, out = new Float64Array(bins);
  for (let k = 0; k < bins; k++) out[k] = Math.hypot(re[k], im[k]);
  return out;
}

/* ---- live trace and spectrum, both drawn on the animation frame ---- */
const WAVE = {first:0, step:1, rate:200, ch:null, play:null, scale:0.4, stamp:0};

function pullWave(){
  fetch("/wave", {cache:"no-store"})
    .then(r => r.ok ? r.json() : null)
    .then(w => {
      if (!w || !w.levels || !w.levels[0].length) return;
      WAVE.first = w.first; WAVE.step = w.step; WAVE.rate = w.rate; WAVE.ch = w.levels;
      const end = w.first + w.levels[0].length * w.step;
      const lag = 0.35 * WAVE.rate * w.step;
      if (WAVE.play === null || WAVE.play > end || WAVE.play < end - 6*lag) WAVE.play = end - lag;
    })
    .catch(() => {});
}

function drawTrace(w, h, g){
  const windowSamples = 2.4 * WAVE.rate * WAVE.step;
  const startAbs = WAVE.play - windowSamples;
  const count = WAVE.ch[0].length;
  const kFrom = Math.max(0, Math.floor((startAbs - WAVE.first)/WAVE.step));
  const kTo = Math.min(count - 1, Math.ceil((WAVE.play - WAVE.first)/WAVE.step));
  const xOf = k => ((WAVE.first + k*WAVE.step) - startAbs)/windowSamples * w;

  let peak = 0;
  for (let k = kFrom; k <= kTo; k++)
    for (let c = 0; c < 3; c++){ const v = Math.abs(WAVE.ch[c][k]); if (v > peak) peak = v; }
  WAVE.scale += (Math.max(peak*1.15, 0.05) - WAVE.scale) * (peak > WAVE.scale ? 0.5 : 0.04);

  const lane = h/3;
  for (let c = 0; c < 3; c++){
    const mid = lane*(2 - c) + lane/2;              // roof on top, level 1 at the bottom
    g.strokeStyle = css("--rule"); g.lineWidth = 1;
    g.beginPath(); g.moveTo(0, mid); g.lineTo(w, mid); g.stroke();
    g.beginPath();
    for (let k = kFrom; k <= kTo; k++){
      const y = mid - (WAVE.ch[c][k]/WAVE.scale)*(lane/2 - 3);
      k === kFrom ? g.moveTo(xOf(k), y) : g.lineTo(xOf(k), y);
    }
    g.strokeStyle = css(SENSOR[c]); g.lineWidth = 1.2; g.lineJoin = "round"; g.stroke();
    g.fillStyle = css("--ink-faint"); g.font = "10px " + css("--mono");
    g.fillText(["Level 1","Level 2","Roof"][c], 3, mid - lane/2 + 10);
  }
  g.fillStyle = css("--ink-faint"); g.font = "10px " + css("--mono");
  g.textAlign = "right"; g.fillText("±" + WAVE.scale.toFixed(2) + " g", w - 3, 10);
  g.textAlign = "left";
}

function drawSpectrum(w, h, g){
  const binHz = WAVE.rate*WAVE.step / NFFT;
  const top = 45, lastBin = Math.min(NFFT >> 1, Math.ceil(top/binHz));
  let hi = 0;
  for (let c = 0; c < 3; c++){
    const mag = spectrumOf(WAVE.ch[c]);
    if (!mag) return;
    if (!SPEC.smooth[c] || SPEC.smooth[c].length !== mag.length) SPEC.smooth[c] = mag;
    else { const s = SPEC.smooth[c]; for (let k = 0; k < mag.length; k++) s[k] += (mag[k] - s[k])*0.10; }
    const s = SPEC.smooth[c];
    for (let k = 1; k < lastBin; k++) if (s[k] > hi) hi = s[k];
  }
  if (hi <= 0) return;
  const lo = hi/2000;
  const Y = v => h - 3 - (Math.log10(Math.max(v, lo)/lo)/Math.log10(hi/lo))*(h - 8);
  const X = k => (k*binHz/top)*w;

  g.strokeStyle = css("--rule"); g.lineWidth = 1;
  [15, 30].forEach(f => { const x = (f/top)*w; g.beginPath(); g.moveTo(x,0); g.lineTo(x,h); g.stroke(); });

  for (let c = 0; c < 3; c++){
    const s = SPEC.smooth[c];
    g.beginPath();
    for (let k = 1; k < lastBin; k++){ const x = X(k), y = Y(s[k]); k === 1 ? g.moveTo(x,y) : g.lineTo(x,y); }
    g.strokeStyle = css(SENSOR[c]); g.lineWidth = 1.3; g.lineJoin = "round"; g.stroke();
  }
}

function frame(now){
  requestAnimationFrame(frame);
  const [gt, wt, ht] = fitCanvas($("trace"));
  const [gs, ws, hs] = fitCanvas($("spectrum"));
  if (!WAVE.ch){
    gt.fillStyle = css("--ink-faint"); gt.font = "13px " + css("--sans");
    gt.fillText("Waiting for data", 4, ht/2); return;
  }
  const dt = WAVE.stamp ? Math.min(0.2, (now - WAVE.stamp)/1000) : 0;
  WAVE.stamp = now;
  const end = WAVE.first + WAVE.ch[0].length*WAVE.step;
  WAVE.play = Math.min(end, (WAVE.play || end) + dt*WAVE.rate*WAVE.step);
  drawTrace(wt, ht, gt);
  drawSpectrum(ws, hs, gs);
}

function drawTrail(canvas, series, colour){
  const [g, w, h] = fitCanvas(canvas);
  const pts = series.filter(v => isFinite(v));
  if (pts.length < 2 || h < 4) return;
  let lo = Math.min(...pts), hi = Math.max(...pts);
  if (hi - lo < 0.25){ const c = (hi+lo)/2; lo = c - 0.125; hi = c + 0.125; }
  g.beginPath();
  pts.forEach((v,i) => {
    const x = i/(pts.length-1)*w, y = h - 3 - (v-lo)/(hi-lo)*(h-6);
    i ? g.lineTo(x,y) : g.moveTo(x,y);
  });
  g.strokeStyle = colour; g.lineWidth = 1.3; g.lineJoin = "round"; g.stroke();
}

function buildLevels(){
  $("levels").innerHTML = ROWS.map(r => `
    <div class="level${r.sensor === null ? " quiet" : ""}" id="row-${r.key}">
      <span class="name"><i class="swatch" style="background:${
        r.sensor === null ? "transparent" : css(SENSOR[r.sensor])}"></i>${r.name}</span>
      <span class="hz num" id="hz-${r.key}">${r.sensor === null ? "reference" : "—"}</span>
      <canvas id="trail-${r.key}"></canvas>
      <span class="mg num" id="mg-${r.key}"></span>
    </div>`).join("");
}

function render(s){
  const got = s.device_rate, want = s.expected_rate;
  const losing = got ? (Math.abs(got-want)/want > 0.03 || s.lost > 2) : false;
  $("lamp").className = "lamp" + (s.arrival_rate > 50 ? " on" : "");
  $("link").className = "link" + (losing ? " fault" : "");
  $("linktext").innerHTML = `${s.status || s.source} &nbsp; <span class="num">${
    got ? got.toFixed(1) : s.arrival_rate}</span> Hz${losing ? " · losing data" : ""}`;

  const v = s.verdict || {}, stale = (v.age || 0) > 6;
  const alarm = v.alarm || 0, trip = s.thresholds.trip, warn = s.thresholds.warn;
  let tone = "";
  if (v.state === "healthy") tone = "ok";
  else if (alarm >= trip) tone = "alert";
  else if (alarm >= warn) tone = "watch";
  $("state").className = "state " + tone;
  $("headline").textContent = v.headline || "";
  $("headline").style.opacity = stale ? ".45" : "1";
  $("detail").textContent = stale
    ? `Last tap ${v.age.toFixed(0)} s ago. Tap again to refresh.` : (v.detail || "");
  $("index").textContent = s.fingerprint ? alarm.toFixed(1) : "—";
  const full = trip*1.6;
  $("gauge").style.width = Math.min(100, alarm/full*100).toFixed(0) + "%";
  $("gauge").style.background = css(tone === "alert" ? "--alert" : tone === "watch" ? "--watch" : "--ok");
  $("mark").style.left = (trip/full*100).toFixed(0) + "%";

  ["floor1","floor2","roof"].forEach(k => {
    const el = $("m-"+k), hit = k === v.floor;
    el.setAttribute("fill", hit ? css(tone === "alert" ? "--alert" : "--watch") : css("--ink-soft"));
    el.setAttribute("height", hit ? 10 : 6);
    el.setAttribute("y", {floor1:hit?177:179, floor2:hit?113:115, roof:hit?49:51}[k]);
    const row = $("row-"+k);
    if (row) row.className = "level" + (hit ? " flagged" : "");
  });

  const sp = s.spectra;
  ROWS.forEach(r => {
    if (r.sensor === null) return;
    const sensor = sp && sp.sensors ? sp.sensors[r.sensor] : null;
    const hz = $("hz-"+r.key), mg = $("mg-"+r.key);
    if (sensor && sensor.peak){
      hz.innerHTML = sensor.peak.toFixed(2) + '<small>Hz</small>';
      mg.textContent = (sensor.rms*1000).toFixed(1) + " mg";
      const t = trail[r.sensor];
      t.push(sensor.peak); if (t.length > 90) t.shift();
      drawTrail($("trail-"+r.key), t, css(SENSOR[r.sensor]));
    } else { hz.textContent = "—"; mg.textContent = ""; }
  });

  const fp = s.fingerprint, base = s.baseline, d = v.delta || {};
  $("readouts").innerHTML = READOUTS.map(r => {
    if (!fp || fp[r.k] == null)
      return `<div class="readout"><div class="k">${r.label}</div><div class="v num">—</div>`
           + `<div class="d">${r.k === "slip" ? "vary your tap strength" : ""}</div></div>`;
    let note = base ? "" : "no reference";
    if (base && d[r.k] != null){
      const n = r.pct ? (d[r.k]*100).toFixed(1) + "%" : d[r.k].toFixed(2);
      note = (d[r.k] >= 0 ? "+" : "") + n + " vs reference";
    }
    const dir = d[r.k] == null ? "" : (d[r.k] > 0 ? " up" : " down");
    return `<div class="readout"><div class="k">${r.label}</div>`
         + `<div class="v num">${r.fmt(fp[r.k])}<small> ${r.unit}</small></div>`
         + `<div class="d${dir} num">${note}</div></div>`;
  }).join("");

  const ex = s.examples || {};
  $("tally").innerHTML = ["floor1","floor2","roof"].map(k =>
      `<span>${s.labels[k]} <span class="num">${ex[k]||0}</span></span>`).join("")
    + (base ? `<span>reference <span class="num">${base.f1.toFixed(2)}</span> Hz</span>`
            : `<span style="color:var(--watch)">no reference</span>`);
  $("specnote").textContent = `three sensors · 2.6 s window · 60 fps`;
  $("tracenote").textContent = `three sensors · ${WAVE.rate||200} samples per second`;

  const col = s.collecting;
  if (col){
    const what = col.kind === "baseline" ? "reference" : s.labels[col.label];
    $("message").textContent = `Recording ${what}. ${col.taps} taps so far, `
      + `${col.left.toFixed(0)} s left. Keep tapping, and vary how hard.`;
  } else if (s.notice) $("message").textContent = s.notice;
  document.querySelectorAll("button[data-cmd]").forEach(b => b.disabled = !!s.busy);
}

let misses = 0;
function poll(){
  fetch("/state", {cache:"no-store"})
    .then(r => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); })
    .then(s => { misses = 0; if (!s.starting) render(s); })
    .catch(() => { if (++misses >= 4){ $("linktext").textContent = "reconnecting"; $("lamp").className = "lamp"; } });
}
buildLevels();
poll(); setInterval(poll, 300);
pullWave(); setInterval(pullWave, 100);
requestAnimationFrame(frame);
window.addEventListener("resize", () => trail.forEach(t => t.length = 0));
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    monitor: Monitor = None
    protocol_version = "HTTP/1.1"        # keep-alive: one connection instead of two a second
    server_version = "Heartbeat/1.0"

    def log_message(self, *args):        # keep the console clean during a demo
        pass

    def send_payload(self, code, body, ctype):
        data = body.encode("utf-8")
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass                         # the browser navigated away mid-reply; harmless

    def do_GET(self):
        try:
            route = urlparse(self.path).path
            if route in ("/", "/index.html"):
                self.send_payload(200, PAGE, "text/html; charset=utf-8")
            elif route == "/wave":
                self.send_payload(200, json.dumps(jsonable(self.monitor.wave), allow_nan=False),
                                  "application/json")
            elif route == "/state":
                self.send_payload(200, json.dumps(self.monitor.state, allow_nan=False),
                                  "application/json")
            else:
                self.send_payload(404, "not found", "text/plain; charset=utf-8")
        except Exception:
            self.send_payload(500, json.dumps({"error": "internal"}), "application/json")

    def do_POST(self):
        try:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            if parsed.path == "/api/baseline":
                self.monitor.request("baseline")
            elif parsed.path == "/api/learn":
                label = (query.get("label") or [""])[0]
                if label not in TRAINABLE:
                    self.send_payload(400, '{"error":"bad label"}', "application/json")
                    return
                self.monitor.request("learn", label)
            else:
                self.send_payload(404, '{"error":"not found"}', "application/json")
                return
            self.send_payload(200, '{"ok":true}', "application/json")
        except Exception:
            self.send_payload(500, '{"error":"internal"}', "application/json")


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ======================================================================================
# 7. Command line
# ======================================================================================

def list_ports():
    try:
        from serial.tools import list_ports as lp
    except ImportError:
        print("pyserial is not installed. Run: pip install pyserial")
        return
    found = list(lp.comports())
    if not found:
        print("No serial ports found. Is the Feather plugged in?")
    for p in found:
        print(f"  {p.device:10s} {p.description}")


def open_window(url):
    """Open the console in its own window, with no tabs or address bar.

    Chrome and Edge both take --app for this. If neither is where it is expected, this
    falls back to an ordinary browser tab, which works just as well.
    """
    import shutil
    import subprocess
    candidates = ["chrome", "google-chrome", "msedge", "chromium",
                  r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                  r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"]
    for name in candidates:
        path = shutil.which(name) or (name if os.path.exists(name) else None)
        if not path:
            continue
        try:
            subprocess.Popen([path, f"--app={url}", "--window-size=1280,980"])
            return
        except OSError:
            continue
    webbrowser.open(url)


def main():
    ap = argparse.ArgumentParser(description="Building heartbeat monitor")
    ap.add_argument("--port", default="COM17", help="serial port of the Feather (default COM17)")
    ap.add_argument("--replay", metavar="CSV", help="replay a recording instead of the hardware")
    ap.add_argument("--http", type=int, default=8000, help="dashboard port (default 8000)")
    ap.add_argument("--train-seconds", type=float, default=TRAIN_SECONDS,
                    help=f"tapping gathered per baseline or example (default {TRAIN_SECONDS:.0f})")
    ap.add_argument("--no-save", action="store_true", help="do not save training captures")
    ap.add_argument("--list", action="store_true", help="list serial ports and exit")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser")
    ap.add_argument("--app", action="store_true",
                    help="open in its own window with no browser toolbar, if Chrome or Edge is installed")
    args = ap.parse_args()

    if args.list:
        list_ports()
        return

    source = ReplaySource(args.replay) if args.replay else SerialSource(args.port)
    library = Library(LIBRARY_PATH)
    monitor = Monitor(source, library,
                      train_seconds=max(10.0, args.train_seconds),
                      save_captures=not args.no_save)

    threading.Thread(target=monitor.read_loop, daemon=True, name="reader").start()
    threading.Thread(target=monitor.analyse_loop, daemon=True, name="analyser").start()

    Handler.monitor = monitor
    server = Server(("127.0.0.1", args.http), Handler)
    url = f"http://localhost:{args.http}"
    print("Heartbeat  structural health monitor")
    print(f"  source   : {source.status}")
    print(f"  baseline : {'stored' if library.baseline else 'none yet - press Set reference'}")
    print(f"  dashboard: {url}")
    print("  Ctrl+C to stop.")
    if not args.no_browser:
        threading.Timer(1.0, lambda: open_window(url) if args.app else webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        monitor.stop()
        server.server_close()


if __name__ == "__main__":
    main()
