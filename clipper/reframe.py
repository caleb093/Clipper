"""Decide, frame by frame, how to turn a landscape clip into a vertical one.

Narrated explainer videos are mostly stills and B-roll, not talking heads, so
instead of tracking a person we look at where the visual interest is in each shot:

  * crop - the interest sits in one area (a subject, a face, an object): fill the
           vertical frame with a 9:16 window centred on it.
  * fit  - the interest is spread across the frame (text, charts, maps, wide
           scenery): show the whole frame over a blurred copy of itself so
           nothing important gets cut off.

Pipeline: decode the clip small -> find hard cuts -> score interest on sampled
frames -> crop/fit votes smoothed over time within each shot (so a dissolve from
a photo into a chart can switch layout) -> a smoothed crop path where things move.

Fit is the safe failure (nothing gets cut off), so cropping needs clear evidence.

Podcasts (content_type="podcast") follow people instead: see speakers.py. Shots
without people in them still use the interest logic above.
"""

from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from clipper import faces
from clipper.media import even, open_frame_reader, read_frames
from clipper.speakers import FaceWatcher, Rect, loudness, people_in, plan_people, speech_mask

ANALYSIS_WIDTH = 320
PODCAST_ANALYSIS_WIDTH = 640  # faces in a wide two-shot are small; mouths need the detail
SAMPLES_PER_SECOND = 6
DEFAULT_CAPTION_Y = 1420  # on a 1080x1920 frame

# Cut detection (same idea and defaults as PySceneDetect's AdaptiveDetector).
CUT_MIN_SCORE = 15.0
CUT_RATIO = 3.0
CUT_WINDOW = 2
MIN_SHOT_SECONDS = 0.5

# Crop when the best 9:16 window holds this much more of the interest than an
# evenly spread frame would give it (for 16:9 input: ~57% of the interest vs 32%).
CROP_MARGIN = 0.25
MIN_LAYOUT_SECONDS = 1.5  # shortest stretch of crop or fit inside a shot
FACE_WEIGHT = 1.5  # a face counts 1.5x as much as everything else in the frame
STATIC_SPREAD = 0.08  # centre moves less than this (fraction of width) -> fixed crop
JUMP_DISTANCE = 0.2  # centre jumps more than this between samples -> cut, don't pan
SMOOTH_SECONDS = 1.0
MAX_PAN_SPEED = 0.35  # fraction of frame width per second


@dataclass
class Segment:
    """A stretch of frames shown with one layout."""

    first_frame: int
    last_frame: int  # inclusive
    layout: str  # crop | fit | speaker | split
    concentration: float = 0.0
    caption_y: int = DEFAULT_CAPTION_Y


@dataclass
class FramePlan:
    fps: Fraction
    window_frac: float  # crop width as a fraction of source width
    layouts: list[str]  # per frame: crop | fit | speaker | split
    centers: np.ndarray  # per frame, for "crop": crop centre as a fraction of source width
    segments: list[Segment]
    rects: list[tuple[Rect, ...] | None] = field(default_factory=list)  # per frame, for speaker / split

    def at(self, i: int) -> tuple[str, float, tuple[Rect, ...] | None]:
        i = min(i, len(self.layouts) - 1)
        return self.layouts[i], float(self.centers[i]), self.rects[i] if self.rects else None

    def caption_y_at(self, seconds: float) -> int:
        frame = int(seconds * float(self.fps))
        for seg in self.segments:
            if seg.first_frame <= frame <= seg.last_frame:
                return seg.caption_y
        return self.segments[-1].caption_y if self.segments else DEFAULT_CAPTION_Y


def plan_clip(
    video: Path,
    start: float,
    duration: float,
    fps: Fraction,
    src_w: int,
    src_h: int,
    layout: str,
    out_aspect: float,
    content_type: str = "explainer",
    speech: list[tuple[float, float]] | None = None,
) -> FramePlan:
    """speech: (start, end) of each spoken word, in seconds from the clip start (podcasts)."""
    window_frac = min(1.0, src_h * out_aspect / src_w)
    if window_frac >= 0.98:
        layout = "fit"  # source is already (nearly) vertical
    podcast = content_type == "podcast" and layout == "auto"

    aw = PODCAST_ANALYSIS_WIDTH if podcast else ANALYSIS_WIDTH
    ah = even(aw * src_h / src_w)
    interest_size = (ANALYSIS_WIDTH, even(ANALYSIS_WIDTH * src_h / src_w))
    sample_every = max(1, round(float(fps) / SAMPLES_PER_SECOND))
    watcher = FaceWatcher(float(fps)) if podcast else None

    diffs: list[float] = []
    samples: list[tuple[int, float, float]] = []  # (frame, centre, concentration)
    prev = None
    n = 0
    reader = open_frame_reader(video, start, duration, fps, aw, ah)
    for n, frame in enumerate(read_frames(reader, aw, ah)):
        small = cv2.cvtColor(cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
        small = small.astype(np.float32)
        diffs.append(0.0 if prev is None else float(np.mean(np.abs(small - prev))))
        prev = small
        if n % sample_every == 0 and layout != "fit":
            interest = frame if aw == ANALYSIS_WIDTH else cv2.resize(frame, interest_size, interpolation=cv2.INTER_AREA)
            samples.append((n, *_best_window(interest, window_frac)))
        if watcher is not None:
            watcher.add(n, frame)
    total = len(diffs)
    if total == 0:
        raise RuntimeError(f"Could not decode frames from {video} at {start:.1f}s")

    layouts = ["fit"] * total
    centers = np.full(total, 0.5, dtype=np.float64)
    rects: list[tuple[Rect, ...] | None] = [None] * total
    segments: list[Segment] = []
    threshold = window_frac + CROP_MARGIN
    min_run = max(1, round(MIN_LAYOUT_SECONDS * SAMPLES_PER_SECOND))
    talking = speech_mask(speech or [], total, float(fps)) if podcast else None
    loud = loudness(video, start, duration, fps, total) if podcast else None

    for first, last in _split_shots(diffs, float(fps)):
        people = people_in(watcher.observations, first, last, float(fps)) if watcher is not None else []
        if people:
            plans = plan_people(people, first, last, float(fps), talking, loud, window_frac, src_w / src_h, out_aspect)
            for ps in plans:
                segments.append(Segment(ps.first, ps.last, ps.layout, caption_y=ps.caption_y))
                frames = slice(ps.first, ps.last + 1)
                count = ps.last - ps.first + 1
                layouts[frames] = [ps.layout] * count
                if ps.layout == "crop":
                    centers[frames] = np.clip(ps.centre, window_frac / 2, 1 - window_frac / 2)
                rects[frames] = [ps.rects or None] * count
            continue

        idx = [k for k, s in enumerate(samples) if first <= s[0] <= last]
        if not idx and samples:  # very short shot: borrow the nearest sample
            mid = (first + last) / 2
            idx = [min(range(len(samples)), key=lambda k: abs(samples[k][0] - mid))]
        if not idx:  # layout forced to fit: nothing was sampled
            segments.append(Segment(first, last, "fit", 0.0))
            continue

        if layout == "auto":
            votes = [samples[k][2] >= threshold for k in idx]
        else:
            votes = [layout == "crop"] * len(idx)

        for a, b, is_crop in _layout_runs(votes, min_run):
            # Switch layouts halfway between the last sample of one run and the first of the next.
            seg_first = first if a == 0 else (samples[idx[a - 1]][0] + samples[idx[a]][0] + 1) // 2
            seg_last = last if b == len(idx) - 1 else (samples[idx[b]][0] + samples[idx[b + 1]][0] + 1) // 2 - 1
            seg_first, seg_last = max(first, seg_first), min(last, seg_last)
            run = idx[a : b + 1]
            seg = Segment(seg_first, seg_last, "crop" if is_crop else "fit", float(np.median([samples[k][2] for k in run])))
            segments.append(seg)

            frames = slice(seg_first, seg_last + 1)
            layouts[frames] = [seg.layout] * (seg_last - seg_first + 1)
            if is_crop:
                centers[frames] = _crop_path(
                    [samples[k][0] for k in run],
                    [samples[k][1] for k in run],
                    seg_first,
                    seg_last,
                    float(fps),
                    window_frac,
                )

    segments.sort(key=lambda s: s.first_frame)
    return FramePlan(fps=fps, window_frac=window_frac, layouts=layouts, centers=centers, segments=segments, rects=rects)


def _layout_runs(votes: list[bool], min_run: int) -> list[tuple[int, int, bool]]:
    """Smooth per-sample crop votes into runs (start, end inclusive, is_crop) of at least min_run samples."""
    half = min_run // 2
    padded = np.pad(np.asarray(votes, dtype=np.float64), half, mode="edge")
    smoothed = np.convolve(padded, np.ones(2 * half + 1) / (2 * half + 1), mode="valid") >= 0.5

    runs: list[list] = []
    for i, value in enumerate(smoothed):
        if runs and runs[-1][2] == value:
            runs[-1][1] = i
        else:
            runs.append([i, i, bool(value)])

    while len(runs) > 1:
        k = min(range(len(runs)), key=lambda j: runs[j][1] - runs[j][0])
        if runs[k][1] - runs[k][0] + 1 >= min_run:
            break
        # Fold the shortest run into a neighbour, then join neighbours that now match.
        if k == 0:
            runs[1][0] = runs[0][0]
        else:
            runs[k - 1][1] = runs[k][1]
        del runs[k]
        joined = [runs[0]]
        for run in runs[1:]:
            if run[2] == joined[-1][2]:
                joined[-1][1] = run[1]
            else:
                joined.append(run)
        runs = joined
    return [(a, b, v) for a, b, v in runs]


# ------------------------------------------------------------------ shots


def _split_shots(diffs: list[float], fps: float) -> list[tuple[int, int]]:
    """Frame ranges (first, last inclusive) between hard cuts."""
    d = np.asarray(diffs)
    min_len = max(1, int(MIN_SHOT_SECONDS * fps))
    cuts = [0]
    for i in range(1, len(d)):
        neighbours = np.concatenate([d[max(1, i - CUT_WINDOW) : i], d[i + 1 : i + 1 + CUT_WINDOW]])
        avg = float(neighbours.mean()) if neighbours.size else 0.0
        if d[i] >= CUT_MIN_SCORE and d[i] / (avg + 1e-3) >= CUT_RATIO and i - cuts[-1] >= min_len:
            cuts.append(i)
    # Absorb a too-short final shot into the one before it.
    if len(cuts) > 1 and len(d) - cuts[-1] < min_len:
        cuts.pop()
    bounds = cuts + [len(d)]
    return [(bounds[k], bounds[k + 1] - 1) for k in range(len(cuts))]


# ---------------------------------------------------------- interest map


def _best_window(frame: np.ndarray, window_frac: float) -> tuple[float, float]:
    """Return (centre, concentration) of the 9:16 window holding the most interest."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    w = gray.shape[1]

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = _unit(np.sqrt(gx * gx + gy * gy).sum(axis=0))
    sal = _unit(np.interp(np.linspace(0, 1, w), np.linspace(0, 1, 64), _spectral_residual(gray).sum(axis=0)))
    profile = 0.5 * edges + 0.5 * sal

    xs = np.arange(w)
    frame_area = gray.shape[0] * w
    for x, y, fw, fh in faces.detect(frame)[:, :4]:
        bump = np.exp(-0.5 * ((xs - (x + fw / 2)) / max(fw / 2, 1)) ** 2)
        size = min(1.0, 20 * fw * fh / frame_area)  # tiny background faces count less
        profile += FACE_WEIGHT * size * _unit(bump)
    profile = _unit(profile)

    k = max(1, int(round(window_frac * w)))
    sums = np.convolve(profile, np.ones(k), mode="valid")  # sum of each window position
    concentration = float(sums.max())

    # Mild centre bias breaks near-ties toward the middle of the frame.
    pos = (np.arange(sums.size) + k / 2) / w
    biased = sums * np.exp(-0.5 * ((pos - 0.5) / 0.6) ** 2)
    centre = float(pos[int(np.argmax(biased))])
    return centre, concentration


def _spectral_residual(gray: np.ndarray) -> np.ndarray:
    """Hou & Zhang (2007) saliency: what stands out from the image's usual texture."""
    h = max(8, int(64 * gray.shape[0] / gray.shape[1]))
    img = cv2.resize(gray, (64, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    spectrum = np.fft.fft2(img)
    log_amp = np.log(np.abs(spectrum) + 1e-6).astype(np.float32)
    residual = log_amp - cv2.blur(log_amp, (3, 3))
    sal = np.abs(np.fft.ifft2(np.exp(residual + 1j * np.angle(spectrum)))) ** 2
    return cv2.GaussianBlur(sal.astype(np.float32), (0, 0), 2.5)


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    v = v - v.min()
    total = v.sum()
    return v / total if total > 0 else np.full_like(v, 1.0 / v.size)


# -------------------------------------------------------------- crop path


def _crop_path(
    sample_frames: list[int],
    sample_centres: list[float],
    first: int,
    last: int,
    fps: float,
    window_frac: float,
) -> np.ndarray:
    """Per-frame crop centre: cut when the subject jumps, glide when it moves."""
    # Split where the target jumps (e.g. a dissolve to a new subject): an editor
    # would cut there rather than pan across the frame.
    pieces = [[0]]
    for k in range(1, len(sample_centres)):
        if abs(sample_centres[k] - sample_centres[k - 1]) > JUMP_DISTANCE:
            pieces.append([k])
        else:
            pieces[-1].append(k)
    # Don't cut back and forth: fold pieces shorter than a second into the previous one.
    min_samples = max(1, SAMPLES_PER_SECOND)
    merged = [pieces[0]]
    for piece in pieces[1:]:
        if len(piece) < min_samples:
            merged[-1].extend(piece)
        else:
            merged.append(piece)

    path = np.empty(last - first + 1)
    for p, ks in enumerate(merged):
        f0 = first if p == 0 else (sample_frames[ks[0] - 1] + sample_frames[ks[0]] + 1) // 2
        f1 = last if p == len(merged) - 1 else (sample_frames[ks[-1]] + sample_frames[ks[-1] + 1] + 1) // 2 - 1
        path[f0 - first : f1 - first + 1] = _glide(
            [sample_frames[k] for k in ks], [sample_centres[k] for k in ks], f0, f1, fps
        )
    return np.clip(path, window_frac / 2, 1 - window_frac / 2)


def _glide(sample_frames: list[int], sample_centres: list[float], first: int, last: int, fps: float) -> np.ndarray:
    frames = np.arange(first, last + 1)
    if max(sample_centres) - min(sample_centres) < STATIC_SPREAD:
        return np.full(frames.size, float(np.median(sample_centres)))

    # Moving subject: follow it smoothly, like a camera operator.
    path = np.interp(frames, sample_frames, sample_centres)
    k = max(1, int(SMOOTH_SECONDS * fps))
    padded = np.pad(path, (k // 2, k - 1 - k // 2), mode="edge")
    path = np.convolve(padded, np.ones(k) / k, mode="valid")

    max_step = MAX_PAN_SPEED / fps
    for i in range(1, path.size):
        path[i] = path[i - 1] + np.clip(path[i] - path[i - 1], -max_step, max_step)
    return path
