"""Podcast framing: who is on screen in each shot, and who is talking.

The person talking is the one whose mouth moves *in time with the sound*: over
a second or two, their mouth movement rises and falls with the audio's loudness.
Raw movement alone isn't enough (people chew, laugh, nod and gesture while
listening), but chewing doesn't follow someone else's voice. Mouth movement is
measured as the change around the mouth minus the change around the eyes and
forehead, so moving the whole head doesn't count. The transcript says when
anyone is speaking at all. Per shot:

  * one person                       -> frame them
  * people sitting close together    -> one vertical crop around the group
  * two or more people, one talking  -> frame the speaker; cut when the speaker changes
  * quick back-and-forth, 2 people   -> split screen, left person on top
  * quick back-and-forth, 3+ people  -> the whole frame over a blurred background
"""

import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np

from clipper import faces

DETECTIONS_PER_SECOND = 10
MIN_PRESENCE = 0.3  # on screen for at least this share of the shot to count as a person
SYNC_WINDOW_SECONDS = 1.5  # how much mouth movement and sound are compared at a time
SYNC_MAX_LAG = 3  # frames of slack between seeing the mouth move and hearing it
SYNC_MIN = 0.2  # correlation needed to count as talking
SYNC_MARGIN = 0.12  # the speaker must follow the sound this much better than anyone else
LABEL_SMOOTH_SECONDS = 0.6
EXCHANGE_WINDOW_SECONDS = 3.0  # two changes of speaker within this -> back-and-forth
MIN_SEGMENT_SECONDS = 1.6  # never cut between layouts faster than this
SPEECH_PAD_SECONDS = 0.2
SPEAKER_FRAME_FACES = 5.0  # speaker shots are this many face-heights tall...
MAX_ZOOM = 1.4  # ...but never zoom in more than this, to keep the picture sharp
PANEL_FACES = 3.2  # split-screen panels: head and shoulders

BOTH = -1
UNKNOWN = -2

# A region of the source frame: centre x, centre y (fractions of width / height)
# and height (fraction of frame height); its width follows from the target's shape.
Rect = tuple[float, float, float]


@dataclass
class Observation:
    frame: int
    cx: float  # face box, as fractions of the frame
    cy: float
    w: float
    h: float
    talk: float  # mouth movement since the previous frame


@dataclass
class Person:
    cx: float
    cy: float
    w: float
    h: float
    movement: np.ndarray  # mouth movement for every frame of the shot (0 where the face wasn't seen)


@dataclass
class PeopleSegment:
    first: int  # frames, inclusive
    last: int
    layout: str  # speaker | split | crop | fit
    rects: tuple[Rect, ...] = ()
    centre: float = 0.5  # for "crop"
    caption_y: int = 1420  # where captions go on a 1080x1920 frame


class FaceWatcher:
    """Fed every frame of a clip; records each face and how much its mouth moves."""

    def __init__(self, fps: float):
        self.every = max(1, round(fps / DETECTIONS_PER_SECOND))
        self.observations: list[Observation] = []
        self._faces = faces.NO_FACES
        self._prev: np.ndarray | None = None

    def add(self, n: int, frame: np.ndarray) -> None:
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
        if n % self.every == 0:
            self._faces = faces.detect(frame)
        if self._prev is not None:
            height, width = gray.shape
            for f in self._faces:
                x, y, w, h = (float(v) for v in f[:4])
                self.observations.append(
                    Observation(n, (x + w / 2) / width, (y + h / 2) / height, w / width, h / height,
                                _mouth_movement(gray, self._prev, f))
                )
        self._prev = gray


def _mouth_movement(gray: np.ndarray, prev: np.ndarray, f: np.ndarray) -> float:
    # Regions come from the face box rather than the landmark points, which
    # drift on faces turned to the side (common in wide podcast shots).
    x, y, w, h = f[:4]
    mouth = _change(gray, prev, x + 0.2 * w, x + 0.8 * w, y + 0.62 * h, y + 1.02 * h)  # mouth and jaw
    upper = _change(gray, prev, x + 0.15 * w, x + 0.85 * w, y + 0.1 * h, y + 0.45 * h)  # eyes and brow
    if mouth is None or upper is None:
        return 0.0
    return max(0.0, mouth - upper)


def _change(a: np.ndarray, b: np.ndarray, x0: float, x1: float, y0: float, y1: float) -> float | None:
    height, width = a.shape
    x0, x1 = max(0, int(round(x0))), min(width, int(round(x1)))
    y0, y1 = max(0, int(round(y0))), min(height, int(round(y1)))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return float(cv2.absdiff(a[y0:y1, x0:x1], b[y0:y1, x0:x1]).mean())


# ------------------------------------------------------------------ people


def people_in(observations: list[Observation], first: int, last: int, fps: float) -> list[Person]:
    """The people visible through most of a shot, left to right."""
    obs = sorted((o for o in observations if first + 2 <= o.frame <= last), key=lambda o: o.cx)  # skip the cut
    if not obs:
        return []
    groups = [[obs[0]]]
    for o in obs[1:]:
        if o.cx - groups[-1][-1].cx > max(0.04, 0.5 * o.w):
            groups.append([o])
        else:
            groups[-1].append(o)

    n = last - first + 1
    people = []
    for group in groups:
        if len({o.frame for o in group}) < MIN_PRESENCE * n:
            continue
        movement = np.zeros(n)
        for o in group:
            movement[o.frame - first] = max(movement[o.frame - first], o.talk)
        median = {a: float(np.median([getattr(o, a) for o in group])) for a in ("cx", "cy", "w", "h")}
        people.append(Person(movement=movement, **median))
    return sorted(people, key=lambda p: p.cx)


def plan_people(
    people: list[Person],
    first: int,
    last: int,
    fps: float,
    speech: np.ndarray,
    loudness: np.ndarray,
    window_frac: float,
    src_aspect: float,
    out_aspect: float,
) -> list[PeopleSegment]:
    """Layouts for a shot that has people in it."""
    if len(people) == 1:
        return [_speaker_segment(people[0], first, last, src_aspect, out_aspect)]

    width = people[-1].cx - people[0].cx + max(p.w for p in people)
    if width <= 0.8 * window_frac:  # everyone fits in one vertical crop
        centre = (people[0].cx + people[-1].cx) / 2
        return [PeopleSegment(first, last, "crop", centre=centre)]

    labels = speaker_labels(people, speech[first : last + 1], loudness[first : last + 1], fps)
    segments = []
    for a, b, label in runs(labels, max(1, int(MIN_SEGMENT_SECONDS * fps))):
        if label >= 0:
            segments.append(_speaker_segment(people[label], first + a, first + b, src_aspect, out_aspect))
        elif len(people) == 2:
            panel_aspect = 2 * out_aspect  # each panel is half the vertical frame
            rects = tuple(_rect(p, PANEL_FACES, 0.42, src_aspect, panel_aspect, min_h=0.3) for p in people)
            segments.append(PeopleSegment(first + a, first + b, "split", rects=rects, caption_y=960))
        else:
            segments.append(PeopleSegment(first + a, first + b, "fit"))
    return segments


def _speaker_segment(p: Person, first: int, last: int, src_aspect: float, out_aspect: float) -> PeopleSegment:
    rect = _rect(p, SPEAKER_FRAME_FACES, 0.38, src_aspect, out_aspect, min_h=1 / MAX_ZOOM)
    cx, cy, h = rect
    # Captions just below the chin, wherever the face lands in the frame.
    face_y = (p.cy - (cy - h / 2)) / h * 1920
    chin = face_y + 0.55 * p.h / h * 1920
    caption_y = int(min(1560, max(1000, chin + 110)))
    return PeopleSegment(first, last, "speaker", rects=(rect,), caption_y=caption_y)


def _rect(p: Person, face_heights: float, face_at: float, src_aspect: float, aspect: float, min_h: float) -> Rect:
    """A region around a face: face_heights tall, with the face face_at of the way down."""
    h = min(1.0, max(min_h, face_heights * p.h))
    if h * aspect > src_aspect:  # too wide for the frame: shrink to fit
        h = src_aspect / aspect
    top = min(max(p.cy - face_at * h, 0.0), 1.0 - h)
    return (p.cx, top + h / 2, h)


# ---------------------------------------------------------------- speaking


def speaker_labels(people: list[Person], speech: np.ndarray, loudness: np.ndarray, fps: float) -> np.ndarray:
    """For each frame: the index of the person talking, or BOTH for a back-and-forth."""
    # Mouth movement tracks the *changes* in loudness (syllables starting and stopping)
    # better than loudness itself; on test footage this halved the wrong calls.
    change = np.abs(np.convolve(np.diff(loudness, prepend=loudness[:1]), np.ones(3) / 3, mode="same"))
    sync = np.vstack([sync_with_sound(p.movement, change, fps) for p in people])
    n = sync.shape[1]
    order = np.argsort(-sync, axis=0)
    best, second = order[0], order[1]
    cols = np.arange(n)
    top, runner_up = sync[best, cols], sync[second, cols]

    labels = np.full(n, UNKNOWN)
    speaking = speech[:n]
    clear = speaking & (top >= SYNC_MIN) & (top - runner_up >= SYNC_MARGIN)
    both = speaking & (runner_up >= SYNC_MIN) & (top - runner_up < SYNC_MARGIN)
    labels[clear] = best[clear]
    labels[both] = BOTH  # two people following the sound equally well: talking over each other

    known = np.flatnonzero(labels != UNKNOWN)
    if known.size == 0:  # nobody visibly talking: show everyone
        return np.full(n, BOTH)
    labels[: known[0]] = labels[known[0]]
    for i in range(known[0] + 1, n):  # keep the last speaker through pauses
        if labels[i] == UNKNOWN:
            labels[i] = labels[i - 1]

    # Majority vote over a short window removes single-frame flicker.
    choices = [*range(len(people)), BOTH]
    k = max(1, int(LABEL_SMOOTH_SECONDS * fps))
    votes = np.vstack([np.convolve((labels == c).astype(float), np.ones(k), mode="same") for c in choices])
    labels = np.asarray(choices)[np.argmax(votes, axis=0)]

    # Two changes of speaker close together is a back-and-forth: show both people.
    # Count changes between people only; A -> (both) -> B is one change, not two.
    person = labels.copy()
    named = np.flatnonzero(person != BOTH)
    if named.size:
        person[: named[0]] = person[named[0]]
        for i in range(named[0] + 1, n):
            if person[i] == BOTH:
                person[i] = person[i - 1]
    changes = np.r_[0, (person[1:] != person[:-1]).astype(float)]
    w = max(1, int(EXCHANGE_WINDOW_SECONDS * fps))
    busy = np.convolve(changes, np.ones(w), mode="same") >= 2
    labels[busy] = BOTH
    return labels


def sync_with_sound(movement: np.ndarray, sound: np.ndarray, fps: float) -> np.ndarray:
    """For each frame: how closely mouth movement follows the sound around that moment (-1 to 1).

    Correlation over a sliding window, taking the best of a few frames of lag
    because the mouth and the sound are rarely perfectly aligned."""
    w = max(9, int(SYNC_WINDOW_SECONDS * fps)) | 1  # odd, so windows are centred
    best = np.full(movement.size, -1.0)
    for lag in range(-SYNC_MAX_LAG, SYNC_MAX_LAG + 1):
        shifted = np.roll(movement, lag)
        if lag > 0:
            shifted[:lag] = 0
        elif lag < 0:
            shifted[lag:] = 0
        best = np.maximum(best, _windowed_correlation(shifted, sound[: movement.size], w))
    return best


def _windowed_correlation(a: np.ndarray, b: np.ndarray, w: int) -> np.ndarray:
    pad = w // 2
    a, b = np.pad(a, pad, mode="edge"), np.pad(b, pad, mode="edge")

    def window_sum(v: np.ndarray) -> np.ndarray:
        c = np.cumsum(np.r_[0.0, v])
        return c[w:] - c[:-w]

    sa, sb = window_sum(a), window_sum(b)
    cov = window_sum(a * b) - sa * sb / w
    var_a = window_sum(a * a) - sa * sa / w
    var_b = window_sum(b * b) - sb * sb / w
    ok = (var_a > 1e-6) & (var_b > 1e-12)  # a still mouth or silence: no evidence either way
    out = np.zeros(sa.size)
    out[ok] = cov[ok] / np.sqrt(var_a[ok] * var_b[ok])
    return out


def loudness(video: Path, start: float, duration: float, fps: Fraction, frames: int) -> np.ndarray:
    """Audio loudness (RMS) for each video frame of the clip."""
    rate = 16000
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(video),
         "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"],
        capture_output=True,
    )
    audio = np.frombuffer(result.stdout, np.float32)
    per_frame = rate / float(fps)
    out = np.zeros(frames)
    for i in range(frames):
        chunk = audio[int(i * per_frame) : int((i + 1) * per_frame)]
        if chunk.size:
            out[i] = float(np.sqrt(np.mean(chunk * chunk)))
    return out


def runs(labels: np.ndarray, min_len: int) -> list[tuple[int, int, int]]:
    """Group labels into (start, end inclusive, label) runs of at least min_len frames."""
    out: list[list[int]] = []
    for i, label in enumerate(labels):
        if out and out[-1][2] == label:
            out[-1][1] = i
        else:
            out.append([i, i, int(label)])
    while len(out) > 1:
        k = min(range(len(out)), key=lambda j: out[j][1] - out[j][0])
        if out[k][1] - out[k][0] + 1 >= min_len:
            break
        # Fold the shortest run into its longer neighbour, then join matching neighbours.
        left = out[k - 1] if k > 0 else None
        right = out[k + 1] if k + 1 < len(out) else None
        if right is None or (left is not None and left[1] - left[0] >= right[1] - right[0]):
            left[1] = out[k][1]
        else:
            right[0] = out[k][0]
        del out[k]
        joined = [out[0]]
        for run in out[1:]:
            if run[2] == joined[-1][2]:
                joined[-1][1] = run[1]
            else:
                joined.append(run)
        out = joined
    return [(a, b, label) for a, b, label in out]


def speech_mask(speech: list[tuple[float, float]], frames: int, fps: float) -> np.ndarray:
    """True for frames where someone is speaking (from the transcript's word timings)."""
    mask = np.zeros(frames, dtype=bool)
    for start, end in speech:
        a = max(0, int((start - SPEECH_PAD_SECONDS) * fps))
        b = min(frames, int((end + SPEECH_PAD_SECONDS) * fps) + 1)
        mask[a:b] = True
    return mask
