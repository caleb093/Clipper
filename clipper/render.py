"""Compose vertical frames from the reframing plan and encode the final clip."""

import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from clipper.config import Settings
from clipper.media import VideoInfo, even, open_frame_reader, read_frames
from clipper.reframe import FramePlan

BACKGROUND_DIM = 0.55


def render_clip(
    info: VideoInfo,
    start: float,
    duration: float,
    plan: FramePlan,
    ass_file: Path | None,
    out_file: Path,
    settings: Settings,
    on_progress: Callable[[float, str], None] = lambda fraction, detail: None,
) -> None:
    ow, oh = settings.out_width, settings.out_height
    # Decode at no more than output height; 4K sources don't need full resolution.
    k = min(1.0, oh / info.height)
    rw, rh = even(info.width * k), even(info.height * k)

    fps = plan.fps
    expected = max(1, int(duration * float(fps)))
    log_file = out_file.with_suffix(".ffmpeg.log")

    cmd = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ow}x{oh}",
        "-r", f"{fps.numerator}/{fps.denominator}", "-i", "-",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(info.path.resolve()),
        "-map", "0:v:0", "-map", "1:a:0?",
    ]
    if ass_file is not None:
        # Run from the subtitle's folder so the filter sees a plain relative name;
        # Windows drive letters ("C:") otherwise need awkward escaping.
        cmd += ["-vf", f"subtitles={ass_file.name}"]
    cmd += [
        "-c:v", "libx264", "-preset", settings.preset, "-crf", str(settings.crf),
        "-pix_fmt", "yuv420p", "-profile:v", "high",
        "-c:a", "aac", "-b:a", "160k", "-ar", "48000",
        "-shortest", "-movflags", "+faststart",
        str(out_file.resolve()),
    ]

    cwd = ass_file.parent if ass_file is not None else out_file.parent
    stopped: BaseException | None = None
    with open(log_file, "w", encoding="utf-8", errors="replace") as log:
        encoder = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log, cwd=cwd)
        reader = open_frame_reader(info.path, start, duration, fps, rw, rh)
        background = _Background(ow, oh)
        started = time.time()
        try:
            for i, frame in enumerate(read_frames(reader, rw, rh)):
                layout, centre, rects = plan.at(i)
                if layout == "crop":
                    out = _crop(frame, centre, ow, oh)
                elif layout == "speaker":
                    out = _region(frame, rects[0], ow, oh)
                elif layout == "split":
                    out = _split(frame, rects, ow, oh)
                else:
                    out = _fit(frame, ow, oh, background)
                encoder.stdin.write(out.tobytes())
                if i % 15 == 0:
                    rate = (i + 1) / max(time.time() - started, 1e-6)
                    on_progress(min(1.0, (i + 1) / expected), f"{rate:.0f} frames/s")
            encoder.stdin.close()
        except (BrokenPipeError, OSError):
            pass  # encoder died; its log explains why
        except BaseException as exc:  # cancelled (or a bug): stop ffmpeg, clean up below
            stopped = exc
            encoder.kill()
        finally:
            if reader.poll() is None:
                reader.kill()
            code = encoder.wait()

    if stopped is not None:
        out_file.unlink(missing_ok=True)  # half-written clip
        log_file.unlink(missing_ok=True)
        raise stopped
    if code != 0:
        raise RuntimeError(f"ffmpeg failed for {out_file.name}:\n{log_file.read_text(encoding='utf-8', errors='replace')[-2000:]}")
    log_file.unlink(missing_ok=True)


def _crop(frame: np.ndarray, centre: float, ow: int, oh: int) -> np.ndarray:
    h, w = frame.shape[:2]
    cw = min(w, even(h * ow / oh))
    x0 = int(round(centre * w - cw / 2))
    x0 = max(0, min(w - cw, x0))
    region = frame[:, x0 : x0 + cw]
    interp = cv2.INTER_CUBIC if oh > h else cv2.INTER_AREA
    return cv2.resize(region, (ow, oh), interpolation=interp)


def _region(frame: np.ndarray, rect: tuple[float, float, float], tw: int, th: int) -> np.ndarray:
    """Cut a region (centre x, centre y, height; fractions of the frame) shaped like tw x th."""
    fh, fw = frame.shape[:2]
    cx, cy, hf = rect
    h = hf * fh
    w = h * tw / th
    if w > fw:
        w, h = fw, fw * th / tw
    w, h = int(round(w)), int(round(h))
    x0 = int(round(min(max(cx * fw - w / 2, 0), fw - w)))
    y0 = int(round(min(max(cy * fh - h / 2, 0), fh - h)))
    interp = cv2.INTER_CUBIC if th > h else cv2.INTER_AREA
    return cv2.resize(frame[y0 : y0 + h, x0 : x0 + w], (tw, th), interpolation=interp)


def _split(frame: np.ndarray, rects, ow: int, oh: int) -> np.ndarray:
    """Two people stacked: the first rect on top, the second below, with a thin divider."""
    top_h = oh // 2
    out = np.vstack([_region(frame, rects[0], ow, top_h), _region(frame, rects[1], ow, oh - top_h)])
    out[top_h - 2 : top_h + 2] = 12
    return out


class _Background:
    """Blurred, dimmed fill behind 'fit' shots. Blurring a tiny image keeps it cheap."""

    def __init__(self, ow: int, oh: int):
        self.ow, self.oh = ow, oh
        self.small = (max(2, ow // 8), max(2, oh // 8))

    def make(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        cw = min(w, even(h * self.ow / self.oh))
        x0 = (w - cw) // 2
        tiny = cv2.resize(frame[:, x0 : x0 + cw], self.small, interpolation=cv2.INTER_AREA)
        tiny = cv2.GaussianBlur(tiny, (0, 0), 4)
        bg = cv2.resize(tiny, (self.ow, self.oh), interpolation=cv2.INTER_LINEAR)
        return cv2.convertScaleAbs(bg, alpha=BACKGROUND_DIM)


def _fit(frame: np.ndarray, ow: int, oh: int, background: _Background) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = min(ow / w, oh / h)
    fw, fh = min(ow, even(w * scale)), min(oh, even(h * scale))
    fg = cv2.resize(frame, (fw, fh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    out = background.make(frame)
    y0, x0 = (oh - fh) // 2, (ow - fw) // 2
    out[y0 : y0 + fh, x0 : x0 + fw] = fg
    return out
