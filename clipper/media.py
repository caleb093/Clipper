"""Thin wrappers around the ffmpeg / ffprobe binaries."""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


class MediaError(RuntimeError):
    pass


@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: Fraction
    duration: float
    has_audio: bool


def require_binaries() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        _use_winget_ffmpeg()
    missing = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
    if missing:
        raise MediaError(
            f"{', '.join(missing)} not found on PATH. Install ffmpeg "
            "(Windows: `winget install Gyan.FFmpeg`) and open a new terminal."
        )


def _use_winget_ffmpeg() -> None:
    """winget adds ffmpeg to PATH, but terminals opened before the install don't see it yet."""
    packages = Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Packages"
    for bin_dir in sorted(packages.glob("Gyan.FFmpeg*/*/bin"), reverse=True):
        if (bin_dir / "ffmpeg.exe").exists():
            os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            return


def probe(path: Path) -> VideoInfo:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise MediaError(f"ffprobe failed on {path}: {result.stderr.strip()}")
    data = json.loads(result.stdout)

    video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise MediaError(f"No video stream in {path}")

    width, height = int(video["width"]), int(video["height"])
    # Phone footage stores portrait video as landscape + a rotation flag.
    rotation = 0
    for side in video.get("side_data_list", []):
        if "rotation" in side:
            rotation = abs(int(side["rotation"]))
    if rotation in (90, 270):
        width, height = height, width

    fps = Fraction(video.get("avg_frame_rate") or "0/1")
    if fps <= 0:
        fps = Fraction(video.get("r_frame_rate") or "30/1")

    duration = float(data["format"].get("duration") or video.get("duration") or 0)
    has_audio = any(s["codec_type"] == "audio" for s in data["streams"])
    return VideoInfo(Path(path), width, height, fps, duration, has_audio)


def output_fps(source_fps: Fraction, max_fps: float) -> Fraction:
    """Keep the source rate unless it is above max_fps (e.g. 60 -> 30)."""
    if float(source_fps) <= max_fps + 0.5:
        return source_fps
    halved = source_fps / 2
    if abs(float(halved) - max_fps) < 1.0:
        return halved
    return Fraction(max_fps).limit_denominator(1001)


def load_audio(path: Path, sample_rate: int = 16000):
    """Decode the audio track to mono float32 samples (the format Whisper expects)."""
    import numpy as np

    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", "1", "-ar", str(sample_rate), "-f", "f32le", "-"],
        capture_output=True,
    )
    if result.returncode != 0:
        raise MediaError(f"Could not read audio from {path}: {result.stderr.decode(errors='replace').strip()}")
    return np.frombuffer(result.stdout, np.float32)


def open_frame_reader(
    path: Path, start: float, duration: float, fps: Fraction, width: int, height: int
) -> subprocess.Popen:
    """Decode a time range to raw BGR frames of a fixed size on stdout.

    Both the analysis pass and the render pass use this with the same
    start/duration/fps, so frame N means the same moment in both.
    """
    vf = f"fps={fps.numerator}/{fps.denominator},scale={width}:{height}:flags=area"
    cmd = [
        "ffmpeg", "-v", "error", "-nostdin",
        "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(path),
        "-an", "-sn", "-vf", vf,
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=width * height * 3 * 2)


def read_frames(proc: subprocess.Popen, width: int, height: int):
    import numpy as np

    frame_bytes = width * height * 3
    try:
        while True:
            buf = proc.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.wait()


def even(n: float) -> int:
    return max(2, int(round(n / 2)) * 2)
