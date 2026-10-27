"""Turn what the user typed (a YouTube link or a file path) into a local video file."""

import re
import shutil
from pathlib import Path

from clipper.progress import Reporter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOWNLOAD_DIR = PROJECT_ROOT / "downloads"

# Prefer H.264 up to 1080p: fast to decode on any machine and plays in every browser.
YOUTUBE_FORMAT = (
    "bv*[height<=1080][vcodec^=avc1]+ba[ext=m4a]"
    "/bv*[height<=1080][ext=mp4]+ba[ext=m4a]"
    "/bv*[height<=1080]+ba"
    "/b[height<=1080]/b"
)


def is_url(text: str) -> bool:
    return bool(re.match(r"^\s*https?://", text, re.I))


def clean_path(text: str) -> Path:
    """Accept paths pasted from Explorer's "Copy as path" (with quotes) or typed by hand."""
    return Path(text.strip().strip('"').strip("'"))


def download(url: str, reporter: Reporter, dest: Path = DOWNLOAD_DIR) -> Path:
    """Download a YouTube (or other yt-dlp supported) video; reuses an earlier download."""
    from yt_dlp import YoutubeDL
    from yt_dlp.utils import DownloadError

    def on_progress(d: dict) -> None:
        if d.get("status") == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate")
            done = d.get("downloaded_bytes") or 0
            speed = d.get("speed")
            detail = f"{done / 1e6:.0f} MB"
            if total:
                detail += f" of {total / 1e6:.0f} MB"
            if speed:
                detail += f" at {speed / 1e6:.1f} MB/s"
            reporter.progress(done / total if total else None, detail)
        elif d.get("status") == "finished":
            reporter.progress(1.0, "merging video and audio")

    ffmpeg = shutil.which("ffmpeg")
    options = {
        "format": YOUTUBE_FORMAT,
        "merge_output_format": "mp4",
        # One folder per video id keeps the file name readable and downloads unique.
        "outtmpl": str(dest / "%(id)s" / "%(title).120B.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [on_progress],
        # YouTube needs a JavaScript runtime for some formats; use whichever is installed.
        "js_runtimes": {"deno": {}, "node": {}},
    }
    if ffmpeg:
        options["ffmpeg_location"] = str(Path(ffmpeg).parent)

    try:
        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(url, download=True)
    except DownloadError as exc:
        message = re.sub(r"^ERROR:\s*", "", str(exc))
        raise RuntimeError(f"Couldn't download the video: {message}") from exc

    if info.get("_type") == "playlist":
        raise RuntimeError("That link is a playlist. Paste the link of a single video.")
    files = [d.get("filepath") for d in info.get("requested_downloads") or [] if d.get("filepath")]
    if not files or not Path(files[0]).is_file():
        raise RuntimeError("The download finished but the video file wasn't found.")
    reporter.log(f"Downloaded: {info.get('title', 'video')}")
    return Path(files[0])
