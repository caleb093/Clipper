"""HTTP API + static files for the local web UI."""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from clipper.chat import PROMPT_MARKER
from clipper.config import PROJECT_ROOT, Settings
from clipper.highlights import Clip, clean_hashtags
from clipper.media import require_binaries
from clipper.pipeline import OUTPUT_DIR, api_key_available, clips_from_reply, slugify, tiktok_caption, youtube_title
from clipper.sources import clean_path, is_url
from clipper.web.jobs import Job, clip_view, run_job

STATIC_DIR = Path(__file__).resolve().parent / "static"
UPLOAD_DIR = PROJECT_ROOT / "uploads"

app = FastAPI(title="Clipper", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_jobs: dict[str, Job] = {}
_current: Job | None = None
_lock = threading.Lock()
_api: dict = {}  # cached result of the API key check


@app.middleware("http")
async def same_origin_only(request: Request, call_next):
    """Other websites open in the browser must not be able to drive this local app."""
    origin = request.headers.get("origin")
    if request.method not in ("GET", "HEAD") and origin and urlparse(origin).netloc != request.headers.get("host"):
        return JSONResponse({"detail": "Requests from other websites are not allowed."}, status_code=403)
    return await call_next(request)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


# ------------------------------------------------------------------ status


def _api_status() -> dict:
    with _lock:
        if not _api:
            try:
                _api["available"] = api_key_available(Settings().llm_model)
                _api["error"] = ""
            except RuntimeError as exc:
                _api["available"], _api["error"] = False, str(exc)
        return dict(_api)


@app.get("/api/status")
def status() -> dict:
    d = Settings()
    return {
        "api": _api_status(),
        "active_job": _current.to_dict() if _current and _current.active else None,
        "defaults": {
            "content_type": d.content_type,
            "min_seconds": d.min_seconds,
            "max_seconds": d.max_seconds,
            "layout": d.layout,
            "captions": d.captions,
            "uppercase": d.uppercase_captions,
            "hook_seconds": d.hook_seconds,
            "whisper_model": d.whisper_model,
        },
    }


class ApiKey(BaseModel):
    key: str = Field(min_length=10, max_length=300)


@app.post("/api/check-key")
def check_api_key(body: ApiKey) -> dict:
    """Confirm a user's own key works. Nothing is stored: the browser keeps the key and
    sends it with each job, so one user's key is never used for anyone else."""
    import anthropic

    try:
        anthropic.Anthropic(api_key=body.key.strip()).models.retrieve(Settings().llm_model)
    except anthropic.AuthenticationError:
        raise HTTPException(400, "Anthropic rejected that key. Check that you copied all of it.") from None
    except anthropic.APIConnectionError:
        raise HTTPException(400, "Couldn't reach Anthropic to check the key. Check your internet connection.") from None
    except anthropic.APIStatusError as exc:
        raise HTTPException(400, f"Anthropic didn't accept the key (error {exc.status_code}).") from None
    return {"ok": True}


# -------------------------------------------------------------------- jobs


class JobOptions(BaseModel):
    content_type: Literal["explainer", "podcast"] = "explainer"
    min_seconds: float = Field(20, ge=5, le=180)
    max_seconds: float = Field(60, ge=10, le=600)
    max_clips: int | None = Field(None, ge=1, le=50)
    layout: Literal["auto", "crop", "fit"] = "auto"
    captions: bool = True
    uppercase: bool = True
    hook_seconds: float = Field(4, ge=0, le=30)
    whisper_model: Literal["tiny", "base", "small", "medium", "large-v3"] = "small"
    picker: Literal["chat", "api"] | None = None


class NewJob(BaseModel):
    source: str = Field(min_length=1)
    options: JobOptions = JobOptions()
    api_key: str | None = Field(None, max_length=300)  # the user's own key; used for this job only


@app.post("/api/jobs")
def start_job(body: NewJob) -> dict:
    global _current
    o = body.options
    if o.min_seconds >= o.max_seconds:
        raise HTTPException(400, "The shortest clip length must be less than the longest.")
    if not is_url(body.source) and not clean_path(body.source).is_file():
        raise HTTPException(400, f"There's no video file at {clean_path(body.source)}. Check the path, or paste a link.")
    user_key = (body.api_key or "").strip() or None
    api = _api_status()
    can_use_api = bool(user_key) or api["available"]
    picker = o.picker or ("api" if can_use_api else "chat")
    if picker == "api" and not can_use_api:
        raise HTTPException(400, "Add your Anthropic API key under Options, or pick clips with the Claude app.")

    settings = Settings(
        content_type=o.content_type,
        whisper_model=o.whisper_model,
        max_clips=o.max_clips,
        min_seconds=o.min_seconds,
        max_seconds=o.max_seconds,
        layout=o.layout,
        captions=o.captions,
        uppercase_captions=o.uppercase,
        hook_seconds=o.hook_seconds,
    )
    with _lock:
        if _current and _current.active:
            raise HTTPException(409, "Another video is still being processed. Finish or cancel it first.")
        job = Job(body.source.strip(), settings, picker, api_key=user_key if picker == "api" else None)
        _jobs[job.id] = job
        _current = job
    threading.Thread(target=run_job, args=(job,), daemon=True).start()
    return {"id": job.id}


def _job(job_id: str) -> Job:
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "That job no longer exists (the app may have been restarted).")
    return job


@app.get("/api/jobs/{job_id}")
def job_state(job_id: str) -> dict:
    return _job(job_id).to_dict()


@app.get("/api/jobs/{job_id}/prompt")
def job_prompt(job_id: str) -> PlainTextResponse:
    return PlainTextResponse(_job(job_id).prompt)


@app.get("/api/jobs/{job_id}/source")
def job_source(job_id: str) -> FileResponse:
    job = _job(job_id)
    if not job.project:
        raise HTTPException(404, "The video isn't ready yet.")
    return FileResponse(job.project.video)


class Reply(BaseModel):
    text: str


@app.post("/api/jobs/{job_id}/reply")
def job_reply(job_id: str, body: Reply) -> dict:
    job = _job(job_id)
    if job.status != "waiting_reply" or not job.project:
        raise HTTPException(409, "This job isn't waiting for a reply.")
    text = body.text.strip()
    if not text:
        raise HTTPException(400, "Paste the reply first.")
    if PROMPT_MARKER in text:
        raise HTTPException(400, "That's the prompt, not the reply. Send the prompt in the chat, then copy the answer.")
    try:
        clips = clips_from_reply(job.project, job.settings, text)
    except ValueError as exc:
        raise HTTPException(400, f"That doesn't look like the reply to this prompt: {exc}.") from None
    (job.project.out_dir / "claude-reply.txt").write_text(text, encoding="utf-8")
    job.clips = clips
    job.resume()
    return {"clips": len(clips)}


MIN_CLIP_SECONDS = 3.0


class ClipEdit(BaseModel):
    index: int
    include: bool = True
    start: float | None = None  # seconds in the source video; None keeps the AI's choice
    end: float | None = None
    title: str = ""
    hook_text: str = ""
    description: str = ""
    hashtags: str = ""


class RenderRequest(BaseModel):
    clips: list[ClipEdit]


@app.post("/api/jobs/{job_id}/render")
def job_render(job_id: str, body: RenderRequest) -> dict:
    job = _job(job_id)
    if job.status != "review":
        raise HTTPException(409, "This job isn't waiting for a review.")
    chosen: list[Clip] = []
    for edit in body.clips:
        if not edit.include or not 0 <= edit.index < len(job.clips):
            continue
        clip = job.clips[edit.index]
        if edit.start is not None and edit.end is not None:
            start = max(0.0, edit.start)
            end = min(job.project.info.duration, edit.end)
            if end - start < MIN_CLIP_SECONDS:
                raise HTTPException(400, f"Clip {edit.index + 1} is shorter than {MIN_CLIP_SECONDS:.0f} seconds.")
            if (round(start, 2), round(end, 2)) != (round(clip.start, 2), round(clip.end, 2)):
                clip.start, clip.end = round(start, 3), round(end, 3)
                clip.transcript = " ".join(w.text for w in job.project.words if w.end > start and w.start < end)
        clip.title = edit.title.strip()
        clip.hook_text = edit.hook_text.strip()
        clip.description = edit.description.strip()
        clip.hashtags = clean_hashtags(re.split(r"[\s,]+", edit.hashtags))
        chosen.append(clip)
    if not chosen:
        raise HTTPException(400, "Tick at least one clip to render.")
    job.clips = chosen
    job.resume()
    return {"clips": len(chosen)}


@app.post("/api/jobs/{job_id}/cancel")
def job_cancel(job_id: str) -> dict:
    _job(job_id).cancel()
    return {"ok": True}


@app.post("/api/upload")
def upload(file: UploadFile) -> dict:
    """Copy a dropped video into uploads/ so the pipeline can work on it like any local file."""
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", Path(file.filename or "video.mp4").name).strip() or "video.mp4"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / name
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out, length=8 * 1024 * 1024)
    return {"path": str(dest)}


# ---------------------------------------------------------------- projects


def _project_dir(slug: str) -> Path:
    folder = (OUTPUT_DIR / slug).resolve()
    if folder.parent != OUTPUT_DIR.resolve() or not (folder / "clips.json").is_file():
        raise HTTPException(404, "Project not found.")
    return folder


def _rendered_clips(folder: Path) -> tuple[dict, list[dict]]:
    data = json.loads((folder / "clips.json").read_text(encoding="utf-8"))
    clips = []
    for n, raw in enumerate(data.get("clips", []), 1):
        clip = Clip(**{k: v for k, v in raw.items() if k in Clip.__dataclass_fields__})
        # Projects rendered before clips.json recorded file names use the naming scheme.
        file = clip.file or f"{n:02d}-{slugify(clip.title) or 'clip'}.mp4"
        if not (folder / file).is_file():
            continue
        view = clip_view(clip)
        view.update(
            file=file,
            url=f"/media/{folder.name}/{file}",
            thumb=f"/thumb/{folder.name}/{file}",
            caption=tiktok_caption(clip),
            youtube_title=youtube_title(clip),
        )
        clips.append(view)
    return data, clips


@app.get("/api/projects")
def projects() -> list[dict]:
    items = []
    if OUTPUT_DIR.is_dir():
        for folder in OUTPUT_DIR.iterdir():
            if not (folder / "clips.json").is_file():
                continue
            try:
                data, clips = _rendered_clips(folder)
            except (ValueError, TypeError, OSError):
                continue
            if not clips:
                continue
            items.append({
                "slug": folder.name,
                "title": Path(data.get("source", folder.name)).stem,
                "clips": len(clips),
                "updated": (folder / "clips.json").stat().st_mtime,
                "thumb": clips[0]["thumb"],
            })
    return sorted(items, key=lambda p: -p["updated"])


@app.get("/api/projects/{slug}")
def project(slug: str) -> dict:
    folder = _project_dir(slug)
    data, clips = _rendered_clips(folder)
    return {
        "slug": slug,
        "title": Path(data.get("source", slug)).stem,
        "folder": str(folder),
        "clips": clips,
    }


@app.post("/api/projects/{slug}/open")
def open_folder(slug: str) -> dict:
    folder = _project_dir(slug)
    if sys.platform == "win32":
        os.startfile(folder)  # noqa: S606 - local app, opens Explorer
    return {"ok": True}


def _clip_file(slug: str, filename: str) -> Path:
    folder = _project_dir(slug)
    path = (folder / filename).resolve()
    if path.parent != folder or path.suffix.lower() != ".mp4" or not path.is_file():
        raise HTTPException(404, "File not found.")
    return path


@app.get("/media/{slug}/{filename}")
def media(slug: str, filename: str) -> FileResponse:
    return FileResponse(_clip_file(slug, filename), media_type="video/mp4")


@app.get("/thumb/{slug}/{filename}")
def thumb(slug: str, filename: str) -> FileResponse:
    """A still frame of a clip, made once with ffmpeg; much lighter than loading the video."""
    video = _clip_file(slug, filename)
    image = video.parent / ".cache" / "thumbs" / f"{video.stem}.jpg"
    if not image.is_file() or image.stat().st_mtime < video.stat().st_mtime:
        require_binaries()
        image.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", "1.5", "-i", str(video),
             "-frames:v", "1", "-vf", "scale=360:-2", "-q:v", "4", str(image)],
            capture_output=True,
        )
        if not image.is_file():
            raise HTTPException(404, "Couldn't make a thumbnail.")
    return FileResponse(image, media_type="image/jpeg")
