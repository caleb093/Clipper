"""Background jobs for the web UI: one video at a time, pausing for the user when needed."""

import threading
import time
import traceback
import uuid

from clipper.chat import chat_prompt
from clipper.config import Settings
from clipper.highlights import Clip
from clipper.pipeline import (
    Project,
    clip_prompt,
    get_video,
    pick_with_api,
    prepare,
    render_all,
    save_plan,
)
from clipper.progress import Cancelled, Reporter
from clipper.sources import is_url

ACTIVE = ("running", "waiting_reply", "review")
MAX_LOG_LINES = 300


class Job:
    def __init__(self, source: str, settings: Settings, picker: str, api_key: str | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.source = source
        self.settings = settings
        self.picker = picker  # "api" | "chat"
        # The user's own Anthropic key, held in memory for this job only: never
        # written to disk and deliberately left out of to_dict().
        self.api_key = api_key
        self.created = time.time()

        self.status = "running"  # running | waiting_reply | review | done | error | cancelled
        self.stage = "download" if is_url(source) else "transcribe"
        self.stage_label = "Starting"
        self.progress: float | None = None
        self.detail = ""
        self.log: list[str] = []
        self.error = ""
        self.title = source.strip()

        self.project: Project | None = None
        self.prompt = ""
        self.clips: list[Clip] = []
        self.message = ""  # final note, e.g. when no clip was strong enough

        self._cancel = threading.Event()
        self._resume = threading.Event()

    # -- control from the web requests

    def cancel(self) -> None:
        self._cancel.set()
        self._resume.set()

    def resume(self) -> None:
        self._resume.set()

    def check_cancelled(self) -> None:
        if self._cancel.is_set():
            raise Cancelled()

    def wait_for_user(self, status: str) -> None:
        self.status = status
        self._resume.clear()
        while not self._resume.wait(0.5):
            pass
        self.check_cancelled()
        self.status = "running"

    @property
    def active(self) -> bool:
        return self.status in ACTIVE

    def to_dict(self) -> dict:
        data = {
            "id": self.id,
            "status": self.status,
            "stage": self.stage,
            "stage_label": self.stage_label,
            "progress": self.progress,
            "detail": self.detail,
            "log": self.log[-80:],
            "error": self.error,
            "message": self.message,
            "title": self.title,
            "picker": self.picker,
            "slug": self.project.out_dir.name if self.project else "",
            "video_duration": self.project.info.duration if self.project else None,
        }
        if self.status == "waiting_reply":
            data["prompt_chars"] = len(self.prompt)
        if self.status == "review":
            data["clips"] = [clip_view(c) for c in self.clips]
        return data


class JobReporter(Reporter):
    def __init__(self, job: Job):
        self.job = job

    def stage(self, key: str, label: str) -> None:
        self.job.check_cancelled()
        self.job.stage, self.job.stage_label = key, label
        self.job.progress, self.job.detail = None, ""
        self.log(label)

    def progress(self, fraction: float | None, detail: str = "") -> None:
        self.job.check_cancelled()  # how a cancel reaches a long transcription or render
        self.job.progress, self.job.detail = fraction, detail

    def log(self, message: str) -> None:
        self.job.log.append(message)
        del self.job.log[:-MAX_LOG_LINES]


def run_job(job: Job) -> None:
    """Runs in a background thread; the web requests read and steer the job object."""
    reporter = JobReporter(job)
    try:
        video = get_video(job.source, reporter)
        job.title = video.stem
        project = prepare(video, job.settings, reporter)
        job.project = project

        if job.picker == "api":
            clips = pick_with_api(project, job.settings, reporter, api_key=job.api_key)
        else:
            reporter.stage("pick", "Pick the clips with Claude")
            job.prompt = chat_prompt(clip_prompt(project, job.settings), job.settings.content_type)
            (project.out_dir / "claude-prompt.txt").write_text(job.prompt, encoding="utf-8")
            job.wait_for_user("waiting_reply")  # the /reply request fills job.clips
            clips = job.clips

        if not clips:
            job.message = "Nothing in this video was strong enough to stand alone as a short, so no clips were made."
            job.status = "done"
            return

        job.clips = clips
        save_plan(project, clips)
        reporter.stage("review", "Review the clips")
        job.wait_for_user("review")  # the /render request may drop or edit clips

        save_plan(project, job.clips)
        render_all(project, job.clips, job.settings, reporter)
        job.status = "done"
    except Cancelled:
        job.status = "cancelled"
        reporter.log("Cancelled.")
    except Exception as exc:  # shown in the UI; full traceback in the server window
        traceback.print_exc()
        job.error = str(exc) or type(exc).__name__
        job.status = "error"


def clip_view(clip: Clip) -> dict:
    return {
        "start": clip.start,
        "end": clip.end,
        "duration": round(clip.duration, 1),
        "title": clip.title,
        "hook_text": clip.hook_text,
        "description": clip.description,
        "hashtags": clip.hashtags,
        "score": clip.score,
        "reason": clip.reason,
        "transcript": clip.transcript,
        "file": clip.file,
    }
