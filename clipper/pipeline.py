"""Long video in -> vertical clips + post text out.

The stages are separate functions so that the command line (run, below) and
the web UI (clipper.web) can drive them differently; the UI, for example,
pauses for the user to paste Claude's reply and to review clips before rendering.
"""

import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from clipper.captions import write_ass
from clipper.chat import pick_in_chat
from clipper.config import Settings
from clipper.highlights import (
    Clip,
    _fmt_time,
    build_prompt,
    check_sentence_ranges,
    clips_from_sentences,
    load_clips,
    parse_clip_reply,
    save_clips,
    select_clips,
)
from clipper.media import VideoInfo, output_fps, probe, require_binaries
from clipper.progress import ConsoleReporter, Reporter
from clipper.reframe import plan_clip
from clipper.render import render_clip
from clipper.sources import clean_path, download, is_url
from clipper.transcribe import Sentence, Word, build_sentences, transcribe

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"


@dataclass
class Project:
    """A transcribed video and the folder its clips go to."""

    video: Path
    info: VideoInfo
    out_dir: Path
    words: list[Word]
    sentences: list[Sentence]
    language: str

    @property
    def plan_file(self) -> Path:
        return self.out_dir / "clips.json"


# ------------------------------------------------------------------ stages


def get_video(source: str, reporter: Reporter) -> Path:
    """A web link is downloaded (once); anything else is treated as a local file path."""
    require_binaries()
    if is_url(source):
        reporter.stage("download", "Downloading the video")
        return download(source.strip(), reporter)
    path = clean_path(source)
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")
    return path.resolve()  # ffmpeg runs from other folders, so relative paths would break


def prepare(video: Path, settings: Settings, reporter: Reporter, out_root: Path = OUTPUT_DIR) -> Project:
    """Probe and transcribe (or load the saved transcript)."""
    require_binaries()
    info = probe(video)
    if not info.has_audio:
        raise RuntimeError(f"{video.name} has no audio track; there's no narration to work from.")
    reporter.log(f"{video.name}: {info.width}x{info.height}, {float(info.fps):.2f} fps, {_fmt_time(info.duration)}")

    out_dir = out_root / (slugify(video.stem) or "video")
    cache = out_dir / ".cache"
    cache.mkdir(parents=True, exist_ok=True)

    reporter.stage("transcribe", f"Transcribing (whisper '{settings.whisper_model}')")
    words, language = transcribe(video, cache, settings.whisper_model, settings.language, settings.device, reporter)
    sentences = build_sentences(words)
    reporter.log(f"{len(words)} words, {len(sentences)} sentences, language: {language}")
    return Project(video, info, out_dir, words, sentences, language)


def clip_prompt(project: Project, settings: Settings) -> str:
    return build_prompt(
        project.sentences,
        video_name=project.video.stem,
        video_duration=project.info.duration,
        language=project.language,
        max_clips=settings.max_clips,
        min_s=settings.min_seconds,
        max_s=settings.max_seconds,
        content_type=settings.content_type,
    )


def pick_with_api(project: Project, settings: Settings, reporter: Reporter, api_key: str | None = None) -> list[Clip]:
    reporter.stage("pick", f"Asking {settings.llm_model} to find the strong moments")
    reporter.progress(None, "Claude is reading the transcript")
    clips = select_clips(
        clip_prompt(project, settings),
        project.sentences,
        project.words,
        video_duration=project.info.duration,
        min_s=settings.min_seconds,
        max_s=settings.max_seconds,
        model=settings.llm_model,
        content_type=settings.content_type,
        api_key=api_key,
    )
    return _limit(clips, settings)


def clips_from_reply(project: Project, settings: Settings, text: str) -> list[Clip]:
    """Clips from a reply pasted from the Claude app. Raises ValueError with a readable reason."""
    items = parse_clip_reply(text)
    check_sentence_ranges(items, len(project.sentences))
    return _from_items(project, settings, items)


def clips_from_file(project: Project, settings: Settings, path: Path) -> list[Clip]:
    return load_clips(
        path, project.sentences, project.words, project.info.duration, settings.min_seconds, settings.max_seconds
    )


def save_plan(project: Project, clips: list[Clip]) -> Path:
    save_clips(project.plan_file, project.video, clips)
    return project.plan_file


def render_all(project: Project, clips: list[Clip], settings: Settings, reporter: Reporter) -> list[Clip]:
    """Render every clip; each clip's .file is set to its mp4 name."""
    reporter.stage("render", "Rendering")
    info = project.info
    fps = output_fps(info.fps, settings.max_fps)
    out_aspect = settings.out_width / settings.out_height
    total = len(clips)

    for n, clip in enumerate(clips, 1):
        started = time.time()
        name = f"{n:02d}-{slugify(clip.title) or 'clip'}"
        label = f"Clip {n} of {total}"
        podcast = settings.content_type == "podcast"
        reporter.progress((n - 1) / total, f"{label}: {'following the speakers' if podcast else 'finding the subject in each shot'}")

        speech = [(w.start - clip.start, w.end - clip.start) for w in project.words
                  if w.end > clip.start and w.start < clip.end]
        plan = plan_clip(
            project.video, clip.start, clip.duration, fps, info.width, info.height, settings.layout, out_aspect,
            content_type=settings.content_type, speech=speech,
        )

        ass_file = None
        if settings.captions or (clip.hook_text and settings.hook_seconds > 0):
            ass_file = project.out_dir / ".cache" / f"clip-{n:02d}.ass"
            write_ass(
                ass_file,
                project.words,
                clip.start,
                clip.end,
                hook_text=clip.hook_text,
                hook_seconds=settings.hook_seconds,
                captions=settings.captions,
                uppercase=settings.uppercase_captions,
                font=settings.font,
                width=settings.out_width,
                height=settings.out_height,
                caption_y=plan.caption_y_at,
                hook_y=240 if podcast else 470,  # podcasts: above the speaker's head
            )

        def on_progress(fraction: float, detail: str, n: int = n, label: str = label) -> None:
            reporter.progress((n - 1 + fraction) / total, f"{label}: {detail}")

        render_clip(
            info, clip.start, clip.duration, plan, ass_file, project.out_dir / f"{name}.mp4", settings, on_progress
        )
        write_post_text(project.out_dir / f"{name}.txt", clip)
        clip.file = f"{name}.mp4"
        reporter.log(f"{label} done in {time.time() - started:.0f}s: {clip.title} ({_layout_summary(plan)})")

    save_plan(project, clips)  # remembers which mp4 belongs to which clip
    return clips


# ------------------------------------------------------------ command line


def run(
    source: str,
    settings: Settings,
    out_root: Path = OUTPUT_DIR,
    clips_file: Path | None = None,
    dry_run: bool = False,
    chat: bool = False,
) -> Path:
    reporter = ConsoleReporter()
    require_binaries()
    if clips_file is None and not chat and not api_key_available(settings.llm_model):
        reporter.log("No Anthropic API key found, so you'll pick clips in the Claude app (free with a Claude subscription).")
        chat = True

    video = get_video(source, reporter)
    project = prepare(video, settings, reporter, out_root)

    if clips_file is not None:
        reporter.stage("pick", f"Loading clips from {clips_file}")
        clips = clips_from_file(project, settings, clips_file)
    elif chat:
        reporter.stage("pick", "Finding the strong moments, in the Claude app")
        items = pick_in_chat(
            clip_prompt(project, settings), project.out_dir, len(project.sentences), settings.content_type
        )
        clips = _from_items(project, settings, items)
    else:
        clips = pick_with_api(project, settings, reporter)

    if not clips:
        reporter.log("Nothing in this video was strong enough to stand alone as a short, so no clips were made.")
        return project.out_dir
    reporter.log(f"{len(clips)} strong moment{'s' if len(clips) != 1 else ''} found")
    for n, clip in enumerate(clips, 1):
        reporter.log(f"{n:2d}. [{_fmt_time(clip.start)}-{_fmt_time(clip.end)}, {clip.duration:4.0f}s] "
                     f"{clip.score:3d}  {clip.title}")
    reporter.log(f"saved to {save_plan(project, clips)}")
    if dry_run:
        reporter.log("Dry run: nothing rendered. Edit clips.json if you like, then run again with --clips-file.")
        return project.out_dir

    render_all(project, clips, settings, reporter)
    reporter.log(f"All done. Clips and post text are in: {project.out_dir.resolve()}")
    return project.out_dir


# ----------------------------------------------------------------- helpers


def _from_items(project: Project, settings: Settings, items: list[dict]) -> list[Clip]:
    clips = clips_from_sentences(
        items, project.sentences, project.words, settings.min_seconds, settings.max_seconds, project.info.duration
    )
    return _limit(clips, settings)


LAYOUT_NAMES = {
    "speaker": "on the speaker",
    "split": "split screen",
    "crop": "cropped to the subject",
    "fit": "full frame",
}


def _layout_summary(plan) -> str:
    """e.g. "62% on the speaker, 30% split screen, 8% full frame"."""
    total = max(1, len(plan.layouts))
    shares = {name: sum(1 for layout in plan.layouts if layout == key) / total for key, name in LAYOUT_NAMES.items()}
    return ", ".join(f"{share:.0%} {name}" for name, share in sorted(shares.items(), key=lambda kv: -kv[1]) if share)


def _limit(clips: list[Clip], settings: Settings) -> list[Clip]:
    return clips[: settings.max_clips] if settings.max_clips else clips  # clips are best-first


def write_post_text(path: Path, clip: Clip) -> None:
    text = f"""TITLE
{clip.title}

ON-SCREEN HOOK
{clip.hook_text}

TIKTOK / REELS CAPTION
{tiktok_caption(clip)}

YOUTUBE SHORTS TITLE
{youtube_title(clip)}

YOUTUBE SHORTS DESCRIPTION
{tiktok_caption(clip)}

---
Source: {_fmt_time(clip.start)} - {_fmt_time(clip.end)} ({clip.duration:.0f}s)
"""
    if clip.score:
        text += f"AI score: {clip.score}/100\n"
    if clip.reason:
        text += f"Why this clip: {clip.reason}\n"
    text += f"\nTranscript:\n{clip.transcript}\n"
    path.write_text(text, encoding="utf-8")


def tiktok_caption(clip: Clip) -> str:
    tags = " ".join(f"#{t}" for t in clip.hashtags)
    return f"{clip.description}\n\n{tags}".strip()


def youtube_title(clip: Clip) -> str:
    return f"{clip.title} #shorts"


def slugify(text: str, limit: int = 50) -> str:
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return text[:limit].rstrip("-")


def api_key_available(model: str) -> bool:
    """False if no API key is set up. A key that is set but broken fails here,
    in seconds, rather than after a long transcription."""
    import anthropic

    try:
        anthropic.Anthropic().models.retrieve(model)
        return True
    except anthropic.AuthenticationError as exc:
        raise RuntimeError("Your Anthropic API key was rejected. Check ANTHROPIC_API_KEY in the .env file.") from exc
    except anthropic.NotFoundError as exc:
        raise RuntimeError(f"Model '{model}' wasn't found. Check the --model value.") from exc
    except anthropic.APIConnectionError as exc:
        raise RuntimeError("Can't reach the Anthropic API. Check your internet connection.") from exc
    except TypeError:  # the SDK raises this when no key is configured at all
        return False
