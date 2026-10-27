import argparse
import re
import sys
from pathlib import Path

from clipper.config import Settings, load_env

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    defaults = Settings()
    p = argparse.ArgumentParser(
        prog="clipper",
        description="Turn a long narrated video into vertical short-form clips with captions, titles and hashtags.",
    )
    p.add_argument("source", help="a YouTube link or the path of a video file")
    p.add_argument("--type", dest="content_type", choices=["explainer", "podcast"], default=defaults.content_type,
                   help="explainer: voice-over with images/B-roll (default). "
                        "podcast: people talking on camera; follows the speaker, split screen for back-and-forth")
    p.add_argument("-o", "--output", type=Path, default=PROJECT_ROOT / "output", help="output folder (default: ./output)")

    g = p.add_argument_group("choosing clips")
    g.add_argument("--max-clips", type=int, default=None,
                   help="optional ceiling on the number of clips (default: no limit; the AI includes every moment "
                        "strong enough to post, and no weaker ones)")
    g.add_argument("--min", type=float, default=defaults.min_seconds, help=f"shortest clip in seconds (default {defaults.min_seconds:.0f})")
    g.add_argument("--max", type=float, default=defaults.max_seconds, help=f"longest clip in seconds (default {defaults.max_seconds:.0f})")
    g.add_argument("--model", default=defaults.llm_model, help=f"Claude model for picking clips (default {defaults.llm_model})")
    g.add_argument("--chat", action="store_true",
                   help="pick clips by copy-pasting with the Claude app (free with a Claude subscription); "
                        "this is automatic when no API key is set")
    g.add_argument("--clips-file", type=Path,
                   help="use clips from a clips.json or a saved Claude reply instead of asking the AI")
    g.add_argument("--dry-run", action="store_true", help="pick clips and save clips.json, but don't render")

    g = p.add_argument_group("video")
    g.add_argument("--layout", choices=["auto", "crop", "fit"], default=defaults.layout,
                   help="auto: crop to the subject when there is one, else show the full frame over a blurred background")
    g.add_argument("--no-captions", action="store_true", help="don't burn in word-by-word captions")
    g.add_argument("--no-uppercase", action="store_true", help="keep caption capitalisation as spoken")
    g.add_argument("--font", default=defaults.font, help=f"caption font (default '{defaults.font}')")
    g.add_argument("--hook-seconds", type=float, default=defaults.hook_seconds,
                   help=f"how long the hook text stays on screen, 0 to hide (default {defaults.hook_seconds:.0f})")
    g.add_argument("--crf", type=int, default=defaults.crf, help=f"x264 quality, lower is better (default {defaults.crf})")
    g.add_argument("--preset", default=defaults.preset, help=f"x264 speed preset (default {defaults.preset})")

    g = p.add_argument_group("transcription")
    g.add_argument("--whisper-model", default=defaults.whisper_model,
                   help="tiny | base | small | medium | large-v3 (bigger is more accurate but slower)")
    g.add_argument("--language", default=None, help="spoken language code, e.g. en (default: auto-detect)")
    g.add_argument("--device", choices=["auto", "cpu", "cuda"], default=defaults.device)
    p.add_argument("--debug", action="store_true", help="show full tracebacks on errors")

    args = p.parse_args(argv)
    load_env()

    from clipper.sources import clean_path, is_url

    paths = [args.clips_file]
    if not is_url(args.source):
        paths.append(clean_path(args.source))
    for path in paths:
        if path is not None and not path.is_file():
            p.error(f"file not found: {path}{_dollar_hint(path)}")
    if args.min >= args.max:
        p.error("--min must be smaller than --max")
    if args.max_clips is not None and args.max_clips < 1:
        p.error("--max-clips must be at least 1")

    settings = Settings(
        content_type=args.content_type,
        whisper_model=args.whisper_model,
        language=args.language,
        device=args.device,
        llm_model=args.model,
        max_clips=args.max_clips,
        min_seconds=args.min,
        max_seconds=args.max,
        layout=args.layout,
        captions=not args.no_captions,
        uppercase_captions=not args.no_uppercase,
        font=args.font,
        hook_seconds=args.hook_seconds,
        crf=args.crf,
        preset=args.preset,
    )

    from clipper.pipeline import run

    try:
        run(args.source, settings, args.output, clips_file=args.clips_file, dry_run=args.dry_run, chat=args.chat)
    except KeyboardInterrupt:
        print("\nStopped.")
        return 130
    except Exception as exc:  # show a readable message instead of a traceback for expected failures
        if args.debug:
            raise
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    return 0


def _dollar_hint(missing: Path) -> str:
    """PowerShell silently drops "$87" from a double-quoted path; find the real file and say how to quote it."""
    folder = missing.parent
    if not folder.is_dir():
        return ""
    for candidate in folder.iterdir():
        if "$" in candidate.name and re.sub(r"\$\w+", "", candidate.name) == missing.name:
            return (
                "\n\nThe real file name contains a $ sign, which PowerShell removes inside double quotes. "
                f"Use single quotes instead:\n  .\\clip '{candidate}'"
            )
    return ""
