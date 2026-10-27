import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_env() -> None:
    """Read ANTHROPIC_API_KEY (and anything else) from the project's .env file."""
    try:
        from dotenv import load_dotenv

        load_dotenv(PROJECT_ROOT / ".env")
        load_dotenv()
    except ImportError:
        pass
    if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        os.environ.pop("ANTHROPIC_API_KEY", None)  # an empty "ANTHROPIC_API_KEY=" line means no key


CONTENT_TYPES = ("explainer", "podcast")


@dataclass
class Settings:
    # What kind of video: changes how clips are picked and how the frame is turned vertical.
    #   explainer - a voice-over with images / B-roll
    #   podcast   - people talking on camera (follow the speaker, split screen for exchanges)
    content_type: str = "explainer"

    # Transcription
    whisper_model: str = "small"
    language: str | None = None  # None = auto-detect
    device: str = "auto"  # auto | cpu | cuda

    # Clip selection
    llm_model: str = "claude-opus-5-5"
    max_clips: int | None = None  # optional ceiling; by default the AI returns every strong moment
    min_seconds: float = 20.0
    max_seconds: float = 60.0

    # Reframing: auto picks crop or fit per shot
    layout: str = "auto"  # auto | crop | fit

    # Captions / overlays
    captions: bool = True
    uppercase_captions: bool = True
    font: str = "Arial Black"
    hook_seconds: float = 4.0  # 0 disables the on-screen hook text

    # Output
    out_width: int = 1080
    out_height: int = 1920
    max_fps: float = 30.0
    crf: int = 20
    preset: str = "fast"
