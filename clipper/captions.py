"""Word-by-word captions and the hook title as an .ass subtitle file."""

from collections.abc import Callable
from pathlib import Path

from clipper.transcribe import Word

MAX_WORDS_PER_CHUNK = 3
MAX_CHARS_PER_CHUNK = 15
CHUNK_PAUSE_SECONDS = 0.5
HOLD_GAP_SECONDS = 0.7  # keep a caption up through gaps shorter than this

# Style colours are &HAABBGGRR (alpha 00 = opaque); inline \c tags take &HBBGGRR&.
WHITE = "&H00FFFFFF"
BLACK = "&H00000000"
HIGHLIGHT_TAG = r"{\c&H00FFFF&}"  # yellow
NORMAL_TAG = r"{\c&HFFFFFF&}"


def write_ass(
    path: Path,
    words: list[Word],
    clip_start: float,
    clip_end: float,
    *,
    hook_text: str,
    hook_seconds: float,
    captions: bool,
    uppercase: bool,
    font: str,
    width: int,
    height: int,
    caption_y: Callable[[float], int] = lambda seconds: 1420,
    hook_y: int = 470,
) -> None:
    """caption_y gives the caption height for a moment of the clip, and hook_y the hook's,
    both in 1080x1920 units, so captions can sit wherever the current layout leaves room."""
    s = height / 1920
    hook_y = round(hook_y * s)
    centre_x = width // 2

    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        f"Style: Caption,{font},{round(78 * s)},{WHITE},{WHITE},{BLACK},&H64000000,0,0,0,0,100,100,0,0,1,"
        f"{round(7 * s)},{round(3 * s)},5,{round(70 * s)},{round(70 * s)},0,1",
        f"Style: Hook,{font},{round(60 * s)},{BLACK},{BLACK},{WHITE},{BLACK},0,0,0,0,100,100,0,0,3,"
        f"{round(20 * s)},0,5,{round(90 * s)},{round(90 * s)},0,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    duration = clip_end - clip_start
    if hook_text and hook_seconds > 0:
        lines.append(_event(0, min(hook_seconds, duration), "Hook", f"{{\\pos({centre_x},{hook_y})}}{_escape(hook_text)}", 1))

    if captions:
        clip_words = [
            Word(w.start - clip_start, w.end - clip_start, w.text.strip())
            for w in words
            if w.start >= clip_start - 0.05 and w.end <= clip_end + 0.1 and _clean(w.text, uppercase)
        ]
        # Chunk on the raw text (punctuation marks phrase ends), display the cleaned text.
        chunks = [[Word(w.start, w.end, _clean(w.text, uppercase)) for w in c] for c in _chunk(clip_words)]
        for ci, chunk in enumerate(chunks):
            chunk_end = chunk[-1].end
            if ci + 1 < len(chunks) and chunks[ci + 1][0].start - chunk_end < HOLD_GAP_SECONDS:
                chunk_end = chunks[ci + 1][0].start
            y = round(caption_y(max(0.0, chunk[0].start)) * s)  # a phrase stays put while it's on screen
            for wi, word in enumerate(chunk):
                start = max(0.0, word.start)
                end = chunk[wi + 1].start if wi + 1 < len(chunk) else chunk_end
                if end <= start:
                    continue
                text = " ".join(
                    f"{HIGHLIGHT_TAG}{w.text}{NORMAL_TAG}" if k == wi else w.text
                    for k, w in enumerate(chunk)
                )
                lines.append(_event(start, min(end, duration), "Caption", f"{{\\pos({centre_x},{y})}}{text}", 0))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def _chunk(words: list[Word]) -> list[list[Word]]:
    chunks: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        if current:
            chars = sum(len(w.text) + 1 for w in current) + len(word.text)
            paused = word.start - current[-1].end > CHUNK_PAUSE_SECONDS
            if len(current) >= MAX_WORDS_PER_CHUNK or chars > MAX_CHARS_PER_CHUNK or paused:
                chunks.append(current)
                current = []
        current.append(word)
        if word.text[-1] in ".?!,;:":
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def _clean(text: str, uppercase: bool) -> str:
    text = _escape(text.strip()).rstrip(".,;:")
    return text.upper() if uppercase else text


def _escape(text: str) -> str:
    return text.replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def _event(start: float, end: float, style: str, text: str, layer: int) -> str:
    return f"Dialogue: {layer},{_ts(start)},{_ts(end)},{style},,0,0,0,,{text}"


def _ts(seconds: float) -> str:
    cs = int(round(max(0.0, seconds) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"
