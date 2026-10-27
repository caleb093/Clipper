"""Speech-to-text with word timestamps (faster-whisper), cached to disk."""

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from clipper.progress import Reporter

SENTENCE_END = (".", "?", "!", "…")
MAX_SENTENCE_SECONDS = 20.0
PAUSE_BREAK_SECONDS = 1.2


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class Sentence:
    id: int
    start: float
    end: float
    text: str
    first_word: int  # index into the word list
    last_word: int


def transcribe(
    video: Path, cache_dir: Path, model_size: str, language: str | None, device: str, reporter: Reporter
) -> tuple[list[Word], str]:
    stat = video.stat()
    cache_file = cache_dir / f"transcript-{model_size}.json"
    fingerprint = {"file": str(video.resolve()), "size": stat.st_size, "mtime": int(stat.st_mtime)}

    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if cached.get("fingerprint") == fingerprint:
            reporter.log(f"Using the saved transcript ({len(cached['words'])} words)")
            return [Word(**w) for w in cached["words"]], cached["language"]

    if device == "auto":
        import ctranslate2

        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    if device == "cuda":
        try:
            words, lang = _run_whisper(video, model_size, language, "cuda", reporter)
        except Exception as exc:  # no usable GPU / CUDA libs: fall back quietly
            reporter.log(f"GPU transcription unavailable ({type(exc).__name__}); using CPU")
            words, lang = _run_whisper(video, model_size, language, "cpu", reporter)
    else:
        words, lang = _run_whisper(video, model_size, language, device, reporter)

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(
        json.dumps({"fingerprint": fingerprint, "language": lang, "words": [asdict(w) for w in words]}),
        encoding="utf-8",
    )
    return words, lang


def _run_whisper(
    video: Path, model_size: str, language: str | None, device: str, reporter: Reporter
) -> tuple[list[Word], str]:
    from faster_whisper import WhisperModel

    from clipper.media import load_audio

    reporter.progress(None, "loading the speech model")
    compute_type = "float16" if device == "cuda" else "int8"
    model = WhisperModel(model_size, device=device, compute_type=compute_type, cpu_threads=os.cpu_count() or 4)
    segments, info = model.transcribe(
        load_audio(video),
        language=language,
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
    )

    words: list[Word] = []
    started = time.time()
    for seg in segments:
        for w in seg.words or []:
            text = w.word.strip()
            if not text:
                continue
            end = round(max(w.end, w.start + 0.05), 3)
            # Whisper splits "1%", "it's", "$5" into pieces; a piece with no leading
            # space that starts with a non-letter belongs to the previous word.
            if words and not w.word.startswith(" ") and not text[0].isalpha():
                prev = words[-1]
                words[-1] = Word(prev.start, max(prev.end, end), prev.text + text)
            else:
                words.append(Word(round(w.start, 3), end, text))
        fraction = min(1.0, seg.end / max(info.duration, 1e-6))
        reporter.progress(fraction, f"{_mmss(seg.end)} of {_mmss(info.duration)} ({time.time() - started:.0f}s so far)")
    if not words:
        raise RuntimeError("No speech found in the video.")
    return words, info.language


def _mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def build_sentences(words: list[Word]) -> list[Sentence]:
    """Group words into sentences so clips always start and end on clean boundaries."""
    sentences: list[Sentence] = []
    first = 0
    for i, word in enumerate(words):
        is_last = i == len(words) - 1
        ends_sentence = word.text.rstrip("\"')]”’").endswith(SENTENCE_END)
        long_pause = not is_last and words[i + 1].start - word.end > PAUSE_BREAK_SECONDS
        too_long = word.end - words[first].start > MAX_SENTENCE_SECONDS
        if is_last or ends_sentence or long_pause or too_long:
            chunk = words[first : i + 1]
            sentences.append(
                Sentence(
                    id=len(sentences),
                    start=chunk[0].start,
                    end=chunk[-1].end,
                    text=" ".join(w.text for w in chunk),
                    first_word=first,
                    last_word=i,
                )
            )
            first = i + 1
    return sentences
