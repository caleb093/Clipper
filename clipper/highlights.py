"""Pick the best short-form clips from a transcript, plus titles and hashtags."""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from clipper.transcribe import Sentence, Word

START_PAD = 0.12
END_PAD = 0.40


@dataclass
class Clip:
    start: float
    end: float
    title: str = ""
    hook_text: str = ""
    description: str = ""
    hashtags: list[str] = field(default_factory=list)
    score: int = 0
    reason: str = ""
    transcript: str = ""
    file: str = ""  # rendered mp4, relative to the project folder

    @property
    def duration(self) -> float:
        return self.end - self.start


SYSTEM_PROMPTS = {
    "explainer": """You are a senior short-form video editor. You cut long narrated \
explainer videos into TikTok, YouTube Shorts and Instagram Reels clips that people \
watch to the end and share.

The source videos are a voice-over narrating a topic while images and B-roll play, \
so every clip has to work on the narration alone.""",
    "podcast": """You are a senior short-form video editor. You cut long podcast and \
interview videos into TikTok, YouTube Shorts and Instagram Reels clips that people \
watch to the end and share.

The source videos are conversations between two or more people filmed on camera. \
The transcript has no speaker names, so work out from the wording where one person \
stops and another starts.""",
}

CRITERIA = {
    "explainer": """- It stands alone. Someone who hasn't seen the rest of the video understands it. \
The first sentence does not lean on earlier context ("this", "he", "as we saw", \
"so then").
- The first sentence is a hook: a surprising fact, a bold claim, a question, or \
clear stakes. Viewers decide in about two seconds whether to keep watching.
- It ends on a payoff or a finished thought, not halfway through an explanation.
- Strong material: counter-intuitive facts, mini-stories, a reveal, a list, a \
"most people don't know" moment.""",
    "podcast": """- It stands alone. Someone who hasn't seen the episode understands it. If the \
heart of the clip is an answer, start at the question (or enough of it that the answer \
makes sense). Don't start on a reply to something we didn't hear.
- It opens strong: a pointed question, a bold or contrarian opinion, the first line \
of a story, a surprising admission. Cut straight in; skip greetings, filler ("yeah", \
"so", "I mean") and warm-up.
- It ends on the payoff: the punchline, the point of the story, the strongest line, \
or the reaction to it. Don't trail off into the next topic or cut someone off mid-point.
- Strong material: hot takes, disagreements, personal stories, funny exchanges, \
practical advice, quotable lines, emotional or vulnerable moments.""",
}

INSTRUCTIONS = """Find the moments in the transcript below that are strong enough to post as \
their own short.

Video: {name}
Length: {length}
How many: there is no target. Return as many clips as there are genuinely strong \
moments, however many that is, and an empty list if nothing qualifies. Include a \
moment only if you'd be glad to post it. Don't add weaker clips to fill out the list, \
and don't leave out a strong one to keep the list short.{limit}
Clip length: {min_s:.0f}-{max_s:.0f} seconds. Work out each clip's length from the \
sentence timestamps (end of the last sentence minus start of the first).

What makes a good clip here:
{criteria}
- Clips never overlap.

For each clip give:
- start_sentence / end_sentence: the [id] of its first and last sentence (inclusive).
- title: a curiosity-driven title, at most 60 characters, that doesn't promise \
something the clip doesn't deliver.
- hook_text: at most 7 words, shown on screen during the first seconds.
- description: 1-2 sentences for the post caption, ending with a question or \
prompt that invites comments.
- hashtags: 6-10 hashtags without the # sign, lowercase, no spaces. Mix 2-3 broad \
discovery tags with specific tags for this clip's topic.
- score: 1-100, how likely it is to perform well as a short.
- reason: one sentence on why this moment works.

Write titles, hooks, descriptions and hashtags in the transcript's language ({language}).
Order the clips from best to worst.

Transcript, one sentence per line as [id] start-end (seconds) text:
{transcript}"""

CLIP_SCHEMA = {
    "type": "object",
    "properties": {
        "clips": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_sentence": {"type": "integer"},
                    "end_sentence": {"type": "integer"},
                    "title": {"type": "string"},
                    "hook_text": {"type": "string"},
                    "description": {"type": "string"},
                    "hashtags": {"type": "array", "items": {"type": "string"}},
                    "score": {"type": "integer"},
                    "reason": {"type": "string"},
                },
                "required": [
                    "start_sentence", "end_sentence", "title", "hook_text",
                    "description", "hashtags", "score", "reason",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["clips"],
    "additionalProperties": False,
}


def build_prompt(
    sentences: list[Sentence],
    *,
    video_name: str,
    video_duration: float,
    language: str,
    max_clips: int | None,
    min_s: float,
    max_s: float,
    content_type: str = "explainer",
) -> str:
    transcript = "\n".join(f"[{s.id}] {s.start:.1f}-{s.end:.1f} {s.text}" for s in sentences)
    limit = ""
    if max_clips:
        limit = (
            f" The user wants at most {max_clips}: if there are more strong moments than that, "
            f"return only the best {max_clips}. This is a ceiling, not a target."
        )
    return INSTRUCTIONS.format(
        name=video_name,
        length=_fmt_time(video_duration),
        limit=limit,
        criteria=CRITERIA[content_type],
        min_s=min_s,
        max_s=max_s,
        language=language,
        transcript=transcript,
    )


def select_clips(prompt: str, sentences: list[Sentence], words: list[Word], *, video_duration: float,
                 min_s: float, max_s: float, model: str, content_type: str = "explainer",
                 api_key: str | None = None) -> list[Clip]:
    """Pick clips through the Anthropic API.

    api_key: a user's own key for this request; without one the SDK uses the
    server's key (ANTHROPIC_API_KEY)."""
    import anthropic

    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()
    try:
        with client.beta.messages.stream(
            model=model,
            max_tokens=32000,
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            system=SYSTEM_PROMPTS[content_type],
            messages=[{"role": "user", "content": prompt}],
            output_config={
                "effort": "high",
                "format": {"type": "json_schema", "schema": CLIP_SCHEMA},
            },
        ) as stream:
            response = stream.get_final_message()
    except anthropic.AuthenticationError as exc:
        raise RuntimeError("Anthropic rejected the API key. Check it under Options, or use the Claude app instead.") from exc
    except anthropic.PermissionDeniedError as exc:
        raise RuntimeError("That API key isn't allowed to use this model, or the account has no credits left.") from exc
    except anthropic.RateLimitError as exc:
        raise RuntimeError("Anthropic's rate limit was hit. Wait a minute and try again.") from exc
    except anthropic.APIConnectionError as exc:
        raise RuntimeError("Couldn't reach Anthropic. Check the internet connection.") from exc

    if response.stop_reason == "refusal":
        raise RuntimeError("The model declined to process this transcript.")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("The model's answer was cut off; try again with a ceiling such as --max-clips 15.")
    text = next((b.text for b in response.content if b.type == "text"), None)
    if not text:
        raise RuntimeError(f"No clip list in the model's response (stop_reason={response.stop_reason}).")

    raw_clips = json.loads(text)["clips"]
    return clips_from_sentences(raw_clips, sentences, words, min_s, max_s, video_duration)


def parse_clip_reply(text: str) -> list[dict]:
    """Pull the clip list out of a reply that may wrap the JSON in prose or a ``` block."""
    text = text.strip().lstrip("﻿")
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    candidate = fenced.group(1) if fenced else text
    starts = [i for i in (candidate.find("{"), candidate.find("[")) if i >= 0]
    end = max(candidate.rfind("}"), candidate.rfind("]"))
    if not starts or end < min(starts):
        raise ValueError("there's no clip list (JSON) in it")
    try:
        data = json.loads(candidate[min(starts) : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"the clip list isn't valid JSON ({exc.msg} at line {exc.lineno})") from exc
    items = data.get("clips") if isinstance(data, dict) else data
    # An empty list is a valid answer: nothing in the video was strong enough.
    if not isinstance(items, list) or not all(isinstance(i, dict) for i in items):
        raise ValueError("it doesn't contain a list of clips")
    return items


def check_sentence_ranges(items: list[dict], sentence_count: int) -> None:
    """Make sure a reply's sentence numbers fit this video's transcript."""
    for item in items:
        try:
            first, last = int(item["start_sentence"]), int(item["end_sentence"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("a clip is missing its start_sentence / end_sentence numbers") from None
        if not 0 <= first <= last < sentence_count:
            raise ValueError(
                f"clip sentences {first}-{last} don't exist in this transcript (it has {sentence_count}); "
                "was the reply for a different video?"
            )


def clips_from_sentences(
    raw_clips: list[dict],
    sentences: list[Sentence],
    words: list[Word],
    min_s: float,
    max_s: float,
    video_duration: float,
) -> list[Clip]:
    """Turn sentence ranges into padded, non-overlapping clips within the length limits."""
    n = len(sentences)
    clips: list[Clip] = []
    taken: list[tuple[int, int]] = []

    for raw in sorted(raw_clips, key=lambda c: -int(c.get("score", 0))):
        first = max(0, min(n - 1, int(raw["start_sentence"])))
        last = max(first, min(n - 1, int(raw["end_sentence"])))

        # The model sometimes miscounts seconds: trim or extend by whole sentences.
        while last > first and sentences[last].end - sentences[first].start > max_s * 1.15:
            last -= 1
        while last + 1 < n and sentences[last].end - sentences[first].start < min_s * 0.8:
            last += 1

        if any(first <= b and last >= a for a, b in taken):
            continue
        taken.append((first, last))

        start, end = _pad(sentences, first, last, video_duration)
        clips.append(
            Clip(
                start=start,
                end=end,
                title=raw.get("title", "").strip(),
                hook_text=raw.get("hook_text", "").strip(),
                description=raw.get("description", "").strip(),
                hashtags=clean_hashtags(raw.get("hashtags", [])),
                score=int(raw.get("score", 0)),
                reason=raw.get("reason", "").strip(),
                transcript=" ".join(s.text for s in sentences[first : last + 1]),
            )
        )
    return clips


def _pad(sentences: list[Sentence], first: int, last: int, video_duration: float) -> tuple[float, float]:
    start = sentences[first].start - START_PAD
    if first > 0:
        start = max(start, sentences[first - 1].end + 0.02)
    end = sentences[last].end + END_PAD
    if last + 1 < len(sentences):
        end = min(end, sentences[last + 1].start - 0.02)
    return round(max(0.0, start), 3), round(min(video_duration, max(end, sentences[last].end)), 3)


def clean_hashtags(tags: list[str]) -> list[str]:
    seen: list[str] = []
    for tag in tags:
        t = re.sub(r"[^\w]", "", tag.lstrip("#").lower())
        if t and t not in seen:
            seen.append(t)
    return seen


# ---------------------------------------------------------------- clips file


def save_clips(path: Path, video: Path, clips: list[Clip]) -> None:
    data = {
        "source": str(video),
        "note": "Edit start/end (seconds or mm:ss), titles or hashtags, then re-run with --clips-file this-file to render without asking the AI again.",
        "clips": [asdict(c) for c in clips],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def load_clips(
    path: Path, sentences: list[Sentence], words: list[Word], video_duration: float, min_s: float, max_s: float
) -> list[Clip]:
    """Read clips.json (start/end times) or a saved Claude reply (sentence ids)."""
    try:
        items = parse_clip_reply(path.read_text(encoding="utf-8-sig"))
        by_sentence = bool(items) and "start_sentence" in items[0]
        if by_sentence:
            check_sentence_ranges(items, len(sentences))
    except ValueError as exc:
        raise ValueError(f"Couldn't read clips from {path}: {exc}") from exc
    if by_sentence:
        return clips_from_sentences(items, sentences, words, min_s, max_s, video_duration)

    clips = []
    for item in items:
        start = parse_time(item["start"])
        end = parse_time(item["end"])
        if not 0 <= start < end <= video_duration + 0.5:
            raise ValueError(f"Bad clip range {item['start']} - {item['end']} in {path}")
        spoken = [w for w in words if w.end > start and w.start < end]
        fields = {k: v for k, v in item.items() if k in Clip.__dataclass_fields__ and k not in ("start", "end")}
        clip = Clip(start=start, end=min(end, video_duration), **fields)
        clip.hashtags = clean_hashtags(clip.hashtags)
        if not clip.transcript:
            clip.transcript = " ".join(w.text for w in spoken)
        clips.append(clip)
    return clips


def parse_time(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    parts = [float(p) for p in str(value).strip().split(":")]
    seconds = 0.0
    for p in parts:
        seconds = seconds * 60 + p
    return seconds


def _fmt_time(seconds: float) -> str:
    m, s = divmod(int(round(seconds)), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"
