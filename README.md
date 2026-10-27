# Clipper

Turns a long narrated video (voice-over + images / B-roll) into vertical
9:16 clips for TikTok, YouTube Shorts and Reels. Each clip comes with burned-in
word-by-word captions, an on-screen hook, a title, a description and hashtags.

```
YouTube link or video file ─► transcribe ─► Claude picks clips ─► review ─► reframe 16:9→9:16 ─► captions ─► mp4 + post text
```

## Quick start: the app

Double-click **`ui.cmd`** (or run `.\ui` in this folder). Clipper opens in your browser at
http://127.0.0.1:8765. Keep the black window that appears open while you use it; closing it stops the app.

1. Paste a **YouTube link** or a **file path**, or drop a video file onto the page, and click **Make clips**.
2. Clipper downloads the video if needed, then transcribes it.
3. **Copy prompt** → **Open Claude** → paste it into a new chat → copy Claude's answer back into the box.
   (With an API key in `.env` this step happens automatically.)
4. **Review**: play each moment, untick the weak ones, edit titles, hooks, captions and hashtags.
5. **Render**, then copy each clip's caption and title straight from the results page.

Everything runs on this computer. Finished projects are listed under **Recent** on the start page.
YouTube downloads are kept in `downloads\`, so the same link isn't downloaded twice. Only clip videos
you own or have permission to reuse.

## Setup (once)

1. **Python 3.10+** and **ffmpeg** on PATH (`winget install Gyan.FFmpeg`, then open a new terminal).
2. Create the virtual environment and install packages:
   ```
   python -m venv .venv
   .venv\Scripts\python -m pip install -r requirements.txt
   ```
3. *(Optional)* For fully automatic clip picking, copy `.env.example` to `.env` and paste an
   Anthropic API key. Without one, the tool uses the free copy-paste mode described below.

The first run downloads the Whisper speech model (~500 MB for `small`).

## Picking clips: free with the Claude app, or automatic with an API key

**Free (Claude Pro / any Claude subscription):** used automatically when there's no API key, or
with `--chat`. The tool copies a prompt to your clipboard, then:

1. Open [claude.ai](https://claude.ai/new), paste (Ctrl+V) and send.
2. Click **Copy** under Claude's reply.
3. Go back to the terminal and press Enter. It reads the reply and carries on rendering.

The prompt is also saved as `claude-prompt.txt` in the output folder, and Claude's reply as
`claude-reply.txt`. If the clipboard doesn't work for you, save the reply into that file and press Enter.

**Automatic (API key):** no copy-paste, but the API is billed separately from a Claude subscription
(prepaid credits at [platform.claude.com](https://platform.claude.com), roughly $0.25 per 30-minute
video with the default model).

## Use from the command line

Everything the app does also works from a terminal in this folder (in PowerShell, put `.\` in front):

```
.\clip 'C:\videos\my-video.mp4'
.\clip 'https://www.youtube.com/watch?v=...'
```

Use single quotes in PowerShell: inside double quotes it deletes things like `$87` from file names.

`output\demo\` has two clips made from a synthetic test video, so you can see the style.

Results land in `output\<video-name>\`:

| File | What it is |
|---|---|
| `01-<title>.mp4` | the vertical clip, ready to upload |
| `01-<title>.txt` | title, caption, hashtags and YouTube title to paste in |
| `clips.json` | the chosen time ranges and text; edit it and re-render (see below) |
| `.cache\` | transcript and subtitle files, so re-runs skip transcription |

### Useful options

| Option | Effect |
|---|---|
| `--max-clips 5` | never more than 5 clips. By default there's no number: Claude includes every moment strong enough to post (could be 1, could be 15) and nothing weaker |
| `--min 15 --max 45` | clip length range in seconds (default 20–60) |
| `--dry-run` | pick clips and write `clips.json`, but don't render (fast way to preview choices) |
| `--chat` | use the free copy-paste mode even if an API key is set |
| `--clips-file output\x\clips.json` | render the clips in that file (or a saved `claude-reply.txt`) without asking again |
| `--layout fit` | always show the full frame over a blurred background (`crop` = always crop, `auto` = per shot) |
| `--no-captions` / `--hook-seconds 0` | turn off captions / the hook text |
| `--whisper-model base` | faster but less accurate transcription (`medium` / `large-v3` = slower, more accurate) |
| `--preset veryfast` | faster encoding, slightly bigger files |

Run `clip --help` for everything.

### Review, tweak, re-render

1. `clip video.mp4 --dry-run` → look at `output\<video>\clips.json`.
2. Change a `start` / `end` (seconds or `"mm:ss"`), a title, or the hashtags. Delete clips you don't want.
3. `clip video.mp4 --clips-file output\<video>\clips.json`

You can also write a `clips.json` by hand to cut exactly the moments you want:

```json
{"clips": [{"start": "1:05", "end": "1:48", "title": "Why the sea glows", "hook_text": "It makes its own light",
            "hashtags": ["ocean", "science"]}]}
```

## Explainer or podcast

Pick the kind of video on the start page (or `--type podcast` on the command line). It changes two
things: the instructions Claude gets for choosing clips, and how the picture is made vertical.
Everything else works the same.

**Podcast** (people talking on camera). For each shot the tool finds the people and works out who
is talking, by checking whose mouth moves *in time with the sound* (chewing, nodding and laughing
don't). Then:

- **One person in the shot** (a close-up camera): framed on them, captions just under the chin.
- **Two or more people, one talking** (a wide shot): cropped in on the speaker, cutting to the other
  person when they take over.
- **Quick back-and-forth, or unsure who's talking**: split screen, left person on top. Showing both
  is never wrong, so it's the fallback.
- **No faces** (B-roll, graphics): handled like an explainer shot.

Claude is told to start a clip at the question when the clip is an answer, cut straight in past
filler, and end on the payoff.

## How the explainer reframing works

There's usually no on-camera person in explainer footage, so instead of tracking a face the
tool looks at **where the visual interest is** in every shot:

1. It finds hard cuts so each shot is handled on its own.
2. On ~6 frames per second it scores each column of the image by edge detail, saliency
   (what stands out from the background) and faces.
3. Per shot it picks a layout:
   - **crop**: most of the interest fits in a 9:16 window, so the window centres on it.
     When the subject moves (or the shot dissolves into another), the window glides
     smoothly instead of jumping.
   - **fit**: the interest is spread out (text, charts, maps, wide scenery), so the whole
     frame is shown over a blurred copy of itself and nothing gets cut off.

Captions sit below the picture and the hook sits above it, so both layouts stay readable.

## Performance notes

Everything runs on the CPU. On a 4-core laptop, expect transcription at roughly 2–4× real time with
the `small` model, and rendering at about 15–25 frames per second (around 1–2 minutes per 60-second
clip). An NVIDIA GPU with CUDA 12 support makes transcription much faster (`--device cuda`).

## Project layout

```
clipper/
  web/           the browser app: app.py (HTTP API), jobs.py (background jobs), static/ (page)
  cli.py         command-line options
  pipeline.py    the stages (get video, transcribe, pick, render) used by both app and CLI
  sources.py     YouTube downloads (yt-dlp) and local paths
  progress.py    progress reporting for the terminal and the app
  transcribe.py  faster-whisper + sentence grouping
  highlights.py  Claude clip selection (API), titles, hashtags; clips.json load/save
  chat.py        free copy-paste clip selection through the Claude app
  reframe.py     cut detection, interest scoring, crop/fit plan per frame
  speakers.py    podcast mode: people per shot, who is talking, speaker / split layouts
  faces.py       face detection (OpenCV YuNet)
  captions.py    word-highlight captions + hook as .ass subtitles
  render.py      frame composition and ffmpeg encoding
  media.py       ffmpeg/ffprobe helpers
```

YouTube needs a JavaScript runtime for some downloads; yt-dlp uses Node.js or Deno if either is
installed. If downloads start failing after YouTube changes something, update the downloader:
`.venv\Scripts\python -m pip install -U "yt-dlp[default]"`.
