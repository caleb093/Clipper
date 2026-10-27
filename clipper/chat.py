"""Free clip picking: copy a prompt into the Claude app, copy the reply back.

For people with a Claude subscription but no API key. The tool writes the same
prompt the API mode would send, puts it on the clipboard, and waits for the
user to paste Claude's reply back.
"""

import subprocess
import sys
from pathlib import Path

from clipper.highlights import SYSTEM_PROMPTS, check_sentence_ranges, parse_clip_reply

CLAUDE_URL = "https://claude.ai/new"

REPLY_FORMAT = """

Answer directly in the chat (not in an artifact or file) with only the JSON below \
in one ```json code block, filled in for each clip:

{"clips": [{"start_sentence": <id>, "end_sentence": <id>, "title": "...", "hook_text": "...", \
"description": "...", "hashtags": ["...", "..."], "score": <1-100>, "reason": "..."}]}"""

# Appears only in the prompt, never in a reply: tells us the clipboard wasn't updated.
PROMPT_MARKER = "Transcript, one sentence per line"


def chat_prompt(prompt: str, content_type: str = "explainer") -> str:
    return f"{SYSTEM_PROMPTS[content_type]}\n\n{prompt}{REPLY_FORMAT}"


def pick_in_chat(prompt: str, out_dir: Path, sentence_count: int, content_type: str = "explainer") -> list[dict]:
    """Walk the user through the copy-paste round trip; return the raw clip list."""
    full_prompt = chat_prompt(prompt, content_type)
    prompt_file = out_dir / "claude-prompt.txt"
    reply_file = out_dir / "claude-reply.txt"
    prompt_file.write_text(full_prompt, encoding="utf-8")
    copied = copy_to_clipboard(full_prompt)

    where = "is on your clipboard" if copied else "couldn't be copied automatically"
    print(f"  The prompt {where} (it's also saved in {prompt_file}).")
    if not copied:
        print("  Open that file, select all (Ctrl+A) and copy (Ctrl+C).")
    print(f"  1. Open {CLAUDE_URL}, paste the prompt (Ctrl+V) and send it.")
    print("  2. When Claude has answered, click the Copy button under its reply.")
    print("  3. Come back here and press Enter.")
    print(f"  (Or save the reply as {reply_file.name} in that folder and press Enter.)")

    prompt_time = prompt_file.stat().st_mtime
    while True:
        try:
            answer = input("\n  Press Enter when you've copied Claude's reply (q to quit): ").strip().lower()
        except EOFError:  # not running in an interactive terminal
            raise RuntimeError(
                f"No terminal to wait in. Paste the prompt from {prompt_file} into Claude, save the reply as "
                f"{reply_file}, then run again with: --clips-file \"{reply_file}\""
            ) from None
        if answer == "q":
            raise KeyboardInterrupt

        candidates = []
        if reply_file.exists() and reply_file.stat().st_mtime > prompt_time:
            candidates.append((reply_file.name, reply_file.read_text(encoding="utf-8-sig")))
        clipboard = read_clipboard()
        if clipboard:
            candidates.append(("clipboard", clipboard))

        problems = []
        for source, text in candidates:
            if PROMPT_MARKER in text:
                problems.append(f"the {source} still holds the prompt, not Claude's reply")
                continue
            try:
                items = parse_clip_reply(text)
                check_sentence_ranges(items, sentence_count)
            except ValueError as exc:
                problems.append(f"the {source} doesn't look like Claude's reply: {exc}")
                continue
            reply_file.write_text(text, encoding="utf-8")  # keep it so the clips can be re-rendered later
            print(f"  Got {len(items)} clips from the {source}.")
            return items

        print("  " + ("; ".join(problems) if problems else "Nothing found on the clipboard") + ".")
        print("  Copy Claude's whole reply (the Copy button under it) and press Enter again.")


# ----------------------------------------------------------------- clipboard
# PowerShell's clipboard cmdlets handle any language's characters; UTF-8 is set
# explicitly because Windows PowerShell otherwise uses the old console code page.


def copy_to_clipboard(text: str) -> bool:
    if sys.platform != "win32":
        return False
    script = "[Console]::InputEncoding = [Text.Encoding]::UTF8; Set-Clipboard -Value ([Console]::In.ReadToEnd())"
    try:
        result = subprocess.run(_powershell(script), input=text.encode("utf-8"), capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def read_clipboard() -> str | None:
    if sys.platform != "win32":
        return None
    script = "[Console]::OutputEncoding = [Text.Encoding]::UTF8; Get-Clipboard -Raw"
    try:
        result = subprocess.run(_powershell(script), capture_output=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.decode("utf-8", errors="replace").lstrip("﻿").strip() or None


def _powershell(script: str) -> list[str]:
    return ["powershell", "-NoProfile", "-NonInteractive", "-Command", script]
