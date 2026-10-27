"""Progress reporting shared by the command line and the web UI."""

import sys


class Cancelled(Exception):
    """Raised inside the pipeline when the user cancels a job."""


class Reporter:
    """Receives progress from the pipeline. Subclasses decide where it goes."""

    def stage(self, key: str, label: str) -> None:
        """A new stage started. key is one of: download, transcribe, pick, review, render."""

    def progress(self, fraction: float | None, detail: str = "") -> None:
        """Progress within the current stage (0-1, or None when unknown)."""

    def log(self, message: str) -> None:
        """A line worth showing the user."""


class ConsoleReporter(Reporter):
    def __init__(self) -> None:
        self._progress_shown = False

    def stage(self, key: str, label: str) -> None:
        self._end_progress_line()
        print(f"\n{label}")

    def progress(self, fraction: float | None, detail: str = "") -> None:
        pct = f"{100 * fraction:5.1f}%  " if fraction is not None else ""
        sys.stdout.write(f"\r    {pct}{detail}".ljust(78))
        sys.stdout.flush()
        self._progress_shown = True

    def log(self, message: str) -> None:
        self._end_progress_line()
        print(f"  {message}")

    def _end_progress_line(self) -> None:
        if self._progress_shown:
            print()
            self._progress_shown = False
