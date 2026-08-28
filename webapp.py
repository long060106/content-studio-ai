"""
webapp.py

A small local web UI for AI Content Studio.

    python webapp.py            # then open http://127.0.0.1:8420

What it does:
  - Paste a YouTube URL and hit Run. The server shells out to `cli.py`
    (the exact same pipeline you'd run by hand), streams its stdout, and
    turns the progress lines into a live checklist.
  - Browse everything already in ./output/ — blog post, thread, captions,
    carousel slides, short-form clip — in one place, with copy buttons and
    inline players.

Deliberately dependency-free: only the Python standard library is used, so
the UI runs with or without the project's venv activated. It binds to
127.0.0.1 only and never sends your API keys to the browser (it just reports
whether they are set).
"""

from __future__ import annotations

import base64
import hmac
import json
import mimetypes
import os
import re
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.cookies import SimpleCookie
from urllib.parse import parse_qs, unquote, urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(BASE_DIR, "ui")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

def _load_env_file() -> None:
    """Read .env into the environment, stdlib only.

    webapp.py deliberately avoids third-party imports so it runs with or
    without the venv, which rules out python-dotenv. Until now that was fine
    because nothing here needed a value from .env — env_status only reports
    whether keys exist. The sharing settings below genuinely need the values,
    and a password that silently reads as empty is an unlocked door, so the
    file gets parsed properly.

    Real environment variables win, so a shell export still overrides .env.
    """
    path = os.path.join(BASE_DIR, ".env")
    try:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except (OSError, UnicodeDecodeError):
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip("'\"")
        if name and name not in os.environ:
            os.environ[name] = value


_load_env_file()

HOST = "127.0.0.1"
PORT = int(os.environ.get("CONTENT_STUDIO_UI_PORT", "8420"))

MAX_LOG_LINES = 4000

# ---------------------------------------------------------------------------
# Sharing controls
#
# The UI is harmless on localhost, but the moment it's reachable from outside
# it hands strangers a button that spends the owner's Anthropic credits and
# deletes their work. Both guards below are off by default so local use is
# unchanged, and both switch on the moment a password is set.
# ---------------------------------------------------------------------------

# A secret carried in the link itself: ?k=<token>. The visitor types nothing
# and sees no login box — clicking the link is the whole authentication — but
# the bare URL without the key is useless to anyone who stumbles across it.
# First request exchanges the key for a cookie, so the address bar stops
# showing it and a shared screenshot doesn't leak access.
LINK_TOKEN = os.environ.get("CONTENT_STUDIO_LINK_TOKEN", "").strip()
COOKIE_NAME = "cs_key"

# An opt-in strict mode: treat *every* request as external, so the key is
# always required and nobody can delete — even the owner at the keyboard.
#
# Off by default, because it is a blunt instrument. It exists for the case
# where this is put behind some proxy whose headers are unknown; if you cannot
# say for certain that the thing in front adds forwarding headers, turn this on
# and lose the local convenience rather than gamble.
#
# It is deliberately *not* set by `static_link.ps1`. Tailscale Funnel was
# measured — it sends X-Forwarded-For, X-Forwarded-Proto and Tailscale-User-*
# — so `_is_remote` can tell a visitor from the owner on its own there.
_STRICT_PUBLIC = os.environ.get("CONTENT_STUDIO_PUBLIC", "").strip() in (
    "1", "true", "yes",
)


def _download_name(path: str) -> str:
    """A filename worth saving, rather than the one on disk.

    Every clip is stored as `short.mp4` inside a folder named for the moment,
    so downloading a batch produced `short.mp4`, `short-2.mp4`, `short-3.mp4` —
    eight files with nothing to tell them apart, on a phone, which is where
    they are least easy to sort out.

    The folder already carries the name (`03_growth_you_have_the_lock_you`), so
    the download borrows it. Generic stems only: a file that is already named
    something meaningful keeps its own name.
    """
    base = os.path.basename(path)
    stem, ext = os.path.splitext(base)
    # `short_plain` belongs here as much as `short` does, and leaving it out
    # reintroduced the exact problem this function was written to solve: every
    # plain version downloaded as `short_plain.mp4`, so saving three gave
    # `short_plain (1)`, `(2)`, `(3)` — three files with nothing to tell them
    # apart, and nothing to say which short each belonged to.
    if stem not in ("short", "short_plain", "clip_raw", "rough_cut", "speech"):
        return base

    folder = os.path.basename(os.path.dirname(path))
    if not folder:
        return base
    # Strip the leading index so the name reads as a title, and keep it to a
    # length a phone will actually display.
    name = re.sub(r"^\d+[_-]", "", folder)[:60].strip("_-") or stem
    if stem == "short_plain":
        # "-plain" rather than "-short_plain": the suffix has to say which of
        # the two renders this is, and repeating "short" says nothing.
        name = f"{name}-plain"
    elif stem != "short":
        name = f"{name}-{stem}"
    return f"{name}{ext}"

# Optional username/password, kept for anyone who prefers a real login box.
# Empty by default; the link token is the normal way in.
PASSWORD = os.environ.get("CONTENT_STUDIO_PASSWORD", "").strip()
USERNAME = os.environ.get("CONTENT_STUDIO_USER", "studio").strip()

# A run costs roughly five cents, so the default caps a shared link at about
# a dollar a day. Runs are counted rather than dollars because the count is
# something we can know exactly and enforce before spending anything.
DAILY_RUN_CAP = int(os.environ.get("CONTENT_STUDIO_DAILY_RUNS", "20") or 20)

# Deleting is irreversible and this project isn't under version control, so
# visitors arriving over a shared tunnel don't get it, while the owner sitting
# at the machine keeps it. Set CONTENT_STUDIO_ALLOW_DELETE=1 to allow it for
# everyone, or 0 to deny it to everyone including locally.
_ALLOW_DELETE_SETTING = os.environ.get("CONTENT_STUDIO_ALLOW_DELETE", "").strip().lower()
USAGE_PATH = os.path.join(OUTPUT_DIR, ".usage.json")
USAGE_LOCK = threading.Lock()


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def runs_today() -> int:
    with USAGE_LOCK:
        try:
            with open(USAGE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return 0
        return int(data.get(_today(), 0) or 0)


def record_run() -> int:
    """Count one run against today's allowance, and return the new total.

    Persisted to disk rather than held in memory so that restarting the server
    doesn't hand out a fresh allowance — otherwise the cap is one crash away
    from meaning nothing.
    """
    with USAGE_LOCK:
        try:
            with open(USAGE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            data = {}
        today = _today()
        data[today] = int(data.get(today, 0) or 0) + 1
        # Keep the file from growing forever; a fortnight is plenty of history.
        for key in sorted(data)[:-14]:
            data.pop(key, None)
        try:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            with open(USAGE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f)
        except OSError:
            pass
        return data[today]

# ---------------------------------------------------------------------------
# Pipeline steps
#
# cli.py prints one "→ ..." line per stage, then indented "✓ / ⚠" result
# lines. We map those to a fixed checklist so the UI can show progress
# without cli.py needing to know anything about the web layer.
# ---------------------------------------------------------------------------

STEPS_BY_KIND = {
    # cli.py — one video, eight formats
    "studio": [
        {"id": "transcript", "label": "Transcript"},
        {"id": "brief", "label": "Content brief"},
        {"id": "blog", "label": "Blog post"},
        {"id": "thread", "label": "X / Twitter thread"},
        {"id": "captions", "label": "Social captions"},
        {"id": "voiceover", "label": "Voice-over"},
        {"id": "carousel", "label": "Carousel slides"},
        {"id": "clip", "label": "Short-form clip"},
    ],
    # make_shorts.py — one talk, many motivational shorts
    "shorts": [
        {"id": "transcript", "label": "Transcript"},
        {"id": "moments", "label": "Finding moments"},
        {"id": "shorts", "label": "Building shorts"},
        {"id": "carousel", "label": "Quote carousel"},
    ],
}

STEPS = STEPS_BY_KIND["studio"]

_MATCHERS_BY_KIND = {
    "studio": [
        ("transcript", ("extracting transcript",)),
        ("brief", ("content brief",)),
        ("blog", ("blog post",)),
        ("thread", ("twitter",)),
        ("captions", ("generating captions",)),
        ("voiceover", ("voice-over",)),
        ("carousel", ("carousel",)),
        ("clip", ("short-form",)),
    ],
    "shorts": [
        # carousel first: "quote carousel" would otherwise match the short rule
        ("carousel", ("carousel",)),
        ("transcript", ("building shorts for", "transcript")),
        ("moments", ("strongest moments", "moments found")),
        ("shorts", ("short ",)),
    ],
}

# "→ Short 2/3: hook text" — the per-short progress line.
_SHORT_PROGRESS = re.compile(r"^Short\s+(\d+)\s*/\s*(\d+)\s*:\s*(.*)$", re.IGNORECASE)

_YOUTUBE_ID_PATTERNS = [
    r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
    r"(?:embed\/)([0-9A-Za-z_-]{11})",
    r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
]


def guess_video_id(url: str) -> str | None:
    """Same parsing rules as youtube_extractor.extract_video_id, minus the raise.

    Duplicated here on purpose: webapp.py must import cleanly without the
    project's third-party dependencies installed.
    """
    for pattern in _YOUTUBE_ID_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url.strip()):
        return url.strip()
    return None


def _step_for_line(text: str, kind: str = "studio") -> str | None:
    lowered = text.lower()
    for step_id, needles in _MATCHERS_BY_KIND.get(kind, _MATCHERS_BY_KIND["studio"]):
        if any(n in lowered for n in needles):
            return step_id
    return None


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------


_FLAGS_CACHE: dict[str, frozenset] = {}


def accepted_flags(script: str) -> frozenset:
    """The long options a script's argparse will accept.

    Read from its own `--help` rather than kept in a list here, because a list
    here is the thing that goes stale. Cached per path: this shells out, and
    the answer cannot change while the server runs.
    """
    if script in _FLAGS_CACHE:
        return _FLAGS_CACHE[script]

    found: frozenset = frozenset()
    try:
        python = os.path.join(BASE_DIR, "venv", "Scripts", "python.exe")
        if not os.path.isfile(python):
            python = sys.executable
        r = subprocess.run([python, script, "--help"], capture_output=True,
                           text=True, timeout=60, cwd=BASE_DIR)
        found = frozenset(re.findall(r"--[a-z][a-z0-9-]*", r.stdout or ""))
    except (OSError, subprocess.SubprocessError):
        # Couldn't ask. Return empty, which the filter below treats as
        # "don't know, change nothing" rather than "drop everything".
        found = frozenset()

    _FLAGS_CACHE[script] = found
    return found


def _drop_unknown_flags(cmd: list[str], script: str, log=None) -> list[str]:
    """Remove options the target script would reject, and say so.

    This exists because of a failure that is entirely invisible until someone
    presses the button: the UI builds a command line for a separate script, and
    when the two drift apart argparse exits with code 2 before the first stage
    runs. The user sees "Pipeline exited with code 2" and a usage dump, which
    describes the symptom and not the cause.

    Losing one option is a far better outcome than losing the whole run, so a
    flag the script does not recognise is dropped rather than sent. The
    mismatch is still reported into the run log, so it gets fixed rather than
    silently tolerated.
    """
    known = accepted_flags(script)
    if not known:
        return cmd

    out: list[str] = []
    dropped: list[str] = []
    i = 0
    while i < len(cmd):
        token = cmd[i]
        if token.startswith("--") and token not in known:
            dropped.append(token)
            i += 1
            # Also drop its value, if it had one.
            if i < len(cmd) and not cmd[i].startswith("--"):
                i += 1
            continue
        out.append(token)
        i += 1

    if dropped and log:
        log(f"⚠ Ignoring option(s) {' '.join(dropped)} — "
            f"{os.path.basename(script)} does not accept them. "
            f"The run continues without them; this mismatch needs fixing.")
    return out


class Job:
    def __init__(self, url: str, kind: str = "studio", options: dict | None = None):
        self.id = uuid.uuid4().hex[:12]
        self.url = url
        self.kind = kind if kind in STEPS_BY_KIND else "studio"
        self.options = options or {}
        self.video_id = guess_video_id(url)
        self.status = "running"  # running | done | error | cancelled
        self.error: str | None = None
        self.created_at = time.time()
        self.finished_at: float | None = None
        self.log: list[str] = []
        self.steps = {s["id"]: {"state": "pending", "detail": ""} for s in self.spec}
        self.current: str | None = None
        self.step_started: float = time.time()
        # Stages that hit a ⚠ at some point. A stage covering several items
        # (each short) keeps running afterwards, so the warning has to be
        # remembered or the finished checklist reads as all-clear.
        self.warned: set[str] = set()
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    @property
    def spec(self) -> list[dict]:
        return STEPS_BY_KIND[self.kind]

    # -- state as seen by the browser ---------------------------------------

    def snapshot(self, log_from: int = 0) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "url": self.url,
                "kind": self.kind,
                "options": self.options,
                "video_id": self.video_id,
                "status": self.status,
                "error": self.error,
                "created_at": self.created_at,
                "finished_at": self.finished_at,
                "elapsed": (self.finished_at or time.time()) - self.created_at,
                # step_elapsed lets the UI show how long the current stage has
                # been going. The clip stage is slow and near-silent, so
                # without it a healthy run looks frozen.
                "step_elapsed": time.time() - self.step_started,
                "steps": [
                    {"id": s["id"], "label": s["label"], **self.steps[s["id"]]}
                    for s in self.spec
                ],
                "log": self.log[log_from:],
                "log_total": len(self.log),
            }

    # -- running ------------------------------------------------------------

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def cancel(self) -> bool:
        """Stop the run — the whole process tree, not just the parent.

        The pipeline's actual work happens in child processes: yt-dlp, ffmpeg
        and Whisper. Terminating only the Python parent leaves those orphaned
        and still running, so a "stopped" encode carries on burning CPU and
        writing files. On Windows `taskkill /T` walks the tree; elsewhere the
        process group gets the signal.
        """
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                return False

        if os.name == "nt":
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    timeout=30,
                )
                return True
            except (OSError, subprocess.TimeoutExpired):
                pass  # fall through to the plain terminate below
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                return True
            except (OSError, AttributeError):
                pass

        proc.terminate()
        return True

    def _python_exe(self) -> str:
        """Prefer the project's venv interpreter — that's where the deps live."""
        candidates = [
            os.path.join(BASE_DIR, "venv", "Scripts", "python.exe"),
            os.path.join(BASE_DIR, "venv", "bin", "python"),
        ]
        for path in candidates:
            if os.path.isfile(path):
                return path
        return sys.executable

    def _command(self) -> list[str]:
        python = self._python_exe()
        if self.kind == "shorts":
            cmd = [
                python, "-u", "make_shorts.py", self.url,
                "--style", str(self.options.get("style", "broll")),
            ]
            count = self.options.get("count")
            if count:
                cmd += ["--count", str(int(count))]
            # The carousel is on by default in the pipeline, so the flag to
            # send is the one that turns it *off*. Sending "--carousel"
            # instead is not a no-op — argparse rejects the unknown option and
            # the run dies at two seconds, before the first stage.
            if not self.options.get("carousel"):
                cmd.append("--no-carousel")
            return _drop_unknown_flags(cmd, self._script_path("make_shorts.py"),
                                       self._append)
        return [python, "-u", "cli.py", self.url]

    def _script_path(self, name: str) -> str:
        return os.path.join(BASE_DIR, name)

    def _run(self) -> None:
        env = dict(os.environ)
        # cli.py prints →/✓/⚠; without this the child crashes writing them to
        # a pipe on Windows (cp1252 can't encode them).
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        cmd = self._command()
        self._append(f"$ {' '.join(cmd)}")

        # The pipeline's imports (anthropic, yt-dlp, torch, Pillow…) take ~10s
        # before anything is printed, so show the first step as busy right away.
        with self._lock:
            self.current = "transcript"
            self.steps["transcript"]["state"] = "running"
            self.steps["transcript"]["detail"] = "Starting the pipeline…"

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=BASE_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                # Own process group on POSIX so cancelling reaches ffmpeg and
                # yt-dlp, not just the Python parent.
                start_new_session=(os.name != "nt"),
            )
        except OSError as e:
            with self._lock:
                self.status = "error"
                self.error = f"Could not start the pipeline: {e}"
                self.finished_at = time.time()
            return

        with self._lock:
            self._proc = proc

        assert proc.stdout is not None
        for raw in proc.stdout:
            self._consume(raw.rstrip("\n"))

        code = proc.wait()
        with self._lock:
            self.finished_at = time.time()
            if self.status == "cancelled":
                for state in self.steps.values():
                    if state["state"] == "running":
                        state["state"] = "skipped"
                        state["detail"] = "Cancelled"
            elif code == 0:
                self.status = "done"
                for step_id, state in self.steps.items():
                    if state["state"] == "running":
                        self._finish(step_id)
            else:
                self.status = "error"
                if not self.error:
                    self.error = f"Pipeline exited with code {code}."
                for state in self.steps.values():
                    if state["state"] == "running":
                        state["state"] = "error"

    def _finish(self, step_id: str) -> None:
        """Close a stage, downgrading to 'warn' if anything went wrong in it.

        Caller holds the lock.
        """
        state = self.steps[step_id]
        if state["state"] in ("error", "skipped"):
            return
        state["state"] = "warn" if step_id in self.warned else "done"

    def _append(self, line: str) -> None:
        with self._lock:
            self.log.append(line)
            if len(self.log) > MAX_LOG_LINES:
                del self.log[: len(self.log) - MAX_LOG_LINES]

    def _consume(self, line: str) -> None:
        self._append(line)
        s = line.strip()
        if not s:
            return

        with self._lock:
            if s.startswith("·") and self.current:
                # Progress chatter inside a stage — updates the detail line
                # without changing the stage's state.
                self.steps[self.current]["detail"] = s[1:].strip()
            elif s.startswith("→"):
                body = s[1:].strip()

                # "Short 2/3: hook" keeps one step busy and counts through it,
                # rather than flipping between stages for every short.
                match = _SHORT_PROGRESS.match(body)
                if match and "shorts" in self.steps:
                    done_before, total, hook = match.groups()
                    for other_id, state in self.steps.items():
                        if other_id != "shorts" and state["state"] == "running":
                            self._finish(other_id)
                    self.current = "shorts"
                    self.step_started = time.time()
                    self.steps["shorts"]["state"] = "running"
                    self.steps["shorts"]["detail"] = f"{done_before}/{total} — {hook}"
                    return

                step_id = _step_for_line(body, self.kind)
                if step_id:
                    # Anything still marked running before this belongs to a
                    # finished stage.
                    for other_id, state in self.steps.items():
                        if other_id != step_id and state["state"] == "running":
                            self._finish(other_id)
                    self.current = step_id
                    self.step_started = time.time()
                    state = self.steps[step_id]
                    if "skipping" in body.lower():
                        state["state"] = "skipped"
                        state["detail"] = body
                    elif state["state"] not in ("warn", "error"):
                        state["state"] = "running"
                        state["detail"] = body
            elif s.startswith("✓ Done"):
                for step_id, state in self.steps.items():
                    if state["state"] == "running":
                        self._finish(step_id)
            elif s.startswith("✓") and self.current:
                state = self.steps[self.current]
                # The shorts stage emits a ✓ per clip, per caption pass and per
                # render, so it stays busy until the run moves on or ends.
                if self.current != "shorts" and state["state"] not in ("warn", "error"):
                    state["state"] = "done"
                state["detail"] = s[1:].strip()
            elif s.startswith("⚠") and self.current:
                state = self.steps[self.current]
                state["state"] = "warn"
                state["detail"] = s[1:].strip()
                self.warned.add(self.current)
            elif s.startswith("✗"):
                self.error = s[1:].strip()
                if self.current:
                    self.steps[self.current]["state"] = "error"
                    self.steps[self.current]["detail"] = s[1:].strip()

            # cli.py prints the saved file paths at the end; that's the most
            # reliable place to learn the real video_id for odd URLs.
            if not self.video_id:
                m = re.search(r"output[\\/]([0-9A-Za-z_-]{11})[\\/]", s)
                if m:
                    self.video_id = m.group(1)


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()


def active_job() -> Job | None:
    with JOBS_LOCK:
        for job in JOBS.values():
            if job.status == "running":
                return job
    return None


# ---------------------------------------------------------------------------
# Reading the output library
# ---------------------------------------------------------------------------


def _safe_video_dir(video_id: str) -> str | None:
    if not re.fullmatch(r"[0-9A-Za-z_-]{1,32}", video_id):
        return None
    path = os.path.join(OUTPUT_DIR, video_id)
    if not os.path.isdir(path):
        return None
    return path


def _read_text(path: str) -> str | None:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except (OSError, UnicodeDecodeError):
        return None


def _read_json(path: str):
    text = _read_text(path)
    if text is None:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# Everything that's worth reclaiming. The generated text and JSON is a few KB
# per talk; the video is essentially all of it.
MEDIA_EXTS = {".mp4", ".mp3", ".wav", ".m4a", ".webm", ".mkv", ".mov"}


def _folder_bytes(folder: str) -> tuple[int, int]:
    """(total bytes, media bytes) for everything under a folder."""
    total = media = 0
    for root, _dirs, names in os.walk(folder):
        for name in names:
            try:
                size = os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
            total += size
            if os.path.splitext(name)[1].lower() in MEDIA_EXTS:
                media += size
    return total, media


def _safe_video_dir(video_id: str) -> str | None:
    """Resolve a library id to a real folder inside OUTPUT_DIR, or None.

    Deletion is the one operation here that can destroy work, and the id
    arrives from an HTTP path, so it gets checked properly rather than
    trusted: reject anything with a separator or a parent reference outright,
    then confirm the *resolved* path is genuinely inside the resolved output
    directory. The realpath comparison is what catches a symlink pointing
    somewhere else entirely, which the string checks alone would miss.
    """
    if not video_id or video_id in (".", ".."):
        return None
    if "/" in video_id or "\\" in video_id or os.path.isabs(video_id):
        return None

    candidate = os.path.realpath(os.path.join(OUTPUT_DIR, video_id))
    root = os.path.realpath(OUTPUT_DIR)
    if candidate != root and not candidate.startswith(root + os.sep):
        return None
    if not os.path.isdir(candidate):
        return None

    # Same rule library_index() uses, so the only deletable things are the
    # per-video folders the UI actually lists.
    if not re.fullmatch(r"[0-9A-Za-z_-]{11}", video_id) and not os.path.exists(
        os.path.join(candidate, "transcript.json")
    ):
        return None
    return candidate


def _force_rmtree(path: str) -> bool:
    """Delete a tree, clearing read-only flags that would otherwise stop it.

    Necessary because this project usually lives inside OneDrive, and OneDrive
    marks synced folders ReadOnly with a reparse point (Files On-Demand).
    `shutil.rmtree` fails outright on those with "Access is denied", even
    though nothing is holding the files open — which is why deletes were
    leaving empty husks behind while `Remove-Item -Force` cleared them fine.
    Clearing the attribute and retrying is what -Force does.

    Returns True if the tree is gone.
    """
    def on_error(func, target, _exc):
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except OSError:
            pass

    shutil.rmtree(path, onerror=on_error)
    return not os.path.exists(path)


def sweep_deleting() -> None:
    """Clear any `*.deleting` husks left by an earlier delete.

    Removing a renamed folder can fail on the directories themselves even when
    every file inside has gone — a sync client or indexer holding a handle is
    enough. The entry has already left the library at that point, so the husk
    is invisible clutter rather than a problem, but it would accumulate one
    folder per delete forever. Retrying on the next pass clears them once
    whatever held the handle has let go.
    """
    if not os.path.isdir(OUTPUT_DIR):
        return
    for name in os.listdir(OUTPUT_DIR):
        if not name.endswith(".deleting"):
            continue
        path = os.path.join(OUTPUT_DIR, name)
        if os.path.isdir(path):
            _force_rmtree(path)


def delete_library_item(video_id: str) -> dict:
    """Remove a library entry: its folder and everything in it."""
    sweep_deleting()

    folder = _safe_video_dir(video_id)
    if folder is None:
        return {"error": "Unknown or unsafe library id", "status": 404}

    # A run writing into this folder would recreate half of what we remove,
    # and leave the job failing on missing files partway through.
    with JOBS_LOCK:
        busy = any(
            job.status == "running" and getattr(job, "video_id", None) == video_id
            for job in JOBS.values()
        )
    if busy:
        return {"error": "A run is using this talk right now", "status": 409}

    before_total, _media = _folder_bytes(folder)

    # Rename first, delete second. rmtree removes files before directories, so
    # a failure partway through leaves the folder gutted while still reporting
    # an error — the worst outcome, because the caller thinks nothing happened.
    # A rename is a single operation that either works or doesn't: if anything
    # in here is locked, it fails with everything still intact.
    staged = folder + ".deleting"
    try:
        if os.path.exists(staged):
            _force_rmtree(staged)
        os.rename(folder, staged)
    except OSError as e:
        return {
            "error": f"Couldn't delete this talk — something is using its files ({e.strerror or e})",
            "status": 409,
        }

    _force_rmtree(staged)
    if os.path.isdir(staged):
        # The contents are gone and the entry has left the library either way;
        # only an empty directory skeleton remains, which is harmless and gets
        # cleared on the next attempt.
        return {"ok": True, "video_id": video_id, "freed_bytes": before_total,
                "note": "Some empty folders could not be removed and were left behind."}
    return {"ok": True, "video_id": video_id, "freed_bytes": before_total}


def library_index() -> list[dict]:
    if not os.path.isdir(OUTPUT_DIR):
        return []

    # Cheap, and it means husks disappear on their own as soon as whatever was
    # holding them releases — without the user ever having to know they existed.
    sweep_deleting()

    items = []
    for name in sorted(os.listdir(OUTPUT_DIR)):
        video_dir = os.path.join(OUTPUT_DIR, name)
        if not os.path.isdir(video_dir):
            continue
        # output/ also holds non-video working folders (posters/, projects/).
        # A library entry is a per-video folder: an 11-character YouTube id,
        # or anything that actually produced a transcript.
        if not re.fullmatch(r"[0-9A-Za-z_-]{11}", name) and not os.path.exists(
            os.path.join(video_dir, "transcript.json")
        ):
            continue

        transcript = _read_json(os.path.join(video_dir, "transcript.json")) or {}
        brief = _read_json(os.path.join(video_dir, "brief.json")) or {}
        carousel_dir = os.path.join(video_dir, "carousel")

        def has(*parts: str) -> bool:
            return os.path.exists(os.path.join(video_dir, *parts))

        sizes = _folder_bytes(video_dir)

        title = transcript.get("title") or (brief.get("title_suggestions") or [name])[0]
        items.append(
            {
                "video_id": name,
                "title": title,
                "channel": transcript.get("channel", ""),
                "url": transcript.get("url", f"https://www.youtube.com/watch?v={name}"),
                "duration_seconds": transcript.get("duration_seconds", 0),
                "summary": brief.get("summary", ""),
                "topics": brief.get("topics", []),
                "modified": os.path.getmtime(video_dir),
                "size_bytes": sizes[0],
                "media_bytes": sizes[1],
                "assets": {
                    "brief": has("brief.json"),
                    "blog": has("blog_post.md"),
                    "thread": has("twitter_thread.json"),
                    "captions": has("captions.json"),
                    "carousel": os.path.isdir(carousel_dir)
                    and any(f.endswith(".png") for f in os.listdir(carousel_dir)),
                    "voiceover": has("voiceover.mp3") or has("voiceover_script.txt"),
                    "clip": has("short_form_clip.mp4"),
                    "transcript": has("transcript.json"),
                    "motivational": bool(read_shorts(video_dir)["shorts"]),
                },
            }
        )

    items.sort(key=lambda i: i["modified"], reverse=True)
    return items


def read_shorts(video_dir: str) -> dict:
    """Whatever the motivational-shorts pipeline left in this video's folder."""
    shorts_dir = os.path.join(video_dir, "shorts")
    if not os.path.isdir(shorts_dir):
        return {"shorts": [], "carousel": []}

    index = _read_json(os.path.join(shorts_dir, "index.json")) or {}
    records = []

    # Trust the folders on disk over index.json, so a run that was interrupted
    # before writing the index still shows whatever it managed to render.
    for name in sorted(os.listdir(shorts_dir)):
        folder = os.path.join(shorts_dir, name)
        if not os.path.isdir(folder) or name == "carousel":
            continue
        video_path = os.path.join(folder, "short.mp4")
        if not os.path.isfile(video_path):
            continue
        moment = _read_json(os.path.join(folder, "moment.json")) or {}
        moment["folder_name"] = name
        moment["media"] = os.path.relpath(video_path, OUTPUT_DIR).replace("\\", "/")
        # The plain version, when the pipeline wrote one. Optional rather than
        # required: shorts rendered before it existed have no plain file, and
        # they should still appear rather than vanishing from the library.
        plain_path = os.path.join(folder, "short_plain.mp4")
        moment["media_plain"] = (
            os.path.relpath(plain_path, OUTPUT_DIR).replace("\\", "/")
            if os.path.isfile(plain_path) else None
        )
        # The editing brief travels with the clip: it is the part that says
        # what must not be trimmed, and it is useless if you have to go
        # looking for it in a folder.
        moment["brief"] = _read_text(os.path.join(folder, "brief.md")) or ""
        tags = _read_json(os.path.join(folder, "hashtags.json")) or {}
        moment["hashtags"] = tags.get("hashtags", [])
        records.append(moment)

    # The carousel is words now, not pictures — the cards get designed by hand
    # afterwards, so what's wanted here is the copy to paste, not an image.
    carousel_dir = os.path.join(shorts_dir, "carousel")
    carousel_text = _read_text(os.path.join(carousel_dir, "carousel.txt")) or ""
    carousel_cards = (_read_json(os.path.join(carousel_dir, "carousel.json")) or {})

    return {
        "shorts": records,
        "carousel": carousel_text,
        "carousel_cards": carousel_cards.get("cards", carousel_cards.get("slides", [])),
        "source_url": index.get("source_url", ""),
    }


def asset_status() -> dict:
    """Library counts plus whether a stock key is configured."""
    manifest = _read_json(os.path.join(BASE_DIR, "assets", "library.json")) or {}
    assets = manifest.get("assets", [])
    counts = {kind: sum(1 for a in assets if a.get("kind") == kind)
              for kind in ("video", "image", "music")}

    # The manifest covers assets/image, assets/video and assets/music — it has
    # never covered assets/broll, which `curated_broll` walks directly. That
    # went unnoticed while the stock cache existed and padded the video count.
    # With the cache deleted the count read zero next to a library of nearly
    # two hundred clips, which is worse than no number at all: it says the
    # pipeline has nothing to cut to, while every cutaway in the last run came
    # from exactly that folder.
    broll_dir = os.path.join(BASE_DIR, "assets", "broll")
    if os.path.isdir(broll_dir):
        counts["video"] += sum(
            1
            for _root, _dirs, files in os.walk(broll_dir)
            for f in files
            if f.lower().endswith((".mp4", ".mov", ".m4v", ".webm"))
        )

    keys = {"PEXELS_API_KEY": False, "PIXABAY_API_KEY": False}
    env_text = _read_text(os.path.join(BASE_DIR, ".env")) or ""
    for name in keys:
        value = os.environ.get(name, "").strip()
        if not value:
            for line in env_text.splitlines():
                line = line.strip()
                if line.startswith(f"{name}=") and not line.startswith("#"):
                    value = line.split("=", 1)[1].strip().strip("'\"")
                    break
        keys[name] = bool(value) and value != "your_key_here"

    return {
        "counts": counts,
        "stock": keys["PEXELS_API_KEY"] or keys["PIXABAY_API_KEY"],
        "pexels": keys["PEXELS_API_KEY"],
        "pixabay": keys["PIXABAY_API_KEY"],
    }


def library_detail(video_id: str) -> dict | None:
    video_dir = _safe_video_dir(video_id)
    if video_dir is None:
        return None

    transcript = _read_json(os.path.join(video_dir, "transcript.json")) or {}
    brief = _read_json(os.path.join(video_dir, "brief.json"))
    thread = _read_json(os.path.join(video_dir, "twitter_thread.json"))
    captions = _read_json(os.path.join(video_dir, "captions.json"))
    carousel = _read_json(os.path.join(video_dir, "carousel", "carousel.json"))
    clip_info = _read_json(os.path.join(video_dir, "clip_info.json"))

    carousel_dir = os.path.join(video_dir, "carousel")
    slide_files = []
    if os.path.isdir(carousel_dir):
        slide_files = sorted(f for f in os.listdir(carousel_dir) if f.endswith(".png"))

    segments = transcript.get("transcript_segments") or []
    transcript_text = transcript.get("transcript_text", "")

    return {
        "video_id": video_id,
        "title": transcript.get("title")
        or (brief or {}).get("title_suggestions", [video_id])[0],
        "channel": transcript.get("channel", ""),
        "url": transcript.get("url", f"https://www.youtube.com/watch?v={video_id}"),
        "duration_seconds": transcript.get("duration_seconds", 0),
        "modified": os.path.getmtime(video_dir),
        "folder": video_dir,
        "brief": brief,
        "blog": _read_text(os.path.join(video_dir, "blog_post.md")),
        "thread": thread,
        "captions": captions,
        "carousel": {"slides": (carousel or {}).get("slides", []), "images": slide_files},
        "voiceover": {
            "script": _read_text(os.path.join(video_dir, "voiceover_script.txt")),
            "audio": os.path.exists(os.path.join(video_dir, "voiceover.mp3")),
        },
        "clip": {
            "info": clip_info,
            "video": os.path.exists(os.path.join(video_dir, "short_form_clip.mp4")),
        },
        "motivational": read_shorts(video_dir),
        "transcript": {
            "language": transcript.get("transcript_language", ""),
            "word_count": len(transcript_text.split()),
            "segment_count": len(segments),
            "text": transcript_text,
        },
    }


def env_status() -> dict:
    """Report which keys are configured — never the values themselves."""
    keys = {"ANTHROPIC_API_KEY": False, "ELEVENLABS_API_KEY": False}
    for key in keys:
        value = os.environ.get(key, "").strip()
        keys[key] = bool(value) and value != "your_key_here"

    env_path = os.path.join(BASE_DIR, ".env")
    text = _read_text(env_path)
    if text:
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, _, value = line.partition("=")
            name, value = name.strip(), value.strip().strip("'\"")
            if name in keys and value and value not in ("your_key_here", "your_elevenlabs_key_here"):
                keys[name] = True

    return {
        "anthropic": keys["ANTHROPIC_API_KEY"],
        "elevenlabs": keys["ELEVENLABS_API_KEY"],
        "env_file": os.path.exists(env_path),
    }


# ---------------------------------------------------------------------------
# HTTP layer
# ---------------------------------------------------------------------------


def _matches(supplied: str, secret: str) -> bool:
    """Constant-time compare that tolerates a missing value."""
    if not supplied or not secret:
        return False
    return hmac.compare_digest(supplied, secret)


class _Auth:
    """How a request proved it was allowed in."""
    NONE = "none"
    OPEN = "open"        # nothing configured; localhost use
    COOKIE = "cookie"
    LINK = "link"        # ?k=... — needs a cookie set on the way out
    BASIC = "basic"


class Handler(BaseHTTPRequestHandler):
    server_version = "ContentStudioUI"

    def log_message(self, fmt, *args):  # quieter console
        pass

    # -- access -------------------------------------------------------------

    def _how_authorised(self) -> str:
        """Which credential let this request through, if any."""
        if not LINK_TOKEN and not PASSWORD:
            return _Auth.OPEN

        # The owner at the keyboard doesn't need the key. The server binds to
        # 127.0.0.1, so a request without Cloudflare's headers came from this
        # machine — and anyone sitting here can already read .env and the whole
        # output folder, so a key would guard nothing.
        #
        # cloudflared also connects from 127.0.0.1, which is why the address
        # can't be the test. Cf-Ray is added on Cloudflare's side and a visitor
        # has no way to strip it, so its presence reliably marks tunnel traffic.
        if not self._is_remote():
            return _Auth.OPEN

        if LINK_TOKEN:
            cookie = SimpleCookie(self.headers.get("Cookie", ""))
            if COOKIE_NAME in cookie and _matches(cookie[COOKIE_NAME].value, LINK_TOKEN):
                return _Auth.COOKIE
            supplied = (parse_qs(urlparse(self.path).query).get("k") or [""])[0]
            if _matches(supplied, LINK_TOKEN):
                return _Auth.LINK
            # /s/<key> as well as ?k=<key>. A path survives sharing better than
            # a query string: a trailing "?k=..." is easy to lose when a long
            # URL wraps onto two lines, and some apps trim query parameters
            # when they generate a link preview.
            here = unquote(urlparse(self.path).path)
            if here.startswith("/s/") and _matches(here[3:].strip("/"), LINK_TOKEN):
                return _Auth.LINK

        if PASSWORD and self._basic_ok():
            return _Auth.BASIC
        return _Auth.NONE

    def _authorised(self) -> bool:
        return self._how_authorised() != _Auth.NONE

    def _basic_ok(self) -> bool:
        header = self.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return False
        try:
            decoded = base64.b64decode(header[6:]).decode("utf-8", "replace")
            user, _, password = decoded.partition(":")
        except Exception:
            return False
        return _matches(user, USERNAME) and _matches(password, PASSWORD)

    def _is_remote(self) -> bool:
        """True when this request arrived through the Cloudflare tunnel.

        Cloudflare stamps every proxied request with Cf-Ray, and a visitor
        cannot strip it — it's added on Cloudflare's side, not the client's.
        Requests made directly to localhost carry none of these, so this
        cleanly separates "the owner at the keyboard" from "someone I sent a
        link to" without needing separate accounts.

        Not a security boundary on its own: someone already on this machine
        could forge the header, but they have full access regardless. It exists
        to stop a guest deleting the owner's work.
        """
        # The stakes are worth stating: when this returns False,
        # `_how_authorised` returns OPEN — no key at all. Getting it wrong for a
        # publicly reachable server means the studio answers the whole internet.
        # So the header list below must cover whatever proxy is actually in
        # front, and `_STRICT_PUBLIC` exists for when that cannot be verified.
        if _STRICT_PUBLIC:
            return True
        return any(
            self.headers.get(h)
            for h in (
                # Cloudflare quick tunnel (share_link.ps1)
                "Cf-Ray", "Cf-Connecting-Ip",
                # Tailscale Funnel (static_link.ps1). Confirmed by echoing a
                # real request back: X-Forwarded-For and X-Forwarded-Proto
                # arrive on every proxied request, and Tailscale-User-* as well
                # when the visitor is signed in to the tailnet.
                "Tailscale-Headers-Info", "Tailscale-User-Login",
                # Common to proxies generally.
                "X-Forwarded-For", "X-Forwarded-Proto", "X-Forwarded-Host",
                "X-Real-Ip", "Forwarded",
            )
        )

    def _may_delete(self) -> bool:
        if _ALLOW_DELETE_SETTING in ("1", "true", "yes"):
            return True
        if _ALLOW_DELETE_SETTING in ("0", "false", "no"):
            return False
        return not self._is_remote()

    def _demand_login(self) -> None:
        body = b'{"error": "Sign in required"}'
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="Content Studio", charset="UTF-8"')
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # -- helpers ------------------------------------------------------------

    def _auth_cookie(self) -> None:
        """Hand back the link key as a cookie, once, on the first request."""
        if not getattr(self, "_set_cookie", False):
            return
        self.send_header(
            "Set-Cookie",
            f"{COOKIE_NAME}={LINK_TOKEN}; Path=/; Max-Age=2592000; "
            "HttpOnly; SameSite=Lax",
        )
        self._set_cookie = False

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self._auth_cookie()
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: str, download: bool = False) -> None:
        if not os.path.isfile(path):
            self._send_json({"error": "Not found"}, 404)
            return

        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        size = os.path.getsize(path)
        mtime = int(os.path.getmtime(path))
        start, end = 0, size - 1
        status = 200

        # Everything here is served without cache headers otherwise, which
        # means browsers cache it on a heuristic and keep doing so. That is how
        # a UI change shipped and neither the laptop nor the phone showed it:
        # the server was serving the new app.js and both browsers were still
        # running the old one. It bites hardest on the phone, where there is no
        # convenient hard-refresh.
        #
        # `no-cache` is not "do not cache" — it means revalidate before use, so
        # an unchanged file still costs one small 304 rather than a re-download.
        # That is what makes it safe to apply to the videos too, which matters
        # because rebuilding a talk overwrites short.mp4 in place and a cached
        # copy would quietly keep showing the previous cut.
        etag = f'"{size:x}-{mtime:x}"'
        if self.headers.get("If-None-Match") == etag:
            self.send_response(304)
            self._auth_cookie()
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            return

        # Range support so the <video>/<audio> players can seek.
        range_header = self.headers.get("Range", "")
        m = re.match(r"bytes=(\d*)-(\d*)", range_header)
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                if m.group(2):
                    end = min(int(m.group(2)), size - 1)
            else:  # suffix range: last N bytes
                start = max(size - int(m.group(2)), 0)
            if start > end or start >= size:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = 206

        length = end - start + 1
        self.send_response(status)
        self._auth_cookie()
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "no-cache")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="{_download_name(path)}"',
            )
        self.end_headers()

        with open(path, "rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)

    def _read_body_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return {}
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    # -- routes -------------------------------------------------------------

    def do_GET(self):  # noqa: N802
        mode = self._how_authorised()
        if mode == _Auth.NONE:
            self._demand_login()
            return
        # Arrived with ?k=... — swap it for a cookie so the key stops riding
        # in the URL bar and survives navigation within the app.
        self._set_cookie = (mode == _Auth.LINK)

        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        # Trade the key for a cookie, then send them on to a clean address, so
        # the key stops showing in the URL bar — a screenshot of the studio
        # shouldn't hand out access to it.
        if path.startswith("/s/"):
            self.send_response(302)
            self._auth_cookie()
            self.send_header("Location", "/")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if path in ("/", "/index.html"):
            self._send_file(os.path.join(UI_DIR, "index.html"))
            return

        if path.startswith("/ui/"):
            rel = path[len("/ui/") :]
            target = os.path.normpath(os.path.join(UI_DIR, rel))
            if not target.startswith(UI_DIR):
                self._send_json({"error": "Forbidden"}, 403)
                return
            self._send_file(target)
            return

        if path == "/api/state":
            with JOBS_LOCK:
                jobs = [j.snapshot(log_from=10 ** 9) for j in JOBS.values()]
            self._send_json(
                {
                    "env": env_status(),
                    "assets": asset_status(),
                    "library": library_index(),
                    "jobs": sorted(jobs, key=lambda j: j["created_at"], reverse=True),
                    "steps": STEPS_BY_KIND,
                }
            )
            return

        if path == "/api/assets":
            self._send_json(asset_status())
            return

            return

        if path == "/api/library":
            self._send_json({"library": library_index()})
            return

        if path.startswith("/api/library/"):
            detail = library_detail(path[len("/api/library/") :])
            if detail is None:
                self._send_json({"error": "Unknown video id"}, 404)
                return
            self._send_json(detail)
            return

        if path.startswith("/api/jobs/"):
            job_id = path[len("/api/jobs/") :]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if job is None:
                self._send_json({"error": "Unknown job"}, 404)
                return
            try:
                log_from = int(dict(
                    p.split("=", 1) for p in parsed.query.split("&") if "=" in p
                ).get("log_from", "0"))
            except ValueError:
                log_from = 0
            self._send_json(job.snapshot(log_from=max(log_from, 0)))
            return

        if path.startswith("/media/"):
            rel = path[len("/media/") :]
            parts = [p for p in rel.split("/") if p not in ("", ".", "..")]
            if not parts:
                self._send_json({"error": "Not found"}, 404)
                return
            target = os.path.normpath(os.path.join(OUTPUT_DIR, *parts))
            if not target.startswith(OUTPUT_DIR):
                self._send_json({"error": "Forbidden"}, 403)
                return
            self._send_file(target, download="download" in parsed.query)
            return

        self._send_json({"error": "Not found"}, 404)

    def do_DELETE(self):  # noqa: N802
        mode = self._how_authorised()
        if mode == _Auth.NONE:
            self._demand_login()
            return
        # Arrived with ?k=... — swap it for a cookie so the key stops riding
        # in the URL bar and survives navigation within the app.
        self._set_cookie = (mode == _Auth.LINK)

        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path.startswith("/api/library/"):
            if not self._may_delete():
                self._send_json(
                    {"error": "Deleting is disabled on this link. The owner can "
                              "enable it with CONTENT_STUDIO_ALLOW_DELETE=1."},
                    403,
                )
                return
            video_id = path[len("/api/library/") :]
            result = delete_library_item(video_id)
            status = result.pop("status", 200 if result.get("ok") else 400)
            self._send_json(result, status)
            return

        self._send_json({"error": "Not found"}, 404)

    def do_POST(self):  # noqa: N802
        mode = self._how_authorised()
        if mode == _Auth.NONE:
            self._demand_login()
            return
        # Arrived with ?k=... — swap it for a cookie so the key stops riding
        # in the URL bar and survives navigation within the app.
        self._set_cookie = (mode == _Auth.LINK)

        path = unquote(urlparse(self.path).path)

        if path == "/api/jobs":
            body = self._read_body_json()
            url = (body.get("url") or "").strip()
            if not url:
                self._send_json({"error": "Paste a YouTube URL first."}, 400)
                return
            if guess_video_id(url) is None:
                self._send_json(
                    {"error": "That doesn't look like a YouTube URL or video ID."}, 400
                )
                return
            running = active_job()
            if running is not None:
                self._send_json(
                    {
                        "error": "A run is already in progress.",
                        "job_id": running.id,
                    },
                    409,
                )
                return
            if not env_status()["anthropic"]:
                self._send_json(
                    {
                        "error": "ANTHROPIC_API_KEY isn't set. Copy .env.example to "
                        ".env, add your key, then restart this server."
                    },
                    400,
                )
                return
            # Checked before the job is created, so hitting the cap costs
            # nothing. The count only matters on a shared link; locally the
            # default of 20 is far more than anyone runs by hand in a day.
            used = runs_today()
            if DAILY_RUN_CAP > 0 and used >= DAILY_RUN_CAP:
                self._send_json(
                    {
                        "error": (
                            f"Daily limit reached ({used}/{DAILY_RUN_CAP} runs). "
                            "This resets at midnight, or the owner can raise "
                            "CONTENT_STUDIO_DAILY_RUNS in .env."
                        )
                    },
                    429,
                )
                return

            kind = body.get("kind", "studio")
            # An absent count means "auto": make_shorts takes its number from
            # the talk's replay peaks rather than being told one.
            raw_count = body.get("count")
            options = {
                "count": (
                    max(1, min(int(raw_count), 8)) if str(raw_count).strip().isdigit() else None
                ),
                "style": body.get("style", "broll"),
                "carousel": bool(body.get("carousel", False)),
            }
            job = Job(url, kind=kind, options=options)
            record_run()
            with JOBS_LOCK:
                JOBS[job.id] = job
            job.start()
            self._send_json(job.snapshot(), 201)
            return

        if path.startswith("/api/jobs/") and path.endswith("/cancel"):
            job_id = path[len("/api/jobs/") : -len("/cancel")]
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if job is None:
                self._send_json({"error": "Unknown job"}, 404)
                return
            if job.status != "running":
                self._send_json({"error": "That run already finished."}, 400)
                return
            job.status = "cancelled"
            job.error = "Cancelled from the UI."
            job.cancel()
            self._send_json(job.snapshot())
            return

        self._send_json({"error": "Not found"}, 404)


def main() -> None:
    # Windows consoles default to cp1252, which can't encode the arrows the
    # pipeline prints. Ask for UTF-8 and degrade gracefully if unavailable.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.isfile(os.path.join(UI_DIR, "index.html")):
        print(f"UI files missing - expected {os.path.join(UI_DIR, 'index.html')}")
        sys.exit(1)

    # Every option this server can send, checked against what the pipeline
    # actually accepts. A drift here costs nothing until someone presses the
    # button, and then it costs the whole run — so it is worth one subprocess
    # at startup to find out while there is still someone reading the console.
    script = os.path.join(BASE_DIR, "make_shorts.py")
    known = accepted_flags(script)
    unknown = sorted({"--style", "--count", "--no-carousel"} - known) if known else []

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    status = env_status()

    print("AI Content Studio - local UI")
    print(f"  {url}")
    print(f"  ANTHROPIC_API_KEY:  {'set' if status['anthropic'] else 'MISSING (runs will fail)'}")
    print(f"  ELEVENLABS_API_KEY: {'set' if status['elevenlabs'] else 'not set (voice-over skipped)'}")
    if unknown:
        print(f"  ⚠ make_shorts.py does not accept {' '.join(unknown)} — "
              f"those options will be dropped from runs. Fix the mismatch.")
    else:
        print("  Pipeline options:   match")
    print("  Ctrl+C to stop.\n")

    if os.environ.get("CONTENT_STUDIO_UI_NO_BROWSER") != "1":
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        # Stop any run we started. Left alone, the child keeps going until its
        # next print hits the closed pipe and dies mid-stage with a
        # BrokenPipeError — losing whatever it hadn't saved yet.
        running = active_job()
        if running is not None:
            print("  stopping the run in progress (its finished outputs are already saved)")
            running.status = "cancelled"
            running.error = "The UI server was stopped."
            running.cancel()
        server.server_close()


if __name__ == "__main__":
    main()
