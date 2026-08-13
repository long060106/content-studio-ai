"""
webapp.py

A small local web UI for AI Content Studio.

    python webapp.py            # then open http://127.0.0.1:8420

What it does:
  - Paste a YouTube URL and hit Run. The server shells out to `cli.py`
    (the exact same pipeline you'd run by hand), streams its stdout, and
    turns the progress lines into a live checklist.
  - Browse everything already in ./output/ — blog post, thread, LinkedIn
    post, captions, carousel slides, short-form clip, voice-over — in one
    place, with copy buttons and inline players.

Deliberately dependency-free: only the Python standard library is used, so
the UI runs with or without the project's venv activated. It binds to
127.0.0.1 only and never sends your API keys to the browser (it just reports
whether they are set).
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(BASE_DIR, "ui")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

HOST = "127.0.0.1"
PORT = int(os.environ.get("CONTENT_STUDIO_UI_PORT", "8420"))

MAX_LOG_LINES = 4000

# ---------------------------------------------------------------------------
# Pipeline steps
#
# cli.py prints one "→ ..." line per stage, then indented "✓ / ⚠" result
# lines. We map those to a fixed checklist so the UI can show progress
# without cli.py needing to know anything about the web layer.
# ---------------------------------------------------------------------------

STEPS = [
    {"id": "transcript", "label": "Transcript"},
    {"id": "brief", "label": "Content brief"},
    {"id": "blog", "label": "Blog post"},
    {"id": "thread", "label": "X / Twitter thread"},
    {"id": "linkedin", "label": "LinkedIn post"},
    {"id": "captions", "label": "Social captions"},
    {"id": "voiceover", "label": "Voice-over"},
    {"id": "carousel", "label": "Carousel slides"},
    {"id": "clip", "label": "Short-form clip"},
]

_STEP_MATCHERS = [
    ("transcript", ("extracting transcript",)),
    ("brief", ("content brief",)),
    ("blog", ("blog post",)),
    ("thread", ("twitter",)),
    ("linkedin", ("linkedin",)),
    ("captions", ("generating captions",)),
    ("voiceover", ("voice-over",)),
    ("carousel", ("carousel",)),
    ("clip", ("short-form",)),
]

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


def _step_for_line(text: str) -> str | None:
    lowered = text.lower()
    for step_id, needles in _STEP_MATCHERS:
        if any(n in lowered for n in needles):
            return step_id
    return None


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------


class Job:
    def __init__(self, url: str):
        self.id = uuid.uuid4().hex[:12]
        self.url = url
        self.video_id = guess_video_id(url)
        self.status = "running"  # running | done | error | cancelled
        self.error: str | None = None
        self.created_at = time.time()
        self.finished_at: float | None = None
        self.log: list[str] = []
        self.steps = {s["id"]: {"state": "pending", "detail": ""} for s in STEPS}
        self.current: str | None = None
        self.step_started: float = time.time()
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

    # -- state as seen by the browser ---------------------------------------

    def snapshot(self, log_from: int = 0) -> dict:
        with self._lock:
            return {
                "id": self.id,
                "url": self.url,
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
                    for s in STEPS
                ],
                "log": self.log[log_from:],
                "log_total": len(self.log),
            }

    # -- running ------------------------------------------------------------

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()

    def cancel(self) -> bool:
        with self._lock:
            proc = self._proc
            if proc is None or proc.poll() is not None:
                return False
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

    def _run(self) -> None:
        env = dict(os.environ)
        # cli.py prints →/✓/⚠; without this the child crashes writing them to
        # a pipe on Windows (cp1252 can't encode them).
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"

        cmd = [self._python_exe(), "-u", "cli.py", self.url]
        self._append(f"$ {' '.join(cmd)}")

        # cli.py's imports (anthropic, yt-dlp, Pillow…) take ~10s before it
        # prints anything, so show the first step as busy right away.
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
                for state in self.steps.values():
                    if state["state"] == "running":
                        state["state"] = "done"
            else:
                self.status = "error"
                if not self.error:
                    self.error = f"Pipeline exited with code {code}."
                for state in self.steps.values():
                    if state["state"] == "running":
                        state["state"] = "error"

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
            if s.startswith("→"):
                body = s[1:].strip()
                step_id = _step_for_line(body)
                if step_id:
                    # Anything still marked running before this belongs to a
                    # finished stage.
                    for other_id, state in self.steps.items():
                        if other_id != step_id and state["state"] == "running":
                            state["state"] = "done"
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
                for state in self.steps.values():
                    if state["state"] == "running":
                        state["state"] = "done"
            elif s.startswith("✓") and self.current:
                state = self.steps[self.current]
                if state["state"] not in ("warn", "error"):
                    state["state"] = "done"
                state["detail"] = s[1:].strip()
            elif s.startswith("⚠") and self.current:
                state = self.steps[self.current]
                state["state"] = "warn"
                state["detail"] = s[1:].strip()
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


def library_index() -> list[dict]:
    if not os.path.isdir(OUTPUT_DIR):
        return []

    items = []
    for name in sorted(os.listdir(OUTPUT_DIR)):
        video_dir = os.path.join(OUTPUT_DIR, name)
        if not os.path.isdir(video_dir):
            continue

        transcript = _read_json(os.path.join(video_dir, "transcript.json")) or {}
        brief = _read_json(os.path.join(video_dir, "brief.json")) or {}
        carousel_dir = os.path.join(video_dir, "carousel")

        def has(*parts: str) -> bool:
            return os.path.exists(os.path.join(video_dir, *parts))

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
                "assets": {
                    "brief": has("brief.json"),
                    "blog": has("blog_post.md"),
                    "thread": has("twitter_thread.json"),
                    "linkedin": has("linkedin_post.md"),
                    "captions": has("captions.json"),
                    "carousel": os.path.isdir(carousel_dir)
                    and any(f.endswith(".png") for f in os.listdir(carousel_dir)),
                    "voiceover": has("voiceover.mp3") or has("voiceover_script.txt"),
                    "clip": has("short_form_clip.mp4"),
                    "transcript": has("transcript.json"),
                },
            }
        )

    items.sort(key=lambda i: i["modified"], reverse=True)
    return items


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

    claims_raw = _read_text(os.path.join(video_dir, "linkedin_claims_to_verify.txt"))
    claims = []
    if claims_raw:
        claims = [
            line.lstrip("- ").strip()
            for line in claims_raw.splitlines()
            if line.strip().startswith("-")
        ]

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
        "linkedin": {
            "post": _read_text(os.path.join(video_dir, "linkedin_post.md")),
            "claims": claims,
        },
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


class Handler(BaseHTTPRequestHandler):
    server_version = "ContentStudioUI"

    def log_message(self, fmt, *args):  # quieter console
        pass

    # -- helpers ------------------------------------------------------------

    def _send_json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
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
        start, end = 0, size - 1
        status = 200

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
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download:
            self.send_header(
                "Content-Disposition",
                f'attachment; filename="{os.path.basename(path)}"',
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
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

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
                    "library": library_index(),
                    "jobs": sorted(jobs, key=lambda j: j["created_at"], reverse=True),
                    "steps": STEPS,
                }
            )
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

    def do_POST(self):  # noqa: N802
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

            job = Job(url)
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

    server = ThreadingHTTPServer((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}"
    status = env_status()

    print("AI Content Studio - local UI")
    print(f"  {url}")
    print(f"  ANTHROPIC_API_KEY:  {'set' if status['anthropic'] else 'MISSING (runs will fail)'}")
    print(f"  ELEVENLABS_API_KEY: {'set' if status['elevenlabs'] else 'not set (voice-over skipped)'}")
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
