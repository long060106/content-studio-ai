"""
video_clipper.py

Downloads a specific time range of a YouTube video and converts it into a
vertical (9:16) short-form video, ready for Shorts/Reels/TikTok — using
yt-dlp (to fetch just the needed section, not the whole video) and ffmpeg
(to reformat it).

Vertical conversion uses a blurred, scaled-up copy of the clip itself as a
background fill with the original centered on top — a common, clean way to
fit horizontal footage into a vertical frame without cropping content out.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import time

import yt_dlp

from youtube_extractor import yt_dlp_base_opts

# Re-encoding a 20-60s clip takes seconds; anything near this means ffmpeg is
# stuck rather than slow.
FFMPEG_TIMEOUT = 600


def _progress_hook(status: dict) -> None:
    """Print an occasional one-line progress update.

    yt-dlp is otherwise completely silent here, and this step routinely runs
    for minutes — without this the pipeline looks hung when it's working fine.
    We print at most once every few seconds so logs stay readable.
    """
    if status.get("status") == "finished":
        print("    …download finished, handing off to ffmpeg", flush=True)
        return
    if status.get("status") != "downloading":
        return

    now = time.monotonic()
    if now - _progress_hook.last < 5:
        return
    _progress_hook.last = now

    total = status.get("total_bytes") or status.get("total_bytes_estimate")
    done = status.get("downloaded_bytes") or 0
    if total:
        print(f"    …downloading {done / total:.0%} ({done / 1e6:.1f} of {total / 1e6:.1f} MB)", flush=True)
    else:
        print(f"    …downloading {done / 1e6:.1f} MB", flush=True)


_progress_hook.last = 0.0


class YouTubeBlockedDownload(RuntimeError):
    """YouTube refused to serve the media, rather than anything being wrong locally."""


def _explain_download_failure(error: Exception) -> Exception:
    """Turn yt-dlp's opaque failures into something actionable.

    The surface error is "ffmpeg exited with code 3436169992" — a meaningless
    Windows exit status that sends you hunting through ffmpeg and the
    filtergraph. Underneath it is an HTTP 403 on the media URL.

    The usual cause is *not* authentication, however much it looks like it.
    YouTube hands out media URLs that only work once a JavaScript "n challenge"
    has been solved, and when yt-dlp can't solve it the URLs it does produce are
    dead on arrival. Cookies and fresh accounts change nothing here; a JS
    runtime does. So that's listed first, and the cookie advice second.
    """
    text = str(error)
    blocked = (
        "403" in text
        or "Forbidden" in text
        or "Sign in to confirm" in text
        or "not a bot" in text
        # yt-dlp reports the ffmpeg downloader's exit status rather than the
        # 403 that caused it, so an ffmpeg failure here means the same thing.
        or "ffmpeg exited with code" in text
    )
    if not blocked:
        return error
    from youtube_extractor import yt_dlp_js_opts

    if not yt_dlp_js_opts():
        return YouTubeBlockedDownload(
            "YouTube refused to serve this video's media (HTTP 403), and no "
            "JavaScript runtime was found — almost certainly the cause.\n"
            "  Fix: install Node (or Deno) so yt-dlp can solve YouTube's n "
            "challenge, plus the solver scripts:\n"
            "    pip install yt-dlp-ejs"
        )
    return YouTubeBlockedDownload(
        "YouTube refused to serve this video's media (HTTP 403), despite a JS "
        "runtime being available.\n"
        "  Check first: pip install yt-dlp-ejs, and that yt-dlp is current.\n"
        "  If that's already so, try session cookies: export cookies.txt from a "
        "logged-in browser and set YTDLP_COOKIES_FILE=<path> in .env."
    )


def download_clip(url: str, start_seconds: float, end_seconds: float, output_path: str) -> str:
    """Downloads only the given time range of the video as an MP4. Returns the path."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    def _ranges(info_dict, ydl):
        return [{"start_time": start_seconds, "end_time": end_seconds}]

    base = output_path[:-4] if output_path.endswith(".mp4") else output_path

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": base + ".%(ext)s",
        "download_ranges": _ranges,
        "force_keyframes_at_cuts": True,
        "quiet": True,
        "no_warnings": True,
        # Without these a stalled CDN connection hangs the whole pipeline
        # indefinitely; retries cover the usual transient failures.
        "socket_timeout": 30,
        "retries": 3,
        "progress_hooks": [_progress_hook],
        **yt_dlp_base_opts(),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise _explain_download_failure(e) from e

    if not os.path.exists(output_path):
        # yt-dlp may have produced a different container; find whatever it made.
        for ext in ("mp4", "mkv", "webm"):
            candidate = f"{base}.{ext}"
            if os.path.exists(candidate):
                if candidate != output_path:
                    os.replace(candidate, output_path)
                break
        else:
            raise FileNotFoundError(
                f"Expected downloaded clip at {output_path}, but it wasn't created."
            )

    return output_path


def download_full(url: str, output_path: str, max_height: int = 1080) -> str:
    """Download the whole video once, so cuts can be taken locally.

    Worth it as soon as a talk yields more than one cut. Each ranged download
    costs 20-30 seconds on this machine, of which only about 7 is transfer —
    the rest is yt-dlp re-extracting the video and re-solving YouTube's
    JavaScript challenge through Node, and that price is paid again for every
    single cut. Fetching once and slicing locally pays the extraction cost a
    single time, and each cut afterwards costs no network at all.

    Capped at 1080p on purpose: the short is a vertical crop out of the middle
    of the frame, so beyond 1080 the extra pixels are mostly cropped away while
    the download grows.
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if os.path.isfile(output_path) and os.path.getsize(output_path) > 0:
        return output_path

    base = output_path[:-4] if output_path.endswith(".mp4") else output_path
    ydl_opts = {
        "format": (
            f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/"
            f"best[height<={max_height}][ext=mp4]/best"
        ),
        "outtmpl": base + ".%(ext)s",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "socket_timeout": 30,
        "retries": 3,
        "progress_hooks": [_progress_hook],
        **yt_dlp_base_opts(),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        raise _explain_download_failure(e) from e

    if not os.path.exists(output_path):
        for ext in ("mp4", "mkv", "webm"):
            candidate = f"{base}.{ext}"
            if os.path.exists(candidate):
                if candidate != output_path:
                    os.replace(candidate, output_path)
                break
        else:
            raise FileNotFoundError(f"Expected the video at {output_path}, but it wasn't created.")
    return output_path


def cut_from_file(source_path: str, start_seconds: float, end_seconds: float, output_path: str) -> str:
    """Cut a time range out of a video already on disk. Returns the path.

    The offline counterpart to `download_clip`, for when you have the whole
    talk as a local file. Everything downstream is identical — the callers only
    care that a clip lands at `output_path`.

    Worth preferring where possible: one local file serves every cut of every
    short, so a talk needs a single fetch instead of one per cut, and no cut
    can fail halfway through a batch because the network changed its mind.

    `-ss` before `-i` seeks fast, and because the output is re-encoded rather
    than stream-copied, the cut is still frame-accurate — ffmpeg decodes from
    the preceding keyframe and discards the lead-in. A stream copy would be
    faster again but would snap to keyframes, which loses words at the start.
    """
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Source video not found: {source_path}")

    duration = max(0.1, float(end_seconds) - float(start_seconds))
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-ss", f"{float(start_seconds):.3f}",
        "-i", os.path.abspath(source_path),
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        os.path.abspath(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffmpeg didn't finish cutting within 600s: {source_path}") from None
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg couldn't cut the clip:\n{result.stderr[-2000:]}")
    if not os.path.isfile(output_path):
        raise RuntimeError(f"ffmpeg reported success but no clip appeared at {output_path}")
    return output_path


def convert_to_vertical(input_path: str, output_path: str) -> str:
    """Converts a video to vertical 1080x1920 with a blurred-background fill."""
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    filter_complex = (
        "[0:v]scale=1080:1920:force_original_aspect_ratio=increase,"
        "crop=1080:1920,gblur=sigma=20[bg];"
        "[0:v]scale=1080:-2[fg];"
        "[bg][fg]overlay=(W-w)/2:(H-h)/2[outv]"
    )

    cmd = [
        "ffmpeg", "-y",
        # -nostdin: ffmpeg inherits our stdin otherwise and can block on it
        # when run from a server or a piped shell.
        "-nostdin",
        "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=FFMPEG_TIMEOUT)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"ffmpeg didn't finish within {FFMPEG_TIMEOUT}s converting to vertical format. "
            f"A 20-60s clip should take well under a minute — check that ffmpeg isn't "
            f"waiting on something."
        ) from None

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed converting to vertical format:\n{result.stderr[-2000:]}")

    return output_path


def create_short_form_clip(url: str, start_seconds: float, end_seconds: float, output_path: str) -> str:
    """Full pipeline: download the time range, then convert it to vertical format."""
    with tempfile.TemporaryDirectory(prefix="content_studio_clip_") as tmp_dir:
        raw_clip_path = os.path.join(tmp_dir, "raw_clip.mp4")
        download_clip(url, start_seconds, end_seconds, raw_clip_path)
        convert_to_vertical(raw_clip_path, output_path)
    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 5:
        print("Usage: python video_clipper.py <youtube_url> <start_seconds> <end_seconds> <output_mp4_path>")
        sys.exit(1)

    url, start, end, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    path = create_short_form_clip(url, start, end, out)
    print(f"Saved short-form clip to {path}")
