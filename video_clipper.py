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

import yt_dlp

from youtube_extractor import yt_dlp_cookie_opts


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
        **yt_dlp_cookie_opts(),
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

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
        "-i", input_path,
        "-filter_complex", filter_complex,
        "-map", "[outv]",
        "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
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
