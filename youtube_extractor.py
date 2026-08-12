"""
youtube_extractor.py

Given a YouTube URL, pull:
  1. The transcript (captions) as plain text
  2. Video metadata (title, channel, description, duration)

Two extraction paths:
  - Primary: youtube-transcript-api (fast, no download, works when captions exist)
  - Metadata: yt-dlp (used for title/description/duration regardless of transcript source)

If no captions are available at all, we raise NoTranscriptAvailable so the
caller can decide whether to fall back to audio transcription (Whisper, etc.)
in a later stage of the pipeline.
"""

from __future__ import annotations

import os
import re
import json
from dataclasses import dataclass, asdict
from typing import Optional

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import CouldNotRetrieveTranscript
import yt_dlp

load_dotenv()


def yt_dlp_cookie_opts() -> dict:
    """
    Build optional yt-dlp options for authenticating as a real browser session.

    YouTube increasingly challenges yt-dlp with "Sign in to confirm you're
    not a bot." Setting YTDLP_COOKIES_FROM_BROWSER in .env (e.g. "chrome",
    "firefox", "edge", "brave") lets yt-dlp reuse that browser's existing
    YouTube login cookies to get past the challenge. Leave unset if you're
    not hitting this error — most requests don't need it.
    """
    browser = os.environ.get("YTDLP_COOKIES_FROM_BROWSER")
    if browser:
        return {"cookiesfrombrowser": (browser,)}
    return {}


class NoTranscriptAvailable(Exception):
    """Raised when no transcript/captions could be retrieved for the video."""


class InvalidYouTubeURL(Exception):
    """Raised when a video ID can't be parsed from the given URL."""


@dataclass
class VideoData:
    video_id: str
    url: str
    title: str
    channel: str
    duration_seconds: int
    description: str
    transcript_text: str
    transcript_language: str
    transcript_segments: list  # [{"text": str, "start": float, "duration": float}, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


_YOUTUBE_ID_PATTERNS = [
    r"(?:v=|\/)([0-9A-Za-z_-]{11}).*",
    r"(?:embed\/)([0-9A-Za-z_-]{11})",
    r"(?:youtu\.be\/)([0-9A-Za-z_-]{11})",
]


def extract_video_id(url: str) -> str:
    """Parse an 11-character YouTube video ID out of any common URL shape."""
    for pattern in _YOUTUBE_ID_PATTERNS:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # Handle bare video IDs passed directly
    if re.fullmatch(r"[0-9A-Za-z_-]{11}", url):
        return url
    raise InvalidYouTubeURL(f"Could not extract a video ID from: {url}")


def get_transcript(video_id: str) -> tuple[str, str, list]:
    """
    Returns (transcript_text, language_code, segments).
    segments = [{"text": str, "start": float, "duration": float}, ...] — needed
    later for picking a short-form video clip window.
    Prefers manually created transcripts over auto-generated ones,
    and prefers English but falls back to whatever's available.

    Note: youtube-transcript-api v1.x uses an instance-based API
    (YouTubeTranscriptApi().list(...)) rather than the v0.x static
    methods (YouTubeTranscriptApi.list_transcripts(...)).
    """
    ytt_api = YouTubeTranscriptApi()
    try:
        transcript_list = ytt_api.list(video_id)
    except CouldNotRetrieveTranscript as e:
        # Covers TranscriptsDisabled, NoTranscriptFound, RequestBlocked,
        # VideoUnavailable, and any other library error indicating we
        # can't get captions this way — caller falls back to Whisper.
        raise NoTranscriptAvailable(str(e)) from e

    available = list(transcript_list)
    if not available:
        raise NoTranscriptAvailable(f"No transcript found for video {video_id}")

    def pick(predicate):
        for t in available:
            if predicate(t):
                return t
        return None

    transcript = (
        pick(lambda t: not t.is_generated and t.language_code.startswith("en"))
        or pick(lambda t: not t.is_generated)
        or pick(lambda t: t.is_generated and t.language_code.startswith("en"))
        or pick(lambda t: t.is_generated)
    )

    if transcript is None:
        raise NoTranscriptAvailable(f"No transcript found for video {video_id}")

    fetched = transcript.fetch()

    def snippet_field(snippet, field, default=""):
        # v1.x yields FetchedTranscriptSnippet objects (attribute access);
        # older versions yielded plain dicts. Support both defensively.
        if isinstance(snippet, dict):
            return snippet.get(field, default)
        return getattr(snippet, field, default)

    segments = []
    text_parts = []
    for s in fetched:
        seg_text = snippet_field(s, "text", "").strip()
        if not seg_text:
            continue
        text_parts.append(seg_text)
        segments.append({
            "text": seg_text,
            "start": float(snippet_field(s, "start", 0.0)),
            "duration": float(snippet_field(s, "duration", 0.0)),
        })

    text = " ".join(text_parts)
    return text, transcript.language_code, segments


def get_metadata(url: str) -> dict:
    """Pull title/channel/duration/description via yt-dlp without downloading video."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        **yt_dlp_cookie_opts(),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
    return {
        "title": info.get("title", ""),
        "channel": info.get("channel") or info.get("uploader", ""),
        "duration_seconds": info.get("duration", 0),
        "description": info.get("description", ""),
    }


def extract(url: str) -> VideoData:
    """Main entry point: URL in, fully populated VideoData out."""
    video_id = extract_video_id(url)
    metadata = get_metadata(url)

    try:
        transcript_text, language, segments = get_transcript(video_id)
    except NoTranscriptAvailable:
        # Lazy import: whisper/torch are heavy, so we only pay that cost
        # when a video actually has no captions and we need the fallback.
        from audio_transcriber import transcribe_from_youtube
        transcript_text, language, segments = transcribe_from_youtube(url)

    return VideoData(
        video_id=video_id,
        url=url,
        title=metadata["title"],
        channel=metadata["channel"],
        duration_seconds=metadata["duration_seconds"],
        description=metadata["description"],
        transcript_text=transcript_text,
        transcript_language=language,
        transcript_segments=segments,
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python youtube_extractor.py <youtube_url>")
        sys.exit(1)

    data = extract(sys.argv[1])
    print(data.to_json())
