"""
audio_transcriber.py

Fallback transcription path for videos that have no YouTube captions
available. Downloads the video's audio via yt-dlp, then transcribes it
locally using OpenAI's Whisper model.

This is slower and heavier than pulling existing captions (it downloads
audio and runs a local ML model), so it's only used when
youtube_extractor.get_transcript() fails with NoTranscriptAvailable.

Requires:
  - ffmpeg installed and on PATH (used by yt-dlp to extract audio, and by
    Whisper to decode it)
  - openai-whisper package (pulls in PyTorch). The first time this runs it
    will also download the model weights (~150MB for the "base" model,
    cached afterward in ~/.cache/whisper).
"""

from __future__ import annotations

import os
import shutil
import tempfile

import whisper
import yt_dlp

from youtube_extractor import yt_dlp_cookie_opts


# Model size tradeoffs (speed vs accuracy vs download size):
#   tiny   -> fastest, least accurate, ~75MB
#   base   -> good default balance, ~150MB      <- used here
#   small  -> more accurate, slower, ~500MB
#   medium -> even more accurate, much slower, ~1.5GB
WHISPER_MODEL_SIZE = "base"

_model = None  # cached in-process so repeated calls don't reload the model


def _get_model():
    global _model
    if _model is None:
        print(f"  (loading Whisper '{WHISPER_MODEL_SIZE}' model — "
              f"first run downloads it, ~150MB, then it's cached)")
        _model = whisper.load_model(WHISPER_MODEL_SIZE)
    return _model


def download_audio(url: str, download_dir: str) -> str:
    """Downloads the best available audio track for a YouTube video as WAV. Returns the file path."""
    output_template = os.path.join(download_dir, "audio.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
            "preferredquality": "192",
        }],
        **yt_dlp_cookie_opts(),
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    audio_path = os.path.join(download_dir, "audio.wav")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(
            f"Expected downloaded audio at {audio_path}, but it wasn't created. "
            f"Is ffmpeg installed and on PATH?"
        )
    return audio_path


def transcribe_audio(audio_path: str) -> tuple[str, str, list]:
    """
    Runs Whisper on a local audio file.
    Returns (text, detected_language_code, segments) where segments =
    [{"text": str, "start": float, "duration": float}, ...] — Whisper
    provides these natively, useful later for picking a video clip window.
    """
    model = _get_model()
    result = model.transcribe(audio_path)

    segments = [
        {
            "text": seg["text"].strip(),
            "start": float(seg["start"]),
            "duration": float(seg["end"] - seg["start"]),
        }
        for seg in result.get("segments", [])
    ]

    return result["text"].strip(), result.get("language", "en"), segments


def transcribe_from_youtube(url: str) -> tuple[str, str, list]:
    """
    Full fallback path: download audio for the given YouTube URL, then
    transcribe it locally. Returns (transcript_text, language_code, segments).
    Cleans up the downloaded audio file afterward regardless of outcome.
    """
    tmp_dir = tempfile.mkdtemp(prefix="content_studio_audio_")
    try:
        print("  → No captions available — downloading audio for local transcription...")
        audio_path = download_audio(url, tmp_dir)
        print("  → Transcribing audio locally with Whisper (this can take a couple minutes)...")
        text, language, segments = transcribe_audio(audio_path)
        return text, language, segments
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python audio_transcriber.py <youtube_url>")
        sys.exit(1)

    text, lang, segments = transcribe_from_youtube(sys.argv[1])
    print(f"Language: {lang}")
    print(text)
