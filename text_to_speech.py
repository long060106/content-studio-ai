"""
text_to_speech.py

Converts a voice-over script into narrated audio using the ElevenLabs
text-to-speech API. Defaults to the "flash" model — faster and cheaper
(fewer credits used) than the higher-fidelity models, a good fit for
testing on a free-tier account.

Note on voices: free-tier accounts can't use ElevenLabs' shared Voice
Library via the API — only voices already added to *your* account. Rather
than hardcoding a voice ID that might not be accessible, this looks up
whatever voice(s) your account actually has and uses the first one found.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs

load_dotenv()

DEFAULT_MODEL_ID = "eleven_flash_v2_5"


def _resolve_voice_id(client: ElevenLabs, voice_id: Optional[str]) -> str:
    if voice_id:
        return voice_id
    env_voice = os.environ.get("ELEVENLABS_VOICE_ID")
    if env_voice:
        return env_voice

    try:
        response = client.voices.search()
        voices = getattr(response, "voices", None) or []
    except Exception as e:
        raise RuntimeError(f"Could not look up available ElevenLabs voices: {e}") from e

    if not voices:
        raise RuntimeError(
            "No voices are available on this ElevenLabs account. Free-tier "
            "accounts can't use the shared Voice Library via the API — go to "
            "https://elevenlabs.io/app/voice-library, pick a voice, and click "
            "'Add to my voices' so it's usable via the API. Then rerun."
        )

    return voices[0].voice_id


def synthesize_speech(
    text: str,
    output_path: str,
    voice_id: Optional[str] = None,
    model_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """Converts text to speech and saves it as an MP3 at output_path. Returns the path."""
    client = ElevenLabs(api_key=api_key or os.environ.get("ELEVENLABS_API_KEY"))

    resolved_voice_id = _resolve_voice_id(client, voice_id)
    model_id = model_id or os.environ.get("ELEVENLABS_MODEL_ID") or DEFAULT_MODEL_ID

    audio = client.text_to_speech.convert(
        text=text,
        voice_id=resolved_voice_id,
        model_id=model_id,
        output_format="mp3_44100_128",
    )

    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "wb") as f:
        for chunk in audio:
            if chunk:
                f.write(chunk)

    return output_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python text_to_speech.py <path_to_script.txt> <output_mp3_path>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        script_text = f.read()

    path = synthesize_speech(script_text, sys.argv[2])
    print(f"Saved audio to {path}")