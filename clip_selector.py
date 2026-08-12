"""
clip_selector.py

Picks the best short window (20-60 seconds) from a video's timestamped
transcript for a short-form video clip (Shorts/Reels/TikTok) — a
self-contained moment that makes sense without the rest of the video.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

from content_brief import ContentBrief

load_dotenv()

MODEL = "claude-sonnet-4-6"

# Cap how many segments we send to keep prompts reasonable on very long videos.
MAX_SEGMENTS = 1000

SYSTEM_PROMPT = """You are an editor who selects the single best short clip (20-60 \
seconds) from a longer video's timestamped transcript, for repurposing as a \
Shorts/Reels/TikTok clip.

Pick a moment that:
- Is self-contained — makes sense on its own without the surrounding video for context
- Captures one complete idea, not a fragment cut off mid-thought
- Starts and ends at natural sentence boundaries (use the segment timestamps to find these)
- Is the most compelling moment available — surprising, clearly-explained, or the \
"aha" point — not necessarily the literal opening of the video

Respond with ONLY valid JSON, no preamble, no markdown code fences."""

SCHEMA_DESCRIPTION = """
Return a JSON object with exactly this shape:

{
  "start_seconds": number,
  "end_seconds": number,
  "reason": string   // one sentence on why this moment was chosen
}
"""


@dataclass
class ClipSelection:
    start_seconds: float
    end_seconds: float
    reason: str

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds


def _build_user_prompt(brief: ContentBrief, segments: list[dict]) -> str:
    capped = segments[:MAX_SEGMENTS]
    segments_text = "\n".join(f'[{s["start"]:.1f}s] {s["text"]}' for s in capped)
    truncated_note = "" if len(segments) <= MAX_SEGMENTS else "\n[transcript truncated for length]"

    return f"""Video summary: {brief.summary}

Timestamped transcript segments:
{segments_text}{truncated_note}

Pick the best 20-60 second window for a short-form video clip.

{SCHEMA_DESCRIPTION}
"""


def select_clip(brief: ContentBrief, segments: list[dict], api_key: Optional[str] = None) -> ClipSelection:
    if not segments:
        raise ValueError("No timestamped transcript segments available to select a clip from.")

    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(brief, segments)}],
    )

    raw_text = response.content[0].text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON. Raw output:\n{raw_text}") from e

    return ClipSelection(
        start_seconds=float(data.get("start_seconds", 0)),
        end_seconds=float(data.get("end_seconds", 0)),
        reason=data.get("reason", ""),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python clip_selector.py <path_to_transcript.json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)

    brief_stub = ContentBrief(
        video_id=data.get("video_id", ""),
        title_suggestions=[],
        summary=data.get("title", ""),
        target_audience="",
        tone="",
        key_points=[],
        hooks=[],
        notable_quotes=[],
        call_to_action="",
        topics=[],
    )
    selection = select_clip(brief_stub, data.get("transcript_segments", []))
    print(f"{selection.start_seconds:.1f}s - {selection.end_seconds:.1f}s ({selection.duration:.1f}s)")
    print(selection.reason)
