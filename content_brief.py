"""
content_brief.py

Takes the raw transcript + metadata from youtube_extractor and turns it into a
structured "Content Brief": the intermediate representation every downstream
generator (blog, thread, LinkedIn post, etc.) will consume.

Why an intermediate brief instead of generating each format directly from the
raw transcript?
  - Higher quality: each generator works from a distilled, structured summary
    instead of a noisy wall of spoken-word text.
  - Cheaper: the brief is generated once; format generators can reuse it
    without re-sending the full transcript each time.
  - Consistent: every output format pulls from the same key points, so your
    blog post and your tweet thread don't contradict each other.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

from youtube_extractor import VideoData

load_dotenv()

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a content strategist. You read video transcripts and \
distill them into a structured content brief that other writers will use to \
create blog posts, social threads, and scripts. Be precise and concrete: pull \
real quotes and real points from the transcript, don't invent generic ones. \
Respond with ONLY valid JSON, no preamble, no markdown code fences."""

BRIEF_SCHEMA_DESCRIPTION = """
Return a JSON object with exactly this shape:

{
  "title_suggestions": [string, string, string],
  "summary": string,                 // 2-3 sentence overview of the video
  "target_audience": string,         // who this content is for
  "tone": string,                    // e.g. "casual and energetic", "technical and authoritative"
  "key_points": [                    // 4-8 main points made in the video, in order
    {
      "point": string,               // the idea, in your own words
      "supporting_quote": string     // a short, real quote from the transcript backing it up (<25 words)
    }
  ],
  "hooks": [string, string, string], // 3 attention-grabbing opening lines derived from the content,
                                      // suitable for social media (tweet/reel openers)
  "notable_quotes": [string, ...],   // up to 5 standout verbatim quotes worth featuring
  "call_to_action": string,          // what the video wants the viewer to do/think, if any
  "topics": [string, ...]            // 3-6 short topic tags for categorization
}
"""


@dataclass
class ContentBrief:
    video_id: str
    title_suggestions: list[str]
    summary: str
    target_audience: str
    tone: str
    key_points: list[dict]
    hooks: list[str]
    notable_quotes: list[str]
    call_to_action: str
    topics: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> "ContentBrief":
        return cls(**json.loads(json_str))


def _build_user_prompt(video: VideoData) -> str:
    # Truncate extremely long transcripts to keep cost/latency sane.
    # ~12,000 words is a generous cap for a single video; adjust as needed.
    words = video.transcript_text.split()
    transcript_excerpt = " ".join(words[:12000])
    truncated_note = "" if len(words) <= 12000 else "\n\n[Transcript truncated for length.]"

    return f"""Video title: {video.title}
Channel: {video.channel}
Duration: {video.duration_seconds} seconds

Transcript:
\"\"\"
{transcript_excerpt}{truncated_note}
\"\"\"

{BRIEF_SCHEMA_DESCRIPTION}
"""


def generate_brief(video: VideoData, api_key: Optional[str] = None) -> ContentBrief:
    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(video)}],
    )

    raw_text = response.content[0].text.strip()
    # Defensive cleanup in case the model wraps output in code fences anyway
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON. Raw output:\n{raw_text}"
        ) from e

    return ContentBrief(
        video_id=video.video_id,
        title_suggestions=data.get("title_suggestions", []),
        summary=data.get("summary", ""),
        target_audience=data.get("target_audience", ""),
        tone=data.get("tone", ""),
        key_points=data.get("key_points", []),
        hooks=data.get("hooks", []),
        notable_quotes=data.get("notable_quotes", []),
        call_to_action=data.get("call_to_action", ""),
        topics=data.get("topics", []),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python content_brief.py <youtube_url>")
        sys.exit(1)

    from youtube_extractor import extract

    video = extract(sys.argv[1])
    brief = generate_brief(video)
    print(brief.to_json())
