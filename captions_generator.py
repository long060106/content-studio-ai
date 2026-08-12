"""
captions_generator.py

Generates short-form social media captions (for Instagram, TikTok, YouTube
Shorts, etc.) from a video's content brief. These are distinct from the
LinkedIn post and Twitter thread: shorter, punchier, built around a strong
first line (since most platforms truncate captions behind a "more" tap), and
paired with relevant hashtags.

Produces 3 variants so you can pick the angle that fits the specific clip/post:
  - hook: leads with a bold, curiosity-driving opening line
  - story: leads with a short personal/narrative framing
  - question: leads with a direct question to the reader
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

from content_brief import ContentBrief

load_dotenv()

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a skilled short-form social media writer who turns video \
content briefs into scroll-stopping captions for Instagram, TikTok, and YouTube \
Shorts.

TONE
Punchy, conversational, genuinely interesting — not generic hype. Avoid corporate \
language, excessive emojis (0-3 max per caption, used naturally not decoratively), \
and clickbait that oversells what the content actually delivers.

STRUCTURE
- The first line is the most important part — most platforms truncate captions \
behind a "more" tap, so the opening line must work as a standalone hook.
- Keep captions short: 1-3 short sentences or line-broken fragments, not paragraphs.
- Each caption variant should take a different angle (specified per-variant below), \
not just reword the same sentence.

ACCURACY
- Never state a claim more absolutely than the brief supports.
- Only use facts and framing explicitly present in the brief. Never invent \
statistics, studies, or quotes.
- Keep claims consistent with the nuance in the brief — don't oversimplify into \
a false absolute just because the format is short.

HASHTAGS
For each caption, provide 5-10 relevant hashtags separately from the caption text \
(not embedded inline). Mix broader topic tags with a couple more specific ones. \
Avoid generic spam tags like #viral or #fyp unless genuinely fitting.

Respond with ONLY valid JSON, no preamble, no markdown code fences."""

SCHEMA_DESCRIPTION = """
Return a JSON object with exactly this shape:

{
  "captions": [
    {
      "style": "hook",       // leads with a bold, curiosity-driving opening line
      "text": string,        // the caption itself, 1-3 short sentences
      "hashtags": [string, ...]   // 5-10 tags, no # symbol needed in the string
    },
    {
      "style": "story",      // leads with a short personal/narrative framing
      "text": string,
      "hashtags": [string, ...]
    },
    {
      "style": "question",   // leads with a direct question to the reader
      "text": string,
      "hashtags": [string, ...]
    }
  ]
}
"""


@dataclass
class Caption:
    style: str
    text: str
    hashtags: list[str]

    def hashtag_line(self) -> str:
        return " ".join(f"#{tag.lstrip('#')}" for tag in self.hashtags)


@dataclass
class CaptionSet:
    captions: list[Caption]

    def to_json(self) -> str:
        return json.dumps(
            {"captions": [asdict(c) for c in self.captions]}, indent=2, ensure_ascii=False
        )

    def to_text(self) -> str:
        blocks = []
        for c in self.captions:
            blocks.append(f"[{c.style}]\n{c.text}\n\n{c.hashtag_line()}")
        return "\n\n---\n\n".join(blocks)


def _build_user_prompt(brief: ContentBrief) -> str:
    key_points_text = "\n".join(f'- {kp.get("point", "")}' for kp in brief.key_points)
    hooks_text = "\n".join(f"- {h}" for h in brief.hooks)

    return f"""Write social media captions based on this video content brief.

Summary: {brief.summary}
Tone: {brief.tone}
Target audience: {brief.target_audience}
Topics: {', '.join(brief.topics)}

Key points available:
{key_points_text}

Possible hooks to inspire the "hook" variant:
{hooks_text}

Call to action: {brief.call_to_action}

{SCHEMA_DESCRIPTION}
"""


def generate_captions(brief: ContentBrief, api_key: Optional[str] = None) -> CaptionSet:
    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(brief)}],
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

    captions = [
        Caption(
            style=c.get("style", ""),
            text=c.get("text", ""),
            hashtags=c.get("hashtags", []),
        )
        for c in data.get("captions", [])
    ]
    return CaptionSet(captions=captions)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python captions_generator.py <path_to_brief.json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        brief = ContentBrief.from_json(f.read())

    caption_set = generate_captions(brief)
    print(caption_set.to_text())
