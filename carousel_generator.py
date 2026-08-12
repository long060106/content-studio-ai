"""
carousel_generator.py

Generates Instagram carousel slide content from a video's content brief.
Text only — carousel_renderer.py turns this into actual slide images.

Each slide can carry up to four elements, in a fixed visual hierarchy:
  - headline: the big idea (required)
  - flow: an optional minimal text diagram (e.g. "A → B → C"), used sparingly
    on 1-2 slides where it genuinely clarifies a mechanism or sequence
  - subtext: an optional supporting line
  - source: an optional small attribution/citation line — only included when
    the brief/video actually gives a real name to attribute to, never invented
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

SYSTEM_PROMPT = """You are a content designer who turns video content briefs into \
Instagram carousel slide text for a minimalist, research-driven aesthetic — think \
"modern research publication," not "motivational meme account." Each slide is a \
single visual card meant to be read in under 2 seconds while someone swipes.

STYLE
- Headlines: punchy, concrete, under 10 words. No filler words, no emojis, no hashtags.
- Subtext (optional): a short supporting line, under 15 words — only include it \
if it genuinely adds something the headline doesn't already say.
- Maintain ONE consistent typography hierarchy across all 8 slides: big idea → \
one supporting line → optional tiny source line. Don't vary this structure \
slide to slide.
- No markdown — this is rendered as an image, not read as text.

STRUCTURE (aim for 8 slides, following this arc)
1. HOOK — a bold, curiosity-driving claim (see ACCURACY below — bold does not \
mean overclaimed). Cover slide people see before swiping.
2. THE OLD BELIEF / MISCONCEPTION — what people commonly assume, that the video \
challenges.
3. THE CORE CONCEPT — introduce the key idea or term plainly. Keep explanation \
minimal here; let the design do the work. Good candidate for a "flow" diagram \
if the brief describes a clear sequence.
4. THE SURPRISING PART — a counterintuitive finding from the brief.
5. THE MECHANISM — what actually drives the effect. Use a real supporting quote \
from the brief if one fits naturally. Another good candidate for "flow."
6. THE NUANCE — an important limitation or "X isn't automatically Y — what \
matters is Z" correction, so the carousel doesn't overclaim.
7. THE BROADER IMPLICATION — how this connects to something bigger (e.g. "this \
cuts both ways," or a wider consequence named in the brief).
8. TAKEAWAY — the single most useful practical point, ending with a genuinely \
reflective question for the reader. Do NOT end with "Follow for more" or similar \
generic engagement bait — a real question is more aligned with this account's voice.

Use AT MOST 1-2 "flow" diagrams across the whole carousel — it's a texture \
element, not the default.

ACCURACY
- Never state a claim more absolutely than the brief supports, especially on the \
cover slide — a bold hook and an overclaimed hook are different things. Prefer \
"changes" over "always changes," "can" over "will," etc., unless the brief \
genuinely supports the stronger version.
- Only use facts and quotes explicitly present in the brief. Never invent \
statistics, studies, or quotes.
- For the "source" field: only include a real name/affiliation if the video \
title or brief actually gives you one to attribute to. Never invent a name, \
title, or affiliation. Leave it null if you don't have a real one.
- Because each slide is so short, resist the urge to oversimplify a nuanced \
claim into a false absolute just to make it punchier — cut detail instead of \
distorting it.

Respond with ONLY valid JSON, no preamble, no markdown code fences."""

SCHEMA_DESCRIPTION = """
Return a JSON object with exactly this shape:

{
  "slides": [
    {
      "headline": string,          // under 10 words
      "flow": string or null,      // e.g. "CHEMISTRY \u2192 STRUCTURE \u2192 FUNCTION" — use on at most 1-2 slides total, null otherwise
      "subtext": string or null,   // under 15 words, or null if not needed
      "source": string or null     // e.g. "Jane Smith, MIT" — ONLY if a real name is available, otherwise null
    }
  ]
}
"""


@dataclass
class CarouselSlide:
    headline: str
    subtext: Optional[str] = None
    flow: Optional[str] = None
    source: Optional[str] = None


@dataclass
class Carousel:
    slides: list[CarouselSlide]

    def to_json(self) -> str:
        return json.dumps(
            {"slides": [asdict(s) for s in self.slides]}, indent=2, ensure_ascii=False
        )


def _build_user_prompt(brief: ContentBrief, video_title: str) -> str:
    key_points_text = "\n".join(
        f'- {kp.get("point", "")} (source quote: "{kp.get("supporting_quote", "")}")'
        for kp in brief.key_points
    )
    hooks_text = "\n".join(f"- {h}" for h in brief.hooks)

    return f"""Write an 8-slide Instagram carousel based on this video content brief.

Video title (may contain the speaker's name/affiliation for the "source" field — \
only use it if it's genuinely present, don't guess): {video_title}

Summary: {brief.summary}
Tone: {brief.tone}
Target audience: {brief.target_audience}

Key points available (each with a real supporting quote):
{key_points_text}

Possible hooks to inspire the cover slide (keep bold, but not more absolute than these):
{hooks_text}

Underlying message / call to action: {brief.call_to_action}

{SCHEMA_DESCRIPTION}
"""


def generate_carousel(brief: ContentBrief, video_title: str = "", api_key: Optional[str] = None) -> Carousel:
    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(brief, video_title)}],
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

    slides = [
        CarouselSlide(
            headline=s.get("headline", ""),
            subtext=s.get("subtext"),
            flow=s.get("flow"),
            source=s.get("source"),
        )
        for s in data.get("slides", [])
    ]
    return Carousel(slides=slides)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python carousel_generator.py <path_to_brief.json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        brief = ContentBrief.from_json(f.read())

    carousel = generate_carousel(brief)
    print(carousel.to_json())
