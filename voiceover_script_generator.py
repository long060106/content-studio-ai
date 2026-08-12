"""
voiceover_script_generator.py

Generates a narration script from a video's content brief, written to be
read aloud by a text-to-speech voice rather than read on a page: contractions,
natural spoken rhythm, no markdown or visual formatting.
"""

from __future__ import annotations

import os
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

from content_brief import ContentBrief

load_dotenv()

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a scriptwriter who turns video content briefs into short \
voice-over narration scripts. This text will be read aloud by a text-to-speech \
voice, not read on a page — write for the EAR, not the eye.

STYLE
- Natural spoken rhythm: contractions ("it's", "don't"), varied sentence length, \
the way someone would actually talk explaining something interesting to a friend.
- No markdown, no headers, no bullet points, no bold/italic syntax, no emojis — \
none of that translates to speech.
- Avoid sentence constructions that are awkward when spoken aloud (long \
parenthetical asides, complex nested clauses).

ACCURACY
- Never state a claim more absolutely than the brief supports.
- Only use facts and quotes explicitly present in the brief. Never invent \
statistics, studies, or quotes.
- Distinguish the speaker's actual claim from your own framing of it.

LENGTH
Keep this short: 100-150 words (roughly 40-60 seconds of narration). This is \
intentionally brief for quick, low-cost testing — longer scripts can be \
generated later once the voice and pacing are confirmed to sound right.

Output ONLY the narration script as plain text — no preamble, no scene \
directions, no "[pause]" markers, no meta-commentary."""


def _build_user_prompt(brief: ContentBrief) -> str:
    key_points_text = "\n".join(f'- {kp.get("point", "")}' for kp in brief.key_points)

    return f"""Write a short voice-over narration script based on this video content brief.

Summary: {brief.summary}
Tone: {brief.tone}
Target audience: {brief.target_audience}

Key points available (pick the most interesting 1-2 — don't try to cover everything in this short script):
{key_points_text}

Call to action / underlying message: {brief.call_to_action}

Write the script now, following the style and length rules above.
"""


def generate_voiceover_script(brief: ContentBrief, api_key: Optional[str] = None) -> str:
    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(brief)}],
    )

    return response.content[0].text.strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python voiceover_script_generator.py <path_to_brief.json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        brief = ContentBrief.from_json(f.read())

    print(generate_voiceover_script(brief))
