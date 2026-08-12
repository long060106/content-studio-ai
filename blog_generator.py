"""
blog_generator.py

Generates a publish-ready blog post (in Markdown) from a video's content
brief. Reads the distilled brief rather than the raw transcript, so output
is well-structured and consistent with the other formats generated from the
same brief (Twitter thread, LinkedIn post, etc.).
"""

from __future__ import annotations

import os
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

from content_brief import ContentBrief

load_dotenv()

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a skilled content writer who turns video content briefs \
into engaging, well-structured blog posts. Write in clear, natural prose — not a \
dry summary of the brief. Use the brief's key points and quotes as source material, \
but write original sentences; don't just restate the brief verbatim. Structure with \
a strong opening hook, clear H2 section headings, natural paragraph breaks, and a \
short closing with a call to action. Output ONLY the blog post in Markdown — no \
preamble, no meta-commentary, no "Here's your blog post" framing."""


def _build_user_prompt(brief: ContentBrief, video_title: str) -> str:
    key_points_text = "\n".join(
        f'- {kp.get("point", "")} (supporting quote: "{kp.get("supporting_quote", "")}")'
        for kp in brief.key_points
    )
    quotes_text = "\n".join(f'- "{q}"' for q in brief.notable_quotes)
    titles_text = ", ".join(brief.title_suggestions)

    return f"""Write a blog post based on this video content brief.

Video title: {video_title}
Summary: {brief.summary}
Target audience: {brief.target_audience}
Tone: {brief.tone}

Key points:
{key_points_text}

Notable quotes available (use sparingly, only if they genuinely strengthen a point):
{quotes_text}

Call to action for the closing: {brief.call_to_action}

Suggested title options (pick the best, adapt it, or write your own that fits better):
{titles_text}

Aim for roughly 500-800 words. Format as Markdown starting with a single H1 title.
"""


def generate_blog_post(brief: ContentBrief, video_title: str = "", api_key: Optional[str] = None) -> str:
    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model=MODEL,
        max_tokens=3000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(brief, video_title)}],
    )

    return response.content[0].text.strip()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python blog_generator.py <path_to_brief.json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        brief = ContentBrief.from_json(f.read())

    post = generate_blog_post(brief)
    print(post)
