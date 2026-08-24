"""
comment_prompt.py

Writes the comment the creator posts under their own short: an open question
that gives people something to argue about.

This is a small piece of text doing a specific job. A short arrives on a feed
with no thread under it, and the first comment decides whether one forms. Left
to the audience, the first comment is usually praise for the speaker — which
reads well and generates nothing, because there is no reply to praise. A
question posted by the account owner, pinned, gives every later viewer an
obvious thing to do.

What makes one work is narrower than it looks, and most of this module is that
constraint written down:

- **It must be answerable in one line, from memory.** Anything requiring
  thought, expertise, or a considered position gets scrolled past. "What's the
  boring thing you do every day that actually changed you?" is answerable
  immediately. "What is discipline, really?" is an essay prompt.
- **It must not be answerable yes or no.** A closed question produces a
  one-word comment and no replies to it.
- **It must be about the viewer, not the speaker.** Asking what people thought
  of the talk produces reviews. Asking what they had to unlearn produces
  stories, and stories get replies.
- **It must not be engagement bait.** "Comment below!", "Tag a friend", "Do you
  agree?" are recognisable as manipulation and cost the account credibility.

The strongest questions invite either a *story* or a *side*. A story ("who gave
you the push you needed?") brings people who want to be heard. A side ("which is
harder — starting, or starting again?") brings people who want to be right.
Both fill a thread; they attract different halves of an audience, which is why
two options come back rather than one.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-6"

# Comments are truncated in the feed long before they are truncated by the
# platform. A question that needs a "read more" tap has already lost.
MAX_CHARS = 120

SYSTEM_PROMPT = """You write the first comment a creator posts under their own \
short video. Its only job is to start a thread of replies.

Write ONE open question, plus one alternative taking a different angle.

WHAT THE QUESTION MUST DO

- Be answerable in a single line, instantly, from the person's own memory. No \
research, no expertise, no working it out.
- Be impossible to answer with yes or no. Never start with Do/Are/Have/Is/Can/\
Would/Should.
- Be about the viewer's own life, not about the speaker or the video. Nobody \
replies to a review; they reply to someone's story.
- Sound like a person asking, not a brand running a campaign.

WHAT IT MUST NEVER BE

- Engagement bait: "Comment below", "Tag a friend", "Drop a ❤️", "Do you \
agree?", "Thoughts?". These are recognisable and they cost credibility.
- Abstract or philosophical: "What does discipline mean to you?" is an essay \
prompt, and gets no answers.
- A restatement of the video's point with a question mark on the end.
- Longer than about 120 characters.

THE TWO ANGLES

Return two questions that pull on different people:

1. A STORY question — invites someone to tell you what happened to them. \
"What's something you had to unlearn before you got good at it?" People answer \
these because they want to be heard.

2. A SIDE question — puts two real options against each other so people \
disagree. "Which is harder: starting, or starting again?" People answer these \
because they want to be right. Both options must be genuinely defensible; a \
false choice reads as a trick.

Ground both in the specific idea of THIS clip. A question that would fit any \
motivational video is a wasted comment.

Respond with ONLY valid JSON, no preamble, no markdown code fences."""

SCHEMA_DESCRIPTION = """
Return a JSON object with exactly this shape:

{
  "question": string,   // the story question - the one to post by default
  "alternate": string,  // the side question - a different angle on the same idea
  "why": string         // one short line: what kind of reply you expect, and
                        // from whom. This is for the person deciding whether to
                        // post it, not for posting.
}
"""


@dataclass
class CommentPrompt:
    question: str
    alternate: str = ""
    why: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_text(self) -> str:
        """The file the poster actually reads, question first."""
        lines = [self.question]
        if self.alternate:
            lines += ["", "Another angle:", self.alternate]
        if self.why:
            lines += ["", f"({self.why})"]
        return "\n".join(lines).strip() + "\n"


def _clean(text) -> str:
    return " ".join(str(text or "").split()).strip()


def generate(
    hook: str,
    quote: str = "",
    theme: str = "",
    means: str = "",
    context: str = "",
    api_key: Optional[str] = None,
) -> CommentPrompt:
    """The question to post under one short.

    `means` is the clip's point with any metaphor already resolved — the same
    field `topic_tags` produces for choosing footage. It matters here for the
    same reason it matters there: a question built from the literal words of
    "you have the lock, a teacher has the key" asks people about keys, which is
    not what the clip is about.
    """
    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    parts = [f"The clip's point: {hook}"]
    if quote:
        parts.append(f'What the speaker says: "{quote}"')
    if means:
        parts.append(f"Put plainly, with any metaphor resolved: {means}")
    if theme:
        parts.append(f"Theme: {theme}")
    if context:
        parts.append(f"From: {context}")
    parts.append(SCHEMA_DESCRIPTION)

    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": "\n\n".join(parts)}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON. Raw output:\n{raw}") from e

    return CommentPrompt(
        question=_clean(data.get("question")),
        alternate=_clean(data.get("alternate")),
        why=_clean(data.get("why")),
    )


def for_moment(moment, means: str = "", context: str = "") -> CommentPrompt:
    """The question for a `moment_finder.Moment`."""
    return generate(
        hook=getattr(moment, "hook", "") or "",
        quote=getattr(moment, "quote", "") or "",
        theme=getattr(moment, "theme", "") or "",
        means=means,
        context=context,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="The comment to post under a short.")
    parser.add_argument("hook", help="the clip's point")
    parser.add_argument("--quote", default="")
    parser.add_argument("--theme", default="")
    parser.add_argument("--means", default="")
    args = parser.parse_args()

    result = generate(args.hook, quote=args.quote, theme=args.theme, means=args.means)
    print(f"\n  post this:\n    {result.question}")
    print(f"\n  or this:\n    {result.alternate}")
    print(f"\n  why: {result.why}\n")
