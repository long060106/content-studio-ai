"""
statement_writer.py

Writes short motivational statements in the two-part "antithesis" form that
poster accounts run on:

    be you
    not them.

    stick to the plan,
    not your mood.

The shape is always the same. A **setup** — what to do, be, or choose — then a
**payoff** that names the thing being rejected. The payoff is the line that
lands, so it gets the bigger type in `poster_renderer.py`.

What makes these work, and what the prompt below enforces:
  - The two halves must genuinely oppose each other. "work hard, not soft" is
    noise; "stick to the plan, not your mood" names a real, specific temptation.
  - Concrete over abstract. "your mood", "them", "the group chat" beat
    "negativity" and "failure".
  - Short. Anything past four words a line stops reading as a poster.
  - Lowercase, ending in a full stop. That's the house style of the genre.

Statements can be written from scratch on a theme, or derived from the moments
`moment_finder.py` already pulled out of a talk — which keeps a poster set and
a batch of shorts saying the same thing.
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

THEMES = [
    "discipline", "ownership", "failure", "fear", "consistency",
    "identity", "focus", "growth", "resilience", "purpose",
]

SYSTEM_PROMPT = """You write the text on motivational posters. Your entire craft \
is the two-part antithesis:

    setup   — what to do, be, or choose  (max 4 words)
    payoff  — what to reject             (max 4 words, almost always starts with "not")

Rules, all of them hard:
- The two halves must actually oppose each other. The payoff names the specific \
temptation the setup defeats. If the payoff could be swapped into any other \
statement, it's wrong.
- Concrete beats abstract. "your mood", "them", "the highlight reel", "3am you" \
land. "negativity", "failure", "obstacles", "excuses" are dead words — avoid them.
- Ruthlessly short. Four words per line is the ceiling; two or three is better.
- Lowercase throughout. The payoff ends with a full stop. The setup ends with a \
comma if it reads as one sentence, nothing if it stands alone.
- Second person or bare imperative. Never "we". Never a brand voice.
- No hashtags, emoji, quotation marks, or exclamation marks.
- No rhyming, no wordplay for its own sake, no "grind"/"hustle"/"beast mode".
- Don't moralise or explain. The reader gets it or they don't.

Respond with ONLY valid JSON, no preamble, no markdown code fences."""

SCHEMA_DESCRIPTION = """
Return a JSON object with exactly this shape:

{
  "statements": [
    {
      "setup": string,             // e.g. "stick to the plan," — lowercase, <= 4 words
      "payoff": string,            // e.g. "not your mood." — lowercase, <= 4 words, ends with a full stop
      "theme": string,             // one of: discipline, ownership, failure, fear, consistency,
                                   // identity, focus, growth, resilience, purpose
      "visual_keywords": [string], // 3-4 concrete, filmable search terms for the photo behind it
                                   // ("boxer in empty gym", "runner at dawn"), never abstractions
      "note": string               // one short line: what temptation this names, so a human can
                                   // judge whether it's true or just neat
    }
  ]
}
"""


@dataclass
class Statement:
    setup: str
    payoff: str
    theme: str
    visual_keywords: list[str]
    note: str = ""

    @property
    def full_text(self) -> str:
        return f"{self.setup} {self.payoff}".strip()

    def to_dict(self) -> dict:
        return asdict(self)


def _clean(text: str) -> str:
    return " ".join(str(text or "").split()).strip().strip('"').strip("'")


def _build_user_prompt(
    count: int,
    theme: Optional[str],
    source_quotes: Optional[list[str]],
) -> str:
    parts = [f"Write {count} statements."]

    if theme:
        parts.append(f"All of them on the theme: {theme}.")
    else:
        parts.append("Vary the theme across the set — don't write ten takes on discipline.")

    if source_quotes:
        quoted = "\n".join(f"- {q}" for q in source_quotes[:12])
        parts.append(
            "Draw them from the ideas in these lines from a talk. Keep the idea, "
            "throw away the wording — the statement has to stand alone as a "
            "poster, not read as a quotation:\n" + quoted
        )

    parts.append(
        "Before you answer, check each one: could the payoff be dropped into a "
        "different statement without anyone noticing? If so, rewrite it."
    )
    parts.append(SCHEMA_DESCRIPTION)
    return "\n\n".join(parts)


def write_statements(
    count: int = 8,
    theme: Optional[str] = None,
    source_quotes: Optional[list[str]] = None,
    api_key: Optional[str] = None,
) -> list[Statement]:
    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": _build_user_prompt(count, theme, source_quotes)}],
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

    statements: list[Statement] = []
    for item in data.get("statements", []):
        setup, payoff = _clean(item.get("setup")), _clean(item.get("payoff"))
        if not payoff:
            continue
        statements.append(Statement(
            setup=setup,
            payoff=payoff,
            theme=_clean(item.get("theme")).lower(),
            visual_keywords=[_clean(k) for k in item.get("visual_keywords", []) if _clean(k)],
            note=_clean(item.get("note")),
        ))
    return statements


def statements_from_moments(moments: list, count: Optional[int] = None) -> list[Statement]:
    """Write statements off the moments already pulled from a talk."""
    quotes = [m.quote for m in moments if getattr(m, "quote", "")]
    hooks = [m.hook for m in moments if getattr(m, "hook", "")]
    return write_statements(count or max(4, len(moments) * 2), source_quotes=quotes + hooks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Write motivational poster statements.")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--theme", choices=THEMES)
    parser.add_argument("--out", help="write the set to this JSON file")
    args = parser.parse_args()

    found = write_statements(count=args.count, theme=args.theme)
    for s in found:
        print(f"\n  {s.setup}")
        print(f"  {s.payoff}")
        print(f"    [{s.theme}] {s.note}")
        print(f"    visuals: {', '.join(s.visual_keywords)}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"statements": [s.to_dict() for s in found]}, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(found)} statements to {args.out}")
