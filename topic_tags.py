"""
topic_tags.py

Turns a moment, a statement, or a bare topic into the two different kinds of
keyword a post needs — and they really are different jobs:

    hashtags        for the caption and post description  (publishing)
    visual queries  for the image search                  (sourcing)

Reaching for hashtags as image-search terms is the obvious shortcut and it
doesn't work. `#neuroplasticity` and `#growthmindset` return nothing usable
from an image API, because nobody photographs an abstraction and nobody tags a
photo that way. "runner at dawn" returns exactly what a poster needs. So both
come out of one call, from the same source, and get used for different things.

This is the same split that already makes `moment_finder.visual_keywords`
work — this module just makes it available for statements and bare topics too,
and adds the publishing half.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

# Haiku: this returns hashtags and a handful of search words. Sonnet was
# doing it 15 times a run at ten times the price for no visible gain.
MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = """You produce two separate keyword sets for a short-form video \
post. They serve different systems and must not be interchangeable.

HASHTAGS — for the caption. These are read by people and by the platform's \
recommendation system.
- 6-10 tags. Mix a few broad ones with a few specific to this exact idea.
- No # symbol in the strings.
- Casing must be consistent: either all lowercase (doThework is wrong, \
dothework is right) or clean CamelCase with every word capitalised \
(DoTheWork). Half-capitalised tags read as typos.
- Avoid spam tags (#viral, #fyp, #foryou) unless genuinely fitting.
- Abstractions are fine here: growthmindset, discipline, neuroplasticity.

VISUAL QUERIES — for a stock footage search. These are matched against real \
video, so they must describe things a camera can point at.

WORK FROM THE MEANING, NEVER FROM THE WORDS. This is the whole job.

- **Metaphors are not search terms.** When a teacher says "you have the lock, \
a teacher has the key", the key is not an object — it means guidance, or the \
mentality you build. Searching for "key" returns a photograph of a door key \
and ruins the shot. Ask what the speaker actually means, then film that: \
someone training alone, a student watching closely, hands working at something \
difficult.
- **Name the feeling first, then find footage that carries it.** Decide what \
the passage makes a viewer feel — peace, strain, resolve, isolation, clarity, \
dread — and choose images that produce that feeling. If the register is peace: \
still water at dawn, slow breath in cold air, an empty room with light coming \
in. If it is struggle: a dark gym, sweat, someone going again after failing.
- 4-6 queries, each 2-4 words.
- Concrete subject plus a setting or action: "boxer in empty gym", "runner at \
dawn", "hands gripping barbell", "empty chair at table".
- NEVER abstractions as the query itself. "success", "motivation", \
"determination" are unphotographable and return junk. The feeling decides \
which concrete thing you search for; it is not the search.
- Vary them. Six queries returning the same gym are worth one query.
- Match the register exactly: a passage about grief must not return a sunny \
beach, and one about stillness must not return a crowded street.

Respond with ONLY valid JSON, no preamble, no markdown code fences."""

SCHEMA_DESCRIPTION = """
Return a JSON object with exactly this shape:

{
  "means": string,                  // what the passage ACTUALLY says, in plain
                                    // words, with any metaphor resolved.
                                    // "you have the lock, a teacher has the key"
                                    // -> "you need guidance to unlock what is
                                    // already in you"
  "feeling": string,                // what it makes a viewer feel: peace,
                                    // strain, resolve, isolation, clarity
  "hashtags": [string, ...],        // 6-10, no # symbol
  "visual_queries": [string, ...],  // 4-6 concrete, filmable searches that
                                    // carry `feeling` - derived from `means`,
                                    // never from the literal nouns spoken
  "mood": string                    // 2-4 words describing the visual register,
                                    // e.g. "dark and still" or "bright, kinetic"
}
"""


@dataclass
class TopicTags:
    hashtags: list[str]
    visual_queries: list[str]
    mood: str = ""
    # Written before the queries so the model has to resolve the metaphor and
    # name the feeling first. Asking for footage straight from a hook produced
    # a photograph of a door key for a line about mentorship.
    means: str = ""
    feeling: str = ""

    def hashtag_line(self) -> str:
        return " ".join(f"#{tag.lstrip('#')}" for tag in self.hashtags)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["hashtag_line"] = self.hashtag_line()
        return data


def _clean(text) -> str:
    return " ".join(str(text or "").split()).strip()


def _normalise_tag(tag: str) -> str:
    """Fix a hashtag's casing, or leave clean CamelCase alone.

    The model reliably produces the words but not the capitalisation, and
    returns things like `doThework` and `noShortcuts` — half-capitalised, which
    reads as a typo rather than a style. Rather than guess where the word
    boundaries were meant to be (`doTheWork`? `doThework`?), anything that
    isn't already consistent gets lowercased. Lowercase is what most accounts
    use, and it can never look like a mistake.

    Deliberate CamelCase is preserved, since a tag like `GrowthMindset` is a
    legitimate choice and readable when every word is capitalised.
    """
    tag = _clean(tag).lstrip("#").replace(" ", "")
    if not tag:
        return ""
    if tag.islower() or tag.isupper():
        return tag
    # Clean CamelCase: every run of letters after the first starts with a
    # capital and the rest of that run is lower — e.g. GrowthMindset, DoTheWork.
    if re.fullmatch(r"(?:[A-Z][a-z0-9]*)+", tag):
        return tag
    return tag.lower()


def generate(
    topic: str,
    quote: str = "",
    theme: str = "",
    api_key: Optional[str] = None,
    context: str = "",
) -> TopicTags:
    """Tags and image queries for one idea.

    `topic` is the headline idea (a hook, a statement, or just a subject).
    `quote` and `theme` sharpen it when available.

    `context` says what world the idea comes from — the talk's title, the
    speaker, the channel. It matters more than it looks. Without it a line
    about consistency from a basketball player produced "athlete meal prep",
    which returned stock footage of vegetables being chopped: a fair reading of
    the words, and completely wrong for the video. Naming the domain keeps the
    footage in the same world as the speaker.
    """
    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    parts = [f"Idea: {topic}"]
    if quote:
        parts.append(f'Line from the source: "{quote}"')
    if theme:
        parts.append(f"Theme: {theme}")
    if context:
        parts.append(
            f"This comes from: {context}. "
            "Keep the visual queries inside this world. Footage from the wrong "
            "domain reads as stock filler even when it matches the words — a "
            "line about consistency from a basketball player wants basketball, "
            "not a generic kitchen."
        )
    parts.append(SCHEMA_DESCRIPTION)

    response = client.messages.create(
        model=MODEL,
        max_tokens=1200,
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

    return TopicTags(
        hashtags=[_normalise_tag(t) for t in data.get("hashtags", []) if _normalise_tag(t)],
        visual_queries=[_clean(q) for q in data.get("visual_queries", []) if _clean(q)],
        mood=_clean(data.get("mood")),
        means=_clean(data.get("means")),
        feeling=_clean(data.get("feeling")),
    )


def for_moment(moment, context: str = "") -> TopicTags:
    """Tags for a `moment_finder.Moment`, reusing the visuals it already has.

    The moment's own `visual_keywords` were written against the transcript, so
    they lead; anything new from this call is appended rather than replacing
    them.
    """
    tags = generate(
        topic=getattr(moment, "hook", "") or "",
        quote=getattr(moment, "quote", "") or "",
        theme=getattr(moment, "theme", "") or "",
        context=context,
    )
    existing = list(getattr(moment, "visual_keywords", []) or [])
    seen = {q.lower() for q in existing}
    tags.visual_queries = existing + [
        q for q in tags.visual_queries if q.lower() not in seen
    ]
    return tags


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Hashtags and image queries for a topic.")
    parser.add_argument("topic", help="the idea, hook, or subject")
    parser.add_argument("--quote", default="")
    parser.add_argument("--theme", default="")
    args = parser.parse_args()

    result = generate(args.topic, quote=args.quote, theme=args.theme)
    print(f"\n  mood: {result.mood}")
    print(f"\n  hashtags:\n    {result.hashtag_line()}")
    print("\n  visual queries (for image search):")
    for q in result.visual_queries:
        print(f"    - {q}")
