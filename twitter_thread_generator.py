"""
twitter_thread_generator.py

Generates a Twitter/X thread from a video's content brief. Enforces the
280-character-per-tweet limit and returns a structured, numbered thread
ready to post tweet-by-tweet (or paste into a scheduler like Buffer/Typefully).
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
MAX_TWEET_LENGTH = 280

SYSTEM_PROMPT = """You are a skilled science and technical communicator who turns \
video content briefs into accurate, engaging Twitter/X threads.

WRITING PHILOSOPHY
Write like a smart, curious friend explaining something fascinating — not like a \
professor lecturing or a marketer hyping. The reader should finish thinking \
"I finally understand this." Prioritize clarity over sophistication, curiosity \
over formality, concrete examples over abstract claims, short sentences over \
dense paragraphs. Never use jargon just to sound sophisticated — if a technical \
term is necessary, introduce the idea in plain language first, then name the term.

TONE
Conversational, curious, confident but humble, genuinely interested in the actual \
idea. Avoid corporate language, hype buzzwords ("revolutionary", "game-changing", \
"will never be the same"), empty motivational statements, excessive emojis, and \
more than 0-2 hashtags total (only on the last tweet, if any).

ACCURACY — this is the most important rule
- Never state a claim more absolutely than the source material supports. Avoid \
words like "only", "always", "every", "the primary driver", "literally" unless \
the brief genuinely supports that strength of claim. Watch especially for \
sweeping constructions like "every fact, every skill, every habit does X" — \
these sound punchy but are almost never literally true.
- If a claim needs nuance (multiple contributing factors, correlation vs. \
causation, etc.), include that nuance briefly rather than flattening it into a \
false absolute. Simplify the EXPLANATION, not the CLAIM itself.
- When something involves several interacting mechanisms or factors, don't force \
them into a neat one-to-one mapping (e.g. "X = short-term, Y = long-term") unless \
the brief explicitly supports that clean a split. Present them as related pieces \
that interact, not separate boxes.
- Avoid vague mechanism qualifiers like "mostly chemical" or "mostly X" unless \
the brief directly supports that specific attribution.
- When a claim, rule of thumb, or finding is described as limited or debated, \
give the most accurate and central reason why — not just the first plausible-\
sounding reason. If the brief specifies the reason, use that one.
- Clearly distinguish the speaker's actual claim from the writer's own extension \
or interpretation of it. When drawing out an implication the speaker didn't \
state directly, signal it explicitly ("[Speaker] argues this points toward...", \
"one implication of this is...") rather than presenting the extension as an \
established fact in the speaker's voice. This applies especially to \
philosophical or interpretive statements (e.g. framing a finding as a moral or \
life lesson) — mark these as your own reading, not as something the research \
itself established.
- Don't state predictions or future directions as settled fact. Instead of \
"this is where the field is headed," say what's actually true now: e.g. \
"researchers are increasingly interested in..." or "this raises the question of..."
- Only use quotes, facts, and attributions explicitly present in the provided \
brief. Never invent statistics, studies, or quotes. Use quoted text exactly as \
given in the brief — don't paraphrase something and present it in quotation marks.

INTERNAL CONSISTENCY
Before finalizing, check the thread's claims against each other. If one tweet \
establishes a nuance (e.g. "not all difficulty helps — it has to be productive \
difficulty"), a later tweet must not quietly revert to the flattened absolute \
version of the same idea (e.g. a closing line implying all difficulty is \
automatically good). The final tweet especially must stay consistent with any \
nuance established earlier — end with something actionable, not a rhetorical \
flourish that re-introduces an overclaim you already corrected.

HOOK (first tweet)
Must create genuine curiosity or challenge a common assumption — not a dramatic, \
clickbait-y claim. Avoid generic openers like "a thread on X" and avoid dramatic \
framing like "...forever" or "...will never be the same again."

STRUCTURE & THE "AHA" MOMENT
Don't just list points in order — build toward one clear, memorable mental model \
that ties the ideas together, usually 1-2 tweets before the final takeaway. A \
strong aha moment often uses a short parallel structure (e.g. "It doesn't know X. \
It just does Y.") that crystallizes everything before it into something shareable \
on its own.

VARY YOUR TRANSITIONS
Don't repeatedly open tweets with the same connector ("Here's...", "And...", \
"So...", "Worth noting..."). Mix it up: "But there's a catch.", "Now consider \
the opposite.", "This is where it gets interesting.", "And here's the \
uncomfortable part.", or simply start with the content itself.

HARD CONSTRAINT
Each tweet must be under 280 characters INCLUDING spaces and punctuation — count \
carefully. Respond with ONLY valid JSON, no preamble, no markdown code fences."""

SCHEMA_DESCRIPTION = """
Return a JSON object with exactly this shape:

{
  "tweets": [string, string, ...]   // 8-12 tweets, each under 280 characters
}

Structure to follow:
- Tweet 1: the hook — a strong, genuinely curious claim (not clickbait)
- Tweets 2-3: the problem or common misconception the video addresses
- Tweets 4-7: the core idea, explained progressively — intuition first, then the
  technical reality, with a concrete example or analogy (draw on the brief's key
  points and supporting quotes; keep the speaker's claims and your own
  extensions of them clearly distinguishable)
- 1-2 tweets: a counterintuitive implication or important nuance — "here's the
  catch"
- 1 tweet: the "aha" moment — a memorable mental model that ties it together
- Final tweet: the single most useful practical takeaway, with an optional soft
  call-to-action
"""


@dataclass
class TwitterThread:
    tweets: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)

    def to_text(self) -> str:
        """Numbered, human-readable format — easy to copy-paste tweet by tweet."""
        header = (
            "# Reminder: this thread quotes real people/claims from an auto-generated\n"
            "# transcript brief. Verify exact quote wording and key facts against the\n"
            "# original video before posting. (This header is a note, not a tweet.)\n"
            "# ---\n\n"
        )
        blocks = [f"{i}/{len(self.tweets)}\n{tweet}" for i, tweet in enumerate(self.tweets, start=1)]
        return header + "\n\n".join(blocks)


def _build_user_prompt(brief: ContentBrief) -> str:
    key_points_text = "\n".join(
        f'- {kp.get("point", "")} (source quote: "{kp.get("supporting_quote", "")}")'
        for kp in brief.key_points
    )
    hooks_text = "\n".join(f"- {h}" for h in brief.hooks)
    quotes_text = "\n".join(f'- "{q}"' for q in brief.notable_quotes)

    return f"""Write a Twitter/X thread based on this video content brief.

Summary: {brief.summary}
Tone: {brief.tone}
Target audience: {brief.target_audience}

Key points to cover across the thread (each with a real supporting quote — use these, don't invent your own):
{key_points_text}

Additional notable quotes available if useful:
{quotes_text}

Possible hooks to inspire (or improve on) the opening tweet:
{hooks_text}

Call to action for the final tweet: {brief.call_to_action}

{SCHEMA_DESCRIPTION}
"""


def _repair_oversized_tweets(
    client: Anthropic, tweets: list[str], indices: list[int]
) -> dict[int, str]:
    """
    Sends only the oversized tweets back for a targeted rewrite under the
    limit, preserving meaning/tone. Returns {index: rewritten_tweet}.
    """
    oversized_text = "\n".join(
        f'{i}: (currently {len(tweets[i])} chars) "{tweets[i]}"' for i in indices
    )

    repair_prompt = f"""These tweets exceed Twitter/X's 280-character limit. Rewrite \
each one to be under 270 characters (leaving a safety margin), preserving the \
original meaning, tone, and any factual claims exactly — don't add new claims or \
soften/strengthen anything, just tighten the wording.

{oversized_text}

Return ONLY valid JSON in this shape, same order, no preamble:
{{"tweets": [string, ...]}}
"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system="You are a precise editor who tightens text to fit a hard character limit without changing its meaning.",
        messages=[{"role": "user", "content": repair_prompt}],
    )

    raw_text = response.content[0].text.strip()
    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
        raw_text = raw_text.strip()

    try:
        data = json.loads(raw_text)
        rewritten = data.get("tweets", [])
    except json.JSONDecodeError:
        rewritten = []

    result = {}
    for pos, idx in enumerate(indices):
        if pos < len(rewritten) and rewritten[pos]:
            result[idx] = rewritten[pos]
    return result


def _hard_truncate(tweet: str, limit: int = MAX_TWEET_LENGTH) -> str:
    """Last-resort safety net: truncate at the last word boundary that fits, with an ellipsis."""
    if len(tweet) <= limit:
        return tweet
    cutoff = limit - 1  # room for the ellipsis character
    truncated = tweet[:cutoff].rsplit(" ", 1)[0]
    return truncated + "…"


def generate_twitter_thread(brief: ContentBrief, api_key: Optional[str] = None) -> TwitterThread:
    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model=MODEL,
        max_tokens=3000,
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

    tweets = data.get("tweets", [])

    oversized_indices = [i for i, t in enumerate(tweets) if len(t) > MAX_TWEET_LENGTH]
    if oversized_indices:
        print(f"  ⚠ {len(oversized_indices)} tweet(s) exceeded {MAX_TWEET_LENGTH} chars — rewriting to fit...")
        try:
            repaired = _repair_oversized_tweets(client, tweets, oversized_indices)
            for idx, new_text in repaired.items():
                tweets[idx] = new_text
        except Exception as e:
            print(f"  ⚠ Repair pass failed ({e}); falling back to truncation for oversized tweets.")

        # Safety net: anything still too long (repair failed or model still overshot) gets hard-truncated.
        still_oversized = [i for i in oversized_indices if len(tweets[i]) > MAX_TWEET_LENGTH]
        for idx in still_oversized:
            tweets[idx] = _hard_truncate(tweets[idx])
        if still_oversized:
            print(f"  ⚠ {len(still_oversized)} tweet(s) needed hard truncation as a last resort.")

    return TwitterThread(tweets=tweets)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python twitter_thread_generator.py <path_to_brief.json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        brief = ContentBrief.from_json(f.read())

    thread = generate_twitter_thread(brief)
    print(thread.to_text())