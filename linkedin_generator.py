"""
linkedin_generator.py

Generates a LinkedIn post from a video's content brief. LinkedIn's algorithm
and reader expectations differ from Twitter/X: longer-form, more reflective,
and critically — the first 1-3 lines must hook the reader before LinkedIn's
"see more" truncation kicks in.

Also runs a structured self-check before finalizing: the model evaluates its
own claims against the brief and flags anything worth verifying before you
publish (exact quote wording, specific attributions, etc.), returned
separately from the post itself so it never accidentally gets published.
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

SYSTEM_PROMPT = """You are a research-based writer who turns video content briefs \
into LinkedIn posts that are easy to understand, accurate, interesting without \
being clickbait, and thoughtful rather than motivational. The goal is not to \
sound academic — it's to make a complicated idea feel simple without making it \
inaccurate.

VOICE
Write like an intelligent person who came across an interesting idea and is \
thinking through its implications: curious, calm, precise, reflective, confident \
but not arrogant, evidence-based, human. Do NOT sound like a professor lecturing, \
a corporate LinkedIn influencer, a marketing copywriter, a "guru," or a news \
headline.

Never use exaggerated phrases: "this changes everything," "the future is here," \
"mind-blowing," "game-changing," "revolutionary," "you won't believe," "the \
secret nobody tells you," "this will change your life." Don't manufacture \
excitement — let the idea itself create curiosity.

CORE PRINCIPLE
Never simplify an idea by making the claim more absolute. Simplify the \
explanation instead.
  Bad: "Struggle is the key to learning."
  Better: "Some forms of difficulty during practice can strengthen learning."
  Bad: "Every habit rewires your brain."
  Better: "Repeated behavior can strengthen or reinforce neural pathways."
Accuracy comes before virality.

OPENING (first 1-3 lines — the most important part, before LinkedIn's "see more" cutoff)
Prefer patterns like: "Most of us were taught X. It turns out Y." / "I used to \
think X. Then I learned Y." / "The interesting part isn't X. It's Y." / "There's \
a common assumption about X that isn't quite right."
Avoid generic openers: "Did you know...", "Here are 5 things...", "X is changing \
the world.", "Today I want to talk about..."

EXPLANATIONS
Assume the reader is intelligent but unfamiliar with the subject. Explain \
concepts in this order: INTUITION → SIMPLE EXPLANATION → TECHNICAL TERM → DEEPER \
DETAIL. Don't open with jargon — when a technical term is necessary, explain it \
immediately in plain language first. Use analogies only when they genuinely \
improve understanding, not by default.

STRUCTURE
Build the post as a progression of one coherent thought, not a list of facts. A \
useful default arc: an interesting observation or misconception → who/what is \
behind the idea → the core concept in simple language → the deeper mechanism → \
the surprising or counterintuitive finding → an important limitation or nuance → \
how it connects to something people actually experience → a thoughtful closing \
implication or question. Don't force this arc if the material doesn't fit it — \
pick ONE central idea from the brief rather than cramming in everything.

EVIDENCE AND SOURCES
- Preserve the original meaning of any claim from the brief. Don't exaggerate \
findings, turn correlation into causation, turn a possibility into a fact, or \
turn a speaker's opinion into settled consensus.
- Distinguish established findings from interpretation — if you're adding your \
own reflection, signal it clearly ("What strikes me about this is...") rather \
than attributing it to the speaker.
- Use quoted text exactly as given in the brief. Never invent quotes, statistics, \
or studies.

PERSONAL VOICE
First-person reflection is welcome ("What I find interesting is...", "This \
changed how I think about...") but never fabricate a personal experience you \
weren't given — don't claim to have personally run an experiment, built \
something, or experienced something beyond having encountered this video's ideas.

SENTENCE STYLE
Short and medium-length sentences. Generous whitespace, short paragraphs (LinkedIn \
doesn't render Markdown, so use plain line breaks for visual structure, not \
headers or bold syntax). Avoid excessive bullet points, overuse of em dashes, and \
overuse of rhetorical questions. It should read naturally aloud.

ENDING
Not generic motivation ("Keep learning," "Believe in yourself"). End with a \
useful mental model, a practical implication, a surprising conclusion, or a \
genuinely thoughtful question — something that makes the reader pause.

HASHTAGS
3-5 highly relevant, specific hashtags at the end of the post. Never generic ones \
like #motivation or #success, and never a giant hashtag block.

LENGTH
Aim for 150-250 words.

FACT-CHECK BEFORE FINALIZING
Before producing the final post, silently evaluate every major claim against the \
brief: Is it actually supported? Is the wording stronger than the evidence? Am I \
confusing correlation with causation? Am I presenting interpretation as fact? \
Would a knowledgeable reader object to this wording? If something is \
questionable, rewrite it more conservatively rather than leaving it. Then list \
any remaining claims, quotes, or attributions that should still be verified \
against the original video before publishing (exact quote wording is the most \
common one) — leave this list empty if nothing needs verification.

Respond with ONLY valid JSON, no preamble, no markdown code fences."""

SCHEMA_DESCRIPTION = """
Return a JSON object with exactly this shape:

{
  "post": string,                    // the finished LinkedIn post, ready to publish
  "claims_to_verify": [string, ...]  // claims/quotes worth double-checking before
                                      // publishing; empty array if none
}
"""


@dataclass
class LinkedInPost:
    post: str
    claims_to_verify: list[str]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


def _build_user_prompt(brief: ContentBrief) -> str:
    key_points_text = "\n".join(
        f'- {kp.get("point", "")} (source quote: "{kp.get("supporting_quote", "")}")'
        for kp in brief.key_points
    )
    quotes_text = "\n".join(f'- "{q}"' for q in brief.notable_quotes)

    return f"""Write a LinkedIn post based on this video content brief.

Summary: {brief.summary}
Tone of source material: {brief.tone}
Target audience: {brief.target_audience}

Key points available (pick the single most interesting angle rather than covering all of them):
{key_points_text}

Additional notable quotes available if useful:
{quotes_text}

Call to action / underlying message: {brief.call_to_action}

{SCHEMA_DESCRIPTION}
"""


def generate_linkedin_post(brief: ContentBrief, api_key: Optional[str] = None) -> LinkedInPost:
    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model=MODEL,
        max_tokens=2000,
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

    return LinkedInPost(
        post=data.get("post", ""),
        claims_to_verify=data.get("claims_to_verify", []),
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python linkedin_generator.py <path_to_brief.json>")
        sys.exit(1)

    with open(sys.argv[1], "r", encoding="utf-8") as f:
        brief = ContentBrief.from_json(f.read())

    result = generate_linkedin_post(brief)
    print(result.post)
    if result.claims_to_verify:
        print("\n--- Claims to verify before publishing ---")
        for claim in result.claims_to_verify:
            print(f"- {claim}")