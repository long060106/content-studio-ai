"""
broll_picker.py

Chooses which clip sits under which spoken line, by meaning rather than by
shared words.

**The problem this solves, measured.** `asset_library.curated_broll` scores a
clip by counting words shared between the moment's visual queries and the
clip's filename. On a real batch, 65-90% of the 732-clip library scored *zero*
against any given moment, and only 3-25 clips scored two words or more. The
selector returns the best it has, so once the handful of real matches run out
the rest of the pool is arbitrary — and with ties broken at random, arbitrary is
literally what it is. One finished short about a man's retirement was cut over
hands holding a fish, a man holding flowers, and a neon parking lot. Nine of its
ten cutaways had matched on a single word, usually "man".

That is not a library problem and no amount of new footage fixes it. "Hands
handing over a set of keys" and `close-up-hands-envelope-desk` share one word by
luck; `oppn-23-close-up-elderly-man-mustache` and "a life's work behind him"
share none at all and is the better shot. Word overlap cannot see either fact.

**So the choice is made by a model that can read.** It gets the line actually
being spoken, the whole library as a list of descriptions, and picks — or
declines. That last part matters as much as the picking: `null` means stay on
the speaker, and a wrong cutaway is worse than none, because the viewer reads it
as a mistake and the speaker's face never is.

One call per short, not per line. The whole shot list goes in together, which is
what lets the model avoid repeating itself and keep a sequence that flows rather
than nine separately-reasonable choices that jump.
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-6"

# The library is sent as filenames. At roughly eight words each, the whole
# 732-clip library is a few thousand tokens — cheap enough that pre-filtering
# would cost more in lost candidates than it saves. Pre-filtering by word
# overlap is exactly the thing that fails here, so filtering the list *before*
# showing it to the model would reintroduce the bug this module exists to fix.
MAX_LIBRARY = 1500

# How many cutaways in one short may share a distinctive subject.
#
# A Brian Tracy talk produced a short with SIX corridors in forty-five seconds
# — six different films, no two adjacent, every one passing its mute test on its
# own. Nothing was checking the set. His language is abstract ("unstoppable",
# "responsibility", "capacity") with almost nothing filmable in it, and faced
# with abstraction the picker reaches for the same visual metaphor again and
# again: a corridor is this library's default for "a person alone with a
# thought".
#
# The existing sequence rules did not catch it because both were satisfied. The
# clips came from different films and were not adjacent. Repetition of SUBJECT
# is a third thing, and it is the one a viewer actually notices.
MAX_SAME_SUBJECT = 2

# Words too common in the library to count as a subject. Nearly every clip is a
# man, at night, in close-up, so capping on these would cap everything.
_UBIQUITOUS = {
    "man", "men", "woman", "women", "person", "people", "figure", "figures",
    "close", "closeup", "night", "day", "daytime", "daylight", "dark", "light",
    "lighting", "indoor", "indoors", "interior", "exterior", "silhouette",
    "view", "shot", "scene", "warm", "cold", "golden", "blue", "green", "red",
    "white", "black", "grey", "gray", "overcast", "dusk", "dawn", "sunset",
    "sunrise", "morning", "evening", "and", "the", "with", "mp4",
}


def _subjects(filename: str) -> set:
    """The distinctive nouns in a clip's description."""
    stem = re.sub(r"^\d+_", "", os.path.splitext(filename)[0].lower())
    parts = stem.split("-")
    # Drop the film prefix and the clip number, keep the description.
    body = parts[2:] if len(parts) > 2 else parts
    return {w for w in body if len(w) > 2 and w not in _UBIQUITOUS}

# Clips that have been judged wrong in a finished short, kept between runs.
#
# Without this the same mistake returns: `phml-12-hands-holding-fish-ocean` was
# picked under a line about reaching out, removed, and picked again on the next
# run for the same reason — the pull of "reach"/"hands" is in the words and does
# not go away because a human disliked the result once.
#
# A plain text file, one filename per line, `#` for comments, so it can be
# edited by hand while reviewing a batch. Blocked clips are withheld from the
# model entirely rather than penalised, because a penalty is a suggestion and
# this is a decision that has already been made.
BLOCKLIST_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "assets", "broll-blocklist.txt"
)


def blocked() -> set:
    """Filenames never to offer. Missing file means nothing is blocked."""
    try:
        with open(BLOCKLIST_PATH, encoding="utf-8") as f:
            names = set()
            for ln in f:
                # The note after `#` is for whoever reads the file; only the
                # filename before it is the entry. Keeping the whole line was a
                # silent no-op — every comparison against a real filename
                # failed and the blocklist blocked nothing.
                name = ln.split("#", 1)[0].strip()
                if name:
                    names.add(name)
            return names
    except OSError:
        return set()


def block(filename: str, reason: str = "") -> None:
    """Add a clip to the blocklist, with the reason beside it."""
    os.makedirs(os.path.dirname(BLOCKLIST_PATH), exist_ok=True)
    if filename in blocked():
        return
    with open(BLOCKLIST_PATH, "a", encoding="utf-8") as f:
        note = f"  # {reason}" if reason else ""
        f.write(f"{filename}{note}\n")

SYSTEM_PROMPT = """You choose b-roll for motivational shorts. You are given the \
lines a speaker says, and a library of clips described by filename. For each \
line you pick the clip that belongs under it, or decline.

WHAT MAKES A CUTAWAY BELONG. Not a shared word — a shared meaning. The filename \
describes what is visible in the clip; the line is what is being said over it. \
A cutaway works when the picture gives the words somewhere to land:

- LITERAL: the line names something the clip shows. "I lived in the back of an \
old car" over a car at night. Use this when it is available and not obvious to \
the point of being silly.
- EMOTIONAL: the clip carries the feeling of the line without illustrating it. \
"I buried a mother, father, sister, brother" over an empty room at dusk. This \
is most of good b-roll.
- CONTRAST: the picture pushes against the words in a way that sharpens them.

WHAT DOES NOT BELONG, and this is the common failure: a clip that is merely \
*present*. A man holding flowers under a line about retirement is not wrong \
about anything, it simply means nothing, and the viewer feels the edit go slack. \
If nothing in the library means anything for a line, return null and let the \
speaker hold the screen. Returning null is a correct answer and you should use \
it — an honest half of a shot list beats a full one of near-misses.

THE LITERAL-ASSOCIATION TRAP, which is the hardest failure to guard against. \
A word in the line matching an object in the clip is NOT a reason to use it. \
"Always reach out to better yourself" over `hands-holding-fish-ocean` is the \
canonical mistake: "reach out" and "hands" are the same idea in a dictionary and \
nothing alike on screen. The viewer does not hear a word and see an object, they \
see a man holding a fish while somebody talks about self-improvement, and the \
edit goes as slack as if the clip were chosen at random.

THE MUTE TEST, and you must WRITE IT OUT for every line rather than applying it \
in your head. Describe what a viewer sees with the sound off, then say whether \
the intent survives. An instruction to "watch for literal associations" does not \
work on its own — the pull of a matching word is strongest exactly when you are \
not looking at it, so the check has to become something you write down and read \
back. Three real failures caught this way:

- "you won't hardly know what's going on" over `underwater-silhouette-hands-blue-light`. \
Muted, that is a diver — not a man losing track of the world. FAILS.
- "sitting on a front porch" over `person-sitting-white-corridor`. Muted, that is \
someone in a corridor. The word "sitting" matched and nothing else did. FAILS.
- "I went to college" over `man-harbor-foggy-dawn`. Muted, that is a harbour. It \
carries no idea at all, it is merely present. FAILS.

All three should have been null, and null would have been better than any of \
them. If the only connection is a word, drop it and return null. Objects that \
turn up inside idioms — hands, doors, roads, keys, bridges, mountains, light, \
water — are where this goes wrong most, because there the idiom carries the \
meaning and the picture does not.

WRITE THE INTENT BEFORE YOU CHOOSE. For every line, say in `intent` what the \
shot has to DO — the feeling or the idea the picture must carry — before you \
name any clip. "Aspiration, something opening up" is an intent. "Hands" is not: \
it is already a search term, and writing one is how the literal trap gets in.

AN ABSTRACT TALK NEEDS DIFFERENT B-ROLL, NOT LESS OF IT. Some speakers deal in \
things you can film — a car, a blackboard, a harbour — and some deal in ideas: \
"unstoppable", "responsibility", "capacity", "potential". The second kind is \
harder and it is not a reason to stop cutting away.

**This version of the short is the one with b-roll in it.** A speaker-only cut \
is rendered separately and already exists, so leaving a line bare here does not \
produce a tasteful edit, it produces the plain version with extra steps. Decline \
when a clip would be actively wrong — that judgement still stands — but do not \
decline your way through an abstract passage.

What abstraction actually calls for is TEXTURE rather than illustration: ink \
dispersing in water, smoke, light through dust, embers, frost forming. Those \
carry a feeling without claiming to depict anything, which is exactly what a \
line like "you are totally responsible" needs. The library has such clips; \
reach for them before you reach for another figure in a room.

The failure to avoid is one visual metaphor over and over because nothing \
better comes to mind. One short came back with SIX corridors in forty-five \
seconds — six different films, none adjacent, each defensible alone, together a \
slideshow. A corridor is this library's default for "a person alone with a \
thought", and an abstract talk asks that of every line. Never use a third clip \
of the same subject — no third corridor, no third road, no third ocean — \
however well each tests on its own. Find a different kind of picture instead.

THE SEQUENCE MATTERS, not just each choice. You see the whole list at once, so:
- Never use the same clip twice.
- Do not put two clips from the same film back to back — the prefix before the \
first number is the film.
- Watch the light. Cutting from a bright daylight exterior to a black interior \
and back reads as a fault. Prefer runs that sit near each other tonally.
- Vary the shot size. Three wide landscapes in a row is a slideshow.

Respond with ONLY valid JSON, no preamble, no code fences."""


def _schema(n: int) -> str:
    return f"""
Return a JSON object with exactly this shape:

{{
  "picks": [
    {{
      "line": number,        // the line's index, 0 to {n - 1}
      "intent": string,      // what the shot must DO, written BEFORE choosing. A feeling or an
                             // idea, never an object or a search term.
      "candidate": string,   // the filename you are considering, before judging it
      "mute_test": string,   // THE CHECK, written out. What a viewer sees with the sound OFF,
                             // then whether the intent survives. Never just "yes".
      "passes": boolean,     // did it survive the mute test?
      "clip": string|null,   // the candidate if it passed. MUST be null when passes is false —
                             // there is no partial credit and no third option.
      "why": string          // a few words on what makes it belong. "" when null.
    }}
  ]
}}

Return exactly one entry per line, in order.
"""


def choose(
    lines: list[str],
    library: list[str],
    hook: str = "",
    theme: str = "",
    api_key: Optional[str] = None,
    model: str = MODEL,
    record_to: Optional[str] = None,
) -> dict[int, str]:
    """Map line index -> chosen filename. Absent keys mean stay on the speaker.

    Never raises for a bad answer: an unusable response returns an empty map and
    the caller falls back to its existing behaviour. A batch that renders with
    weaker b-roll is a much smaller problem than a batch that does not render.
    """
    key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not key or not lines or not library:
        return {}

    # Withhold the blocked clips, comparing without the copy prefix.
    #
    # Clips arrive here named as they are inside a short's folder —
    # `15_f1gp-20-...` — while the blocklist holds plain library names, so a
    # direct comparison never matches. That is the second silent no-op in this
    # one small feature; the first kept the `#` comment as part of the name.
    # Both looked like working code and blocked nothing.
    stop = blocked()
    library = [c for c in library if re.sub(r"^\d+_", "", c) not in stop]
    if not library:
        return {}

    listing = "\n".join(sorted(library)[:MAX_LIBRARY])
    numbered = "\n".join(f"{i}. {t}" for i, t in enumerate(lines))
    context = f"The short's hook: {hook}\nIts theme: {theme}\n\n" if hook else ""

    # The library moves into the system block, marked cacheable.
    #
    # It is ~8,700 tokens and identical on every call, and there is one call per
    # short — so a fifteen-short talk re-sent 130,000 tokens of the same
    # filenames. Caching matches an exact prefix, so the library has to sit in
    # the stable part of the request rather than beside the lines, which change
    # every time. Fourteen of those fifteen sends become cache reads.
    prompt = (
        f"{context}Lines, in the order they are spoken:\n{numbered}\n"
        f"{_schema(len(lines))}"
    )

    try:
        client = Anthropic(api_key=key)
        reply = client.messages.create(
            model=model,
            max_tokens=4000,
            system=[
                {"type": "text", "text": SYSTEM_PROMPT},
                {
                    "type": "text",
                    "text": f"The clip library ({len(library)} clips):\n{listing}",
                    "cache_control": {"type": "ephemeral"},
                },
            ],
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in reply.content if getattr(b, "text", ""))
    except Exception:
        return {}

    # The model is told not to fence its JSON and mostly doesn't; strip it when
    # it does rather than losing the whole answer to a decorative markdown line.
    text = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}

    # Keep the reasoning next to the short.
    #
    # The mute test is the whole mechanism, and a mechanism you cannot read is
    # one you have to take on faith. Written beside the clips it produced, a bad
    # cutaway can be traced to the sentence that justified it — and a check that
    # was performed rather than reasoned through is obvious at a glance.
    if record_to:
        try:
            os.makedirs(os.path.dirname(record_to) or ".", exist_ok=True)
            with open(record_to, "w", encoding="utf-8") as f:
                json.dump({"lines": lines, "picks": data.get("picks", [])},
                          f, indent=2, ensure_ascii=False)
        except OSError:
            pass

    valid = set(library)
    out: dict[int, str] = {}
    seen_subjects: dict[str, int] = {}
    for pick in data.get("picks", []):
        try:
            i = int(pick.get("line"))
        except (TypeError, ValueError):
            continue
        # A written mute test that ends in failure beats a `clip` that
        # contradicts it. The model sometimes reasons its way to FAILS and then
        # fills the slot anyway — the pull toward a complete shot list is
        # strong — so the verdict it wrote is honoured over the choice it made.
        if pick.get("passes") is False:
            continue
        clip = pick.get("clip")
        # Only accept a name that is actually in the library. A hallucinated
        # filename would otherwise reach the renderer as a missing input and
        # fail the whole short.
        if isinstance(clip, str) and clip in valid and 0 <= i < len(lines):
            # Refuse a subject that has already had its turn. Dropping the
            # cutaway is right rather than substituting one: the picker judged
            # nothing else fitted this line, and the speaker's face is never
            # the wrong shot. Six corridors is worse than three corridors and
            # three lines of speaker.
            subjects = _subjects(clip)
            if any(seen_subjects.get(w, 0) >= MAX_SAME_SUBJECT for w in subjects):
                continue
            for w in subjects:
                seen_subjects[w] = seen_subjects.get(w, 0) + 1
            out[i] = clip
    return out
