"""
moment_finder.py

Finds the moments in a long talk that work as standalone motivational shorts.

This is deliberately not `clip_selector.py`. That one picks a single "best clip"
to represent the video. This one hunts for *several* emotional beats — the lines
that land on their own, out of context, with no setup.

Two things shape what comes out:

**Length follows the material, not a target.** A short is as long as its idea
needs. If the whole punch is eight seconds, eight seconds is the right answer;
padding it to hit twenty is how you get something that opens strong and dribbles
out. If a passage genuinely sustains for twenty, it keeps all twenty.

**A short can be stitched from more than one cut.** Plenty of the best
short-form video is two passages from different parts of a talk — the setup from
one place, the payoff from another, with the meandering middle removed. So a
moment is a list of cuts, not a single window. One cut is normal; two is common;
the stitch has to earn itself, because a jump between unrelated passages reads
as a mistake rather than an edit.

When YouTube publishes a most-replayed curve, `heatmap.py` supplies the peaks
and they anchor the search — the cuts land where the audience actually rewatched
rather than where the transcript reads well.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional

from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-4-6"

MAX_SEGMENTS = 1200

# A single cut shorter than this can't carry a complete thought.
MIN_CUT_SECONDS = 4

# Breathing room after the last word, before the clip ends.
#
# Caption timings from YouTube tend to run early, so cutting exactly on a
# segment boundary clips the final word or ends on it. Ending the instant a
# speaker stops also just feels abrupt — the line needs a moment to land. This
# is capped at whatever gap actually exists before the next words, so it never
# eats into the following sentence.
LULL_SECONDS = 0.6

# How far past the model's end boundary to look for a real sentence ending.
#
# The failure is always the same shape — a statement cut short — so the search
# runs forward first. Four seconds is about one more sentence: enough to finish
# a thought that was clipped, not enough to bolt on the next idea.
ENDING_REACH = 4.0

# The finished short, all cuts added together. Overridable per run via
# make_shorts.py's --min-seconds / --max-seconds.
# One topic, covered until the speaker is finished with it. Length is what
# comes out, not what is aimed at.
#
# The floor rules out the fragment with no room to develop anything — a
# quotable line and nothing behind it, which is what a floor of 7 allowed.
#
# The ceiling is deliberately generous and is a backstop, not a target. The rule
# is that a topic is never truncated to fit: if it will not fit, the answer is a
# smaller complete topic inside it, or nothing. Two and a half minutes is past
# what anyone would call a short, which is the point — reaching it means
# something has gone wrong with the selection rather than with the limit.
MIN_TOTAL_SECONDS = 30
MAX_TOTAL_SECONDS = 150

# More than a couple of jumps stops reading as an edit and starts reading as a
# supercut of unrelated fragments.
MAX_CUTS = 3

# How many moments a talk yields is a RESULT, not a target.
#
# Two things were tried and both were wrong. Sizing by the number of replay
# peaks missed everything the audience didn't happen to rewatch. Sizing by the
# talk's length assumed minutes predict material, when a dense twelve-minute
# talk holds more than a rambling forty-minute one.
#
# So the model returns every passage that stands alone, each with a strength
# score, and only those clearing the bar are cut. A weak talk gives three; a
# rich one gives eight. Nothing is padded to reach a number, and nothing good
# is discarded to respect one.
# The ceiling is per WINDOW of transcript now, not per talk.
#
# Four was set when the complaint was too many weak shorts, and it was standing
# in for a quality bar rather than expressing a real preference for four.
# "One topic, finished" is that bar now — the speaker either gets where they
# were going inside the cut or they do not — so the count can follow the
# material.
#
# An hour of dense advice should yield every piece of advice in it, not four.
# Keeping the cap per window rather than per talk is what makes that scale: a
# nine-hour recording searched in fourteen windows can return far more than a
# forty-minute one without the number being set anywhere.
MAX_MOMENTS = 8

# Lowered from 8, and the reason is a measurement rather than a preference.
#
# Raising STRENGTH_THRESHOLD from 6.0 to 7.0 was expected to cut the batch and
# barely moved it: seven shorts became six, because the model scores relative to
# the bar it is given rather than on an absolute scale. Everything it wanted to
# keep arrived at exactly 7.0. A threshold alone cannot make a model more
# selective — it just moves where the scores cluster.
#
# The ceiling is the lever that actually bites, and it stays consistent with the
# design above: it is a maximum, not a target. A thin talk still returns two.

# On a 1-10 scale, where 5 is "fine but forgettable". Raise it if the weakest
# clips in a batch aren't worth posting; lower it if good material is missing.
#
# Raised from 6.0 to 7.0: fewer shorts, each carrying a real hook. The prompt
# tells the model a typical talk has one or two passages at 8+, a few at 6-7,
# and a lot at 4-5 — so 6.0 was admitting the whole middle band, which is where
# the moments live that are pleasant to listen to and have nothing a stranger
# would stop for. Seven shorts a talk became three or four.
#
# The hook standard is what makes this a real bar rather than a knob. A moment
# that cannot be opened on a line giving topic clarity and curiosity does not
# have a hook, and a short without a hook is not a short worth posting however
# good the sentiment inside it is.
STRENGTH_THRESHOLD = 7.0


# Two numbers, because length is a preference and the story is the point.
#
# The hook is not burned into the picture — it goes into `caption.txt`,
# `brief.md` and the carousel, which is to say it is the text you paste when
# posting. So the real constraint is softer than a render limit: a YouTube
# Shorts *title* shows roughly 40-50 characters on a phone before it truncates,
# while a TikTok or Reels *caption* runs to about 125. Sixty is a sensible
# target for the tightest of those and no kind of hard boundary.
#
# It is still worth asking for, because compressing demonstrably improved the
# writing rather than damaging it: forced to cut, the model drops the setup
# clause, which is nearly always the weak half. "When you keep pushing to
# better yourself, things expand — you get happier and healthier" (87) became
# "Stop learning and you'll rot on that porch" (42), and the short one is
# better by any reading.
#
# But a length rule must never overrule a quality judgement, which is exactly
# what a single hard 60 did: a hook the model had chosen as the best of three
# was thrown out for being 66 characters. So 60 is the target, and only past the
# ceiling below does length win.
HOOK_TARGET_CHARS = 60
HOOK_MAX_CHARS = 80


def _pick_hook(chosen: str, candidates: list[str]) -> str:
    """The best hook that actually fits.

    Prefers the model's own choice when it fits. When it doesn't, the shortest
    candidate that does is a better answer than truncating — the candidates are
    complete hooks that were judged against the same framework, while a cut-off
    hook is the exact failure the limit exists to prevent.

    Falls back to the over-long choice rather than returning nothing, so a batch
    still renders; `moments_over_length` reports it instead.
    """
    if chosen and len(chosen) <= HOOK_MAX_CHARS:
        return chosen
    fitting = [c for c in candidates if c and len(c) <= HOOK_MAX_CHARS]
    if fitting:
        return max(fitting, key=len)
    return chosen


def transcript_seconds(segments: list[dict]) -> float:
    """Playing time covered by the transcript."""
    if not segments:
        return 0.0
    last = segments[-1]
    return float(last.get("start", 0.0)) + float(last.get("duration", 0.0))

SYSTEM_PROMPT = """You are an editor who cuts motivational shorts for TikTok, \
Reels and YouTube Shorts out of long-form talks, interviews and podcasts.

You are looking for moments that hit hard with zero context. A good moment:
- Stands completely alone. A stranger scrolling past understands it instantly \
without knowing the speaker, the topic, or anything said before it.
- Contains one complete thought with a turn in it — a reframe, a hard truth, a \
challenge, a "most people think X, but actually Y".
- Starts on the first word of a sentence and ends on the last word of a \
sentence. Never start or end mid-thought.

FINDING THE EXACT BOUNDARIES. This is what separates a real moment from a fragment, and it is the part most easily got wrong. Work in two stages.

STAGE 1 - LOCATE. If replay peaks are given, they tell you where the audience went back. That is where the value is. Start there.

STAGE 2 - BOUND. A peak marks where attention spiked, and attention spikes on the PAYOFF - the line people returned for. That is almost never where the thought begins. Before you write down any timestamp, read the transcript on both sides of the peak and find the whole statement:
- Go BACKWARDS until you reach the real beginning of the idea. The setup that makes the payoff land usually sits before the peak, and cutting it off turns the payoff into a non-sequitur.
- Go FORWARDS until the statement actually finishes. A peak often sits on the first half of a line whose second half completes it.
- Only then read the timestamps off the transcript segments at those two boundaries, and cut there.

Think of where the most-replayed point of Hamlet's soliloquy would land: on "to be, or not to be". Cut at the peak alone and you have a fragment. The verse begins before those words and resolves after them, and nothing but the text tells you where. Treat every peak this way.

Rules that follow from this:
- Never open or close mid-thought, however hot the timestamp.
- If the complete statement runs past the maximum length, do NOT truncate it mid-sentence to fit. Either drop an interior digression using a second cut, or fall back to the strongest complete sub-statement inside it.
- If a peak sits on something the audio alone doesn't carry - a visual, a laugh, a slide - skip it and say so in `reason`.
- Record which peak each moment came from in `peak_rank`, or 0 if you chose it from the transcript alone.

LENGTH FOLLOWS THE MATERIAL. Judge each moment on its own:
- A passage that keeps building — every sentence adding to the last, the idea \
developing rather than repeating — should keep its full length, right up to the \
maximum. A sustained twenty seconds that carries the viewer through a complete \
argument is worth far more than a seven-second fragment of it. Reach for the \
full length whenever the material earns it.
- The short end is a floor for genuinely brief ideas, not a goal. If the whole \
punch really is seven or eight seconds — one hard line that needs nothing around \
it — then seven or eight seconds is correct, and padding it would only weaken it.
- What you must never do is stretch a moment with filler: dead air, \
throat-clearing, "so anyway", the same point restated, or setup that only \
matters inside the full talk. Cut those even if it leaves the moment short.
- Deciding between the two: read the sentences on either side. If removing them \
loses something, keep them. If the moment is just as strong without them, they \
are padding.

YOU MAY STITCH CUTS TOGETHER, and you should when it makes a moment stronger. \
A moment is a list of cuts, played in order:
- One cut is the normal case: a single continuous passage.
- Two cuts are the tool for when the setup and the payoff sit apart in the talk. \
The classic pattern: a sharp framing line early on, then the concrete payoff \
minutes later, with the explanation between them dropped. A lot of the strongest \
short-form video is built exactly this way, and it lets you keep a full-length \
short made entirely of the good parts.
- Reach for a stitch especially when a strong line is stranded — surrounded by \
material too weak to include — but pairs naturally with a passage elsewhere.
- Only stitch when the parts genuinely belong together: the first must set up \
the second, and the join must sound deliberate. If a listener would hear the \
jump as a mistake, use one cut instead.
- Each cut must itself start and end on sentence boundaries.

THE SHAPE OF A MOMENT: HOOK, PROGRESSION, CLIMAX. A short is not a good passage with a good first line, it is a small piece of storytelling with three parts, and the middle one is the part most easily forgotten:

- HOOK — the opening seconds. Names the subject and raises a question.
- PROGRESSION — everything between. Moves *toward* the answer, delivering on what the hook promised. Not restating it, not circling it.
- CLIMAX — the end. The question gets answered. This is what the viewer stayed for and it belongs last.

THE MISTAKE THAT KILLS OTHERWISE GOOD SHORTS IS PAYING OFF TOO EARLY. If the answer arrives five seconds in, there is nothing left to wait for and the viewer leaves — not because the material was weak, but because it was spent. A short that opens "Mr Beast's secret is to make 100 videos and improve each time" has no reason to continue; the same material, with the secret arriving at the end, holds the viewer for the whole clip.

So when you place the boundaries, check WHERE THE STRONGEST LINE SITS inside the moment. It should fall in the last third. If the best line is the first thing said and everything after it is explanation, you have chosen the wrong boundaries: either start earlier, so the payoff has something to arrive after, or make the strong line the closer of a shorter cut.

This constrains hook-stitching. Pulling a framing line to the front is right when it poses the question; it is wrong when it gives away the answer. Name the subject at the front, keep the resolution for the end.

WRITE THE LAST SENTENCE OUT AND JUDGE IT. Before submitting any moment, put its final sentence in `ends_on` verbatim and decide in `ending_check` whether it finishes the thought. This is not the same as checking that the timestamp lands on a sentence boundary — a grammatically complete sentence can still be a promise. One short ended on "And when I say that, I mean this." That is a full sentence, correctly bounded, and it is a cliffhanger with no payoff: the answer came two seconds later and was left out of the cut. Another ended on "Young people." and simply trailed off.

A viewer who reaches the end of a short and has not been given the thing it promised does not feel curious, they feel cheated, and they do not watch the next one. If `ending_complete` would be false, move the end forward until the statement resolves.

END ON THE POINT. The last line heard is what the viewer leaves with, and a \
moment that trails off into whatever the speaker said next wastes everything \
before it. Every moment must finish on a line that lands its idea.

- Prefer a passage that already ends on its own conclusion.
- If the strongest closing line sits somewhere else in the talk, stitch it on \
as a final cut. This is common in compilations, where the same idea is stated \
several times and the sharpest phrasing is not always in the same place as the \
fullest explanation.
- A closer is a statement, not a summary: the line that makes someone stop and \
think, not a restatement of what was just said.
- Read the finished moment back as a whole. If the ending is weaker than the \
middle, you have not finished — either trim back to the strong ending or \
stitch a better one on.

ONE TOPIC, FINISHED. This is the first test a passage has to pass, before hooks, before length, before anything else, and it is the whole test.

A short covers exactly ONE thing the speaker is talking about — a statement, a lesson, an idea, a story — and it runs from where they begin that thing to where they are DONE with it. Not to where a good line happens to land, and not to wherever the maximum length falls. Where they finish.

There is no template for what "finished" contains. Some topics are a claim then a story that proves it. Some are a question then an answer. Some are one idea turned over three times until it is clear. What matters is that a viewer who watched only this clip would not be left waiting for the rest — the speaker got where they were going.

Two ways to get this wrong, and the first is far more common:

- STOPPING EARLY. The cut ends on a strong line while the speaker is still mid-topic, so the viewer is left holding a promise. This is the worst failure a short can have.
- RUNNING ON. The cut carries past the end of one topic into the next, so it stops being about one thing.

LENGTH IS AN OUTCOME, NEVER A TARGET. Do not trim a topic to fit a number and do not pad one to reach a number. If the speaker takes ninety seconds to finish a thought, the short is ninety seconds. If a topic genuinely will not fit inside the maximum, do not truncate it — find a smaller complete topic inside it, or drop the passage and say so in `reason`. A truncated topic is not a short.

The floor exists only to rule out the fragment that has no room to develop anything. It is not a length to aim at.

This test replaces counting. Do not return a fixed number and do not hold back good material to seem selective: return every passage where the speaker covers one topic and finishes it. A dense hour of advice should yield every piece of advice in it. A rambling hour should yield two.

WRITE THREE HOOKS, THEN CHOOSE. Do not write one hook and move on. The first hook that comes to mind is usually the most obvious phrasing of the moment, which is rarely the one that stops a thumb — and the difference between a good and a weak hook for the SAME passage is large. Two real examples from the same moment: "He lived in a car — then published 5 books" against "He lived in a car — then did all of this". Identical material; the second names nothing and promises nothing.

So for every moment: write three genuinely different candidates in `hook_candidates`, name what is wrong with the weaker ones in `hook_rejects` using the four mistakes below, then put the survivor in `hook`. The rejects line is the check that forces the comparison — write it before you choose, not after.

LENGTH: AIM FOR 60 CHARACTERS, NEVER PASS 80. Sixty is a target, not a gate — the story comes first and a hook that needs 66 characters to land should have them. Eighty is the ceiling, because past it a title is truncated on a phone.

Aim short anyway, because compressing tends to improve a hook rather than damage it. Forced to cut, what goes is the setup clause, and that is nearly always the weak half. "When you keep pushing to better yourself, things expand — you get happier and healthier" is 87 characters and rambles; "Stop learning and you'll rot on that porch" is 42 and is better by any reading. If a candidate runs long, try saying the same thing in fewer words before you accept the length — but if the long one is genuinely the strongest, keep it.

THE HOOK IS THE WHOLE JOB. A hook has exactly one purpose: to make a stranger \
decide to keep watching. It does that by giving two things at once — TOPIC \
CLARITY (they know what this is about) and ON-TARGET CURIOSITY (they believe it \
is for them, and they want the next line). A hook that gives only one of the two \
fails.

This applies in two places, and the first matters more:

1. THE MOMENT'S OPENING SPOKEN LINE. What is actually heard in the first two \
seconds. A viewer decides there, before any title is read.
2. The written `hook` field, which becomes the title and caption.

Four ways a hook fails. Check every moment against all four.

DELAY — the context arrives too late. If the first sentence is throat-clearing \
and the topic only appears in the third, the viewer has already gone. The \
opening line must land the subject immediately. This is the strongest reason to \
use a second cut: if the sharp framing line sits later in the talk, STITCH IT TO \
THE FRONT so the moment opens on it. A moment that opens with "so, you know, I \
was thinking about..." is a moment that needs its first cut replaced.

CONFUSION — the viewer cannot parse it. Fewer words, simpler words, active \
voice. Aim at a sixth-grade reading level. Test it: read the hook alone, cold, \
and ask whether there is more than one way to read it. If there is, rewrite it \
so only one reading survives.

IRRELEVANCE — the viewer does not see themselves in it. Two fixes, both cheap:
- Write to "you" and "your", not "I", "me" or "he". "If you have ever felt \
behind" beats "I felt behind for years", and it beats "he felt behind" by \
further still. A third-person hook about the speaker asks the viewer to care \
about a stranger before they have any reason to.
- Aim at a pain the viewer already has. A hook that solves a problem they feel \
beats one that describes something merely interesting.

DISINTEREST — no curiosity. Build it with CONTRAST: A, what the viewer already \
believes, against B, the alternative this moment offers. "Most people think X, \
this says Y." The distance between A and B is what makes them stay for the \
answer. State both sides, or state only B when A is obvious enough to be \
understood without saying it.

Two things follow that are easy to get wrong:
- Curiosity is NOT vagueness. "This one thing changed everything" is a \
non-hook: no topic, so nothing to be curious about. Withhold the ANSWER, never \
the SUBJECT.
- A hook is a promise the moment has to keep. If the payoff is not actually in \
the cut, the hook is a lie and the viewer leaves anyway.

Avoid: housekeeping, introductions, thanking the audience, references to slides \
or "as I mentioned earlier", statistics without a point, and anything that needs \
the previous five minutes to make sense.

Prefer moments that are emotionally direct: discipline, ownership, fear, \
failure, consistency, identity, long-term thinking. Rank them best-first.

Respond with ONLY valid JSON, no preamble, no markdown code fences."""


def schema_description(min_total: int, max_total: int) -> str:
    return f"""
Return a JSON object with exactly this shape:

{{
  "moments": [
    {{
      "cuts": [                      // 1-{MAX_CUTS} cuts, played in this order
        {{
          "start_seconds": number,   // must match a segment start time from the transcript
          "end_seconds": number      // must land on the end of a sentence
        }}
      ],
      "stitch_reason": string,       // if more than one cut: why they belong together.
                                     // "" for a single cut.
      "peak_rank": number,           // which replay peak this came from (its #), 0 if none
      "strength": number,            // 1-10: how strongly this would stop a stranger
                                     // scrolling, judged cold with no context
      "topic": string,               // the ONE thing this short is about, in a few words
      "completeness": string,        // How does the speaker finish it? Name the line or the beat
                                     // where they are done. If they are still going when the cut
                                     // ends, this passage is not a short — extend it or drop it.
      "hook_candidates": [string],   // exactly 3 different hooks for this moment, each a real
                                     // attempt rather than a variation of one idea. Come at it
                                     // from different angles: the concrete detail, the reader's
                                     // pain, the contrast stated outright.
      "hook_rejects": string,        // one line naming which candidates you rejected and which
                                     // of the four mistakes each one makes. Write this BEFORE
                                     // choosing — it is the check, not a justification.
      "hook": string,                // aim for <= 60 characters, hard ceiling 80. Title and
                                     // caption. Must give topic clarity
                                     // AND curiosity — see THE HOOK IS THE WHOLE JOB above.
                                     // Write to "you"/"your", never "he"/"she"/"I". Set up a
                                     // contrast where the material allows one. No hashtags, no
                                     // quotation marks, no vague teases ("this one thing...").
      "opens_on": string,            // the moment's first spoken sentence, verbatim. Write it
                                     // out so you have to look at what the viewer actually hears
                                     // first — if it is throat-clearing rather than a hook,
                                     // change where the moment starts or stitch a hook line to
                                     // the front.
      "ends_on": string,             // the LAST sentence of the moment, verbatim. Write it out.
      "ending_check": string,        // Does that sentence finish the thought, or set up something
                                     // the viewer never hears? Judge it cold, as someone who saw
                                     // only this clip. "Ends on 'And when I say that, I mean
                                     // this.' — that is a promise, and the answer is in the next
                                     // sentence which is not in the cut. BROKEN, extend the end."
      "ending_complete": boolean,    // false means go back and move the end. Do not submit a
                                     // moment with false here.
      "payoff_at": string,           // "opening" | "middle" | "end" — where the moment's
                                     // strongest line actually falls. Say where it IS, not where
                                     // it should be. Anything other than "end" means the
                                     // boundaries want rechecking before you submit them.
      "contrast": string,            // the A-vs-B this hook sets up, as "A -> B", or "" if the
                                     // moment genuinely has no contrast in it. Naming it is how
                                     // you check the hook has one.
      "quote": string,               // the single strongest verbatim line in the moment
      "theme": string,               // one of: discipline, ownership, failure, fear, consistency,
                                     // identity, focus, growth, resilience, purpose
      "tone": string,                // e.g. "calm and certain", "urgent", "confrontational"
      "visual_keywords": [string],   // 3-5 concrete, filmable search terms for background footage.
                                     // Concrete nouns and actions only ("runner at dawn",
                                     // "hands gripping barbell"), never abstractions like
                                     // "success" or "motivation".
      "reason": string               // one sentence: why this lands as a standalone short
    }}
  ]
}}

Total playing time across a moment's cuts must be between {min_total} and \
{max_total} seconds, and no single cut may be shorter than {MIN_CUT_SECONDS} \
seconds. Within that, choose the length the material actually deserves.
"""


@dataclass
class Cut:
    start_seconds: float
    end_seconds: float

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds


@dataclass
class Moment:
    cuts: list[Cut]
    hook: str
    quote: str
    theme: str
    tone: str
    visual_keywords: list[str]
    reason: str
    stitch_reason: str = ""
    # The first sentence the viewer actually hears, and the A-vs-B the hook
    # sets up. Both are asked for so the model has to look at them rather than
    # assume them — a hook is checked against what is heard first, not against
    # the title written afterwards. Carried into the brief so the same check is
    # available to a human reading the folder.
    opens_on: str = ""
    contrast: str = ""
    topic: str = ""
    completeness: str = ""
    ends_on: str = ""
    ending_check: str = ""
    # Where the strongest line falls inside the moment. Asked for so the model
    # has to look: a short whose best line arrives first has spent itself by
    # the fifth second, which is the failure that structure — as opposed to
    # editing — is there to prevent.
    payoff_at: str = ""
    # The three candidates and the reason the others lost. Kept rather than
    # discarded because the rejected ones are often nearly as good, and a human
    # picking a different one is faster than asking for a fresh set — the
    # wording varies more between calls than the moments do.
    hook_candidates: list = field(default_factory=list)
    hook_rejects: str = ""
    # Which replay peak this moment's cuts actually overlap (0 = none).
    # Measured from the finished cuts rather than taken from the model's own
    # answer — see `_peak_for` for why that self-report can't be trusted.
    peak_rank: int = 0
    # 1-10, the model's own judgement of how hard this lands cold. Used to
    # decide how many moments a talk yields — see STRENGTH_THRESHOLD.
    strength: float = 0.0
    # How strongly this overlaps a most-replayed peak, when YouTube publishes
    # one. 0 means either no heatmap or a moment the audience didn't return to.
    heat: float = 0.0

    @property
    def duration(self) -> float:
        """Playing time of the finished short — the cuts added together.

        Deliberately not the span from first start to last end: a two-cut
        moment skips everything in between, and the render length has to match
        what actually gets played.
        """
        return sum(c.duration for c in self.cuts)

    @property
    def start_seconds(self) -> float:
        return self.cuts[0].start_seconds if self.cuts else 0.0

    @property
    def end_seconds(self) -> float:
        return self.cuts[-1].end_seconds if self.cuts else 0.0

    @classmethod
    def from_dict(cls, d: dict) -> "Moment":
        """Rebuild a Moment from its `to_dict` form.

        The inverse of `to_dict`, so a set of moments can be saved, edited by
        hand and rendered again without asking the model a second time. That
        matters for more than convenience: the model returns a different set of
        hooks on every call, so without this there is no way to keep a wording
        you liked. Reviewing several passes and pinning the best one is the
        workflow this enables.

        Fields `to_dict` derives rather than stores — start, end, duration —
        are ignored, since they come back from the cuts.
        """
        return cls(
            cuts=[Cut(float(c["start_seconds"]), float(c["end_seconds"]))
                  for c in d.get("cuts", [])],
            hook=d.get("hook", ""),
            quote=d.get("quote", ""),
            theme=d.get("theme", ""),
            tone=d.get("tone", ""),
            visual_keywords=list(d.get("visual_keywords") or []),
            reason=d.get("reason", ""),
            stitch_reason=d.get("stitch_reason", ""),
            hook_candidates=list(d.get("hook_candidates") or []),
            hook_rejects=d.get("hook_rejects", ""),
            topic=d.get("topic", ""),
            completeness=d.get("completeness", ""),
            opens_on=d.get("opens_on", ""),
            ends_on=d.get("ends_on", ""),
            ending_check=d.get("ending_check", ""),
            payoff_at=d.get("payoff_at", ""),
            contrast=d.get("contrast", ""),
            peak_rank=int(d.get("peak_rank", 0) or 0),
            strength=float(d.get("strength", 0.0) or 0.0),
            heat=float(d.get("heat", 0.0) or 0.0),
        )

    def to_dict(self) -> dict:
        return {
            "cuts": [asdict(c) for c in self.cuts],
            "stitch_reason": self.stitch_reason,
            "peak_rank": self.peak_rank,
            "strength": self.strength,
            "hook": self.hook,
            "hook_candidates": self.hook_candidates,
            "hook_rejects": self.hook_rejects,
            "topic": self.topic,
            "completeness": self.completeness,
            "opens_on": self.opens_on,
            "ends_on": self.ends_on,
            "ending_check": self.ending_check,
            "payoff_at": self.payoff_at,
            "contrast": self.contrast,
            "quote": self.quote,
            "theme": self.theme,
            "tone": self.tone,
            "visual_keywords": self.visual_keywords,
            "reason": self.reason,
            "heat": self.heat,
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "duration_seconds": round(self.duration, 2),
        }


def _build_user_prompt(
    segments: list[dict],
    count: int,
    title: str,
    summary: str,
    hot_windows: Optional[list[dict]] = None,
    min_total: int = MIN_TOTAL_SECONDS,
    max_total: int = MAX_TOTAL_SECONDS,
    auto: bool = False,
) -> str:
    capped = segments[:MAX_SEGMENTS]
    segments_text = "\n".join(f'[{s["start"]:.1f}s] {s["text"]}' for s in capped)
    truncated = "" if len(segments) <= MAX_SEGMENTS else "\n[transcript truncated for length]"

    context = f"Talk title: {title}\n" if title else ""
    if summary:
        context += f"What it's about: {summary}\n"

    heat_block = ""
    if hot_windows:
        listed = "\n".join(
            f"  #{w['rank']}  {w['start']:.0f}s-{w['end']:.0f}s  (replay intensity {w['value']:.2f})"
            for w in hot_windows
        )
        heat_block = (
            '\nYouTube publishes a "most replayed" graph for this video, and these are\n'
            "the timestamps its audience went back to most. The opening drop-off has\n"
            "already been excluded, so these are genuine peaks:\n\n"
            f"{listed}\n\n"
            "Treat these as strong evidence about which passages land — they reflect\n"
            "what real viewers rewatched, not what reads well on the page. Build your\n"
            "cuts around them where you can, and they are especially useful for\n"
            "stitching: a peak that needs setup often pairs with an earlier passage\n"
            "that supplies it.\n\n"
            "Two caveats, and they matter:\n"
            "- A peak marks roughly WHERE attention spiked, not where the sentence\n"
            "  starts. Use the transcript to place the actual cut on clean sentence\n"
            "  boundaries; a moment opening mid-thought is useless however hot the\n"
            "  timestamp.\n"
            "- A peak is not automatically a good standalone clip. If the passage only\n"
            "  makes sense with the surrounding talk, or the spike is on a visual the\n"
            "  audio doesn't carry, skip it. Say so in `reason`.\n\n"
            "You may still choose a moment outside these windows if it is clearly\n"
            "stronger.\n"
        )

    if auto and hot_windows:
        # Peaks first, then the rest of the talk. Peaks are the best evidence
        # available, but a talk holds more good material than it has peaks —
        # people rewatch what surprised them, not everything that would stop a
        # stranger scrolling. Cutting only at peaks leaves most of it behind.
        instruction = (
            f"Return every passage in this talk that works as a standalone "
            f"short, up to {count}. There is NO target number — the right answer "
            f"is however many the talk actually contains.\n\n"
            f"START WITH THE REPLAY PEAKS above. They are the best evidence "
            f"available about what lands, so take every peak that stands alone.\n\n"
            f"THEN READ THE WHOLE TRANSCRIPT for the best statements anywhere "
            f"else, peak or no peak. A talk usually holds more good material than "
            f"it has peaks: the audience replays what surprised them, which is "
            f"not the same as everything that would stop a stranger scrolling.\n\n"
            f"SCORE EACH ONE 1-10 in `strength`, judged cold — imagine it "
            f"arriving in a stranger's feed with no context, no speaker they "
            f"recognise, and a thumb ready to scroll. Be honest and use the whole "
            f"range. A typical talk has one or two passages at 8+, a few at 6-7, "
            f"and a lot of material around 4-5 that is perfectly reasonable to "
            f"listen to and would still be scrolled past.\n\n"
            f"Do NOT inflate a score to get a passage included, and do not hold "
            f"back a high one to seem discerning. Anything below the bar is "
            f"dropped automatically, so an accurate low score costs nothing — an "
            f"inflated one puts a weak clip in front of an audience.\n\n"
            f"SCORE THE HOOK, NOT THE SENTIMENT. The question is not whether "
            f"the passage is wise or well said — most of a good talk is both. It "
            f"is whether the first two seconds would stop a thumb. A passage you "
            f"cannot open on a line that gives topic clarity and curiosity does "
            f"not have a hook, and scores below the bar however true it is. "
            f"Fewer, stronger moments is the desired outcome: returning three "
            f"that land beats returning eight where five are filler.\n\n"
            f"For every moment, apply STAGE 2 before writing timestamps: expand "
            f"outwards to the complete statement, then read the boundaries off "
            f"the segment times."
        )
    else:
        instruction = (
            f"Find the {count} strongest standalone moments. Use the segment "
            f"timestamps to place every cut on sentence boundaries, expanding "
            f"outwards to the complete statement rather than cutting at the "
            f"first strong line you find."
        )

    return f"""{context}
Timestamped transcript segments:
{segments_text}{truncated}
{heat_block}
{instruction}

{schema_description(min_total, max_total)}
"""


def _snap_cut(
    start: float,
    end: float,
    segments: list[dict],
    tail: float = LULL_SECONDS,
) -> tuple[float, float]:
    """Nudge one cut onto real segment boundaries, and let the ending breathe.

    The model works from printed timestamps and is usually close but rarely
    exact. Snapping keeps cuts off the middle of a word.

    The tail matters as much as the snap. YouTube's caption timings routinely
    end a beat before the speaker actually stops, so a cut placed exactly on a
    segment boundary clips the final word or lands hard on it — the statement
    finishes and the video ends in the same instant, which feels abrupt and
    cheap. A short tail lets the last word finish and gives it a moment to
    land.

    How far to extend depends on what follows:

    - If the speaker carries straight on, stop just before the next segment
      begins. That is roughly where the last word truly ends, and it takes back
      what the early caption timing stole without stealing the next sentence.
    - If there is a real pause, take the full lull.
    """
    if not segments:
        return round(start, 2), round(end + tail, 2)

    starts = [float(s["start"]) for s in segments]
    snapped_start = min(starts, key=lambda t: abs(t - start))

    ends = [
        float(s["start"]) + float(s.get("duration", 0.0))
        for s in segments
        if float(s["start"]) >= snapped_start
    ]
    snapped_end = min(ends, key=lambda t: abs(t - end)) if ends else end

    # End on a sentence, not merely on a caption boundary.
    #
    # Snapping to the nearest segment end is not enough and this is where the
    # clipped endings came from. YouTube breaks auto-captions every few seconds
    # regardless of grammar, so the nearest boundary is frequently the middle of
    # a sentence — one short ended on "but I just wanted to share that with you"
    # with "this morning" in the next segment, and another on "Young people."
    #
    # `_natural_breaks` already knows where a statement actually ends, by
    # punctuation or by a real pause; it was only being consulted when a moment
    # overran the maximum length. It applies to every ending.
    #
    # Forward first, because the failure is always a sentence cut short. Going
    # back to the previous break loses the payoff the cut was chosen for, while
    # going forward costs a couple of seconds and completes it.
    breaks = _natural_breaks(segments, snapped_start, snapped_end + ENDING_REACH)
    if breaks:
        ahead = [t for t in breaks if t >= snapped_end - 0.25]
        behind = [t for t in breaks if t < snapped_end - 0.25]
        if ahead:
            snapped_end = min(ahead)
        elif behind:
            snapped_end = max(behind)

    if tail > 0:
        following = [t for t in starts if t > snapped_end + 0.05]
        if following:
            # Never run into the next words; stop a hair short of them.
            snapped_end = min(snapped_end + tail, min(following) - 0.05)
        else:
            snapped_end = snapped_end + tail

    return round(snapped_start, 2), round(max(snapped_end, snapped_start + 0.1), 2)


# A gap this long between caption segments means the speaker actually stopped —
# end of a sentence, or at least of a thought.
PAUSE_SECONDS = 0.35


def _natural_breaks(segments: list[dict], lo: float, hi: float) -> list[float]:
    """Times inside (lo, hi) where it is safe to end a clip.

    Not every caption boundary is one. YouTube breaks captions every few
    seconds regardless of grammar, so ending on an arbitrary segment lands
    mid-sentence — which is exactly how a clip came to end on "...and at least
    in the Kung Fu training it".

    Two signals mark a real break, and both are needed because transcripts vary:

    - **Punctuation**, when the transcript has any. Manually written captions
      usually do.
    - **A pause**, when it doesn't. Auto-generated captions frequently carry no
      punctuation at all, and then the only evidence of a sentence ending is
      that the speaker stopped talking for a moment.
    """
    breaks: list[float] = []
    for i, seg in enumerate(segments):
        end = float(seg.get("start", 0.0)) + float(seg.get("duration", 0.0))
        if not (lo < end < hi):
            continue
        text = str(seg.get("text", "")).rstrip()
        gap = 0.0
        if i + 1 < len(segments):
            gap = float(segments[i + 1].get("start", 0.0)) - end
        if text.endswith((".", "!", "?")) or gap >= PAUSE_SECONDS:
            breaks.append(end)
    return sorted(breaks)


def _trim_to_budget(
    cuts: list[Cut], segments: list[dict], max_total: float
) -> list[Cut]:
    """Bring an over-long moment back under budget without cutting mid-sentence.

    The old behaviour truncated at `start + max`, which lands on an arbitrary
    timestamp — mid-word, sometimes mid-syllable. Instead the final cut is
    pulled back to the last sentence boundary that fits. If no boundary fits,
    the cut is left slightly long: running two seconds over reads as fine,
    while ending mid-word does not.
    """
    total = sum(c.duration for c in cuts)
    if total <= max_total or not cuts:
        return cuts

    overrun = total - max_total
    last = cuts[-1]

    target = last.end_seconds - overrun

    def fits(times: list[float]) -> list[float]:
        return [
            t for t in times
            if t <= target and t - last.start_seconds >= MIN_CUT_SECONDS
        ]

    natural = fits(_natural_breaks(segments, last.start_seconds, last.end_seconds))
    anywhere = fits(sorted(
        float(s["start"]) + float(s.get("duration", 0.0))
        for s in segments
        if last.start_seconds
        < float(s["start"]) + float(s.get("duration", 0.0))
        < last.end_seconds
    ))

    # Prefer a real pause or full stop — but not at any price. On a
    # well-punctuated transcript there are plenty to choose from. On
    # auto-generated captions there may be almost none (this video had 29
    # across twenty minutes), and the nearest one can sit thirty seconds early:
    # taking it would cut a 43-second clip down to 11. A clip ending on a
    # caption boundary is imperfect; one cut to a quarter of its length is
    # ruined, so the natural break has to keep most of the clip to win.
    chosen = None
    if natural:
        chosen = natural[-1]
    if anywhere:
        best = anywhere[-1]
        if chosen is None or (chosen - last.start_seconds) < 0.6 * (best - last.start_seconds):
            chosen = best

    if chosen is not None:
        cuts[-1] = Cut(last.start_seconds, round(chosen, 2))
    elif len(cuts) > 1 and (total - last.duration) >= MIN_TOTAL_SECONDS:
        # No usable boundary inside the last cut — drop it rather than mangle it.
        cuts = cuts[:-1]
    return cuts


def _peak_for(cuts: list[Cut], hot_windows: Optional[list[dict]]) -> int:
    """Rank of the replay peak these cuts genuinely overlap, or 0 for none.

    Measured rather than taken from the model's own `peak_rank`, because that
    self-report doesn't survive contact with reality: on a test run three of
    seven moments named a peak sitting 100-180 seconds away from where they
    actually cut. The claim is cheap for the model to get wrong and cheap for
    us to check, so we check.

    Overlap is scored by how much of the peak window the cuts actually cover,
    so a moment brushing the edge of a hot window doesn't outrank one sitting
    squarely inside it.
    """
    if not hot_windows:
        return 0
    best_rank, best_overlap = 0, 0.0
    for window in hot_windows:
        overlap = 0.0
        for cut in cuts:
            overlap += max(
                0.0,
                min(cut.end_seconds, window["end"]) - max(cut.start_seconds, window["start"]),
            )
        if overlap > best_overlap:
            best_rank, best_overlap = int(window.get("rank", 0) or 0), overlap
    return best_rank


def _heat_for(cuts: list[Cut], hot_windows: Optional[list[dict]]) -> float:
    """Replay intensity of the hottest peak any of these cuts overlaps."""
    if not hot_windows:
        return 0.0
    best = 0.0
    for cut in cuts:
        for window in hot_windows:
            if cut.start_seconds < window["end"] and cut.end_seconds > window["start"]:
                best = max(best, float(window.get("value", 0.0)))
    return round(best, 3)



# How much consecutive windows share, in segments. A moment can be a minute
# long; the overlap has to exceed that or one straddling a boundary is
# truncated in the earlier window and missing its opening in the later one.
WINDOW_OVERLAP = 60


def _find_in_windows(
    segments: list[dict],
    count: int,
    title: str,
    summary: str,
    api_key: Optional[str],
    hot_windows: Optional[list[dict]],
    min_total: int,
    max_total: int,
    auto: bool,
) -> list[Moment]:
    """Search a long talk in overlapping windows and pool the candidates.

    Each window is a separate call, so this costs one request per
    `MAX_SEGMENTS` of transcript — about twelve for a nine-hour recording. That
    is the price of reading the whole thing, and the alternative was not a
    cheaper search but a silent decision to ignore 92% of the material.

    Ranking happens across the pool rather than inside each window, so a talk
    whose best passages are all in hour seven is not forced to take weaker ones
    from hour one for balance.
    """
    step = MAX_SEGMENTS - WINDOW_OVERLAP
    windows = [segments[i:i + MAX_SEGMENTS] for i in range(0, len(segments), step)]
    # A final window of a few segments has nothing to find and still costs a
    # call; its content is already covered by the overlap of the one before it.
    windows = [w for w in windows if len(w) > WINDOW_OVERLAP]

    print(f"  · Transcript is {len(segments)} segments — searching "
          f"{len(windows)} overlapping windows")

    pooled: list[Moment] = []
    for i, window in enumerate(windows, 1):
        lo = float(window[0]["start"])
        hi = float(window[-1]["start"])
        # Only the peaks inside this window mean anything to it. Passing them
        # all would point the model at timestamps it cannot see.
        peaks = [
            w for w in (hot_windows or [])
            if lo <= float(w.get("start", -1)) <= hi
        ] or None
        try:
            found = find_moments(
                window, count=count, title=title, summary=summary,
                api_key=api_key, hot_windows=peaks,
                min_total=min_total, max_total=max_total,
            )
        except Exception as e:
            print(f"    window {i}/{len(windows)} ({lo/60:.0f}-{hi/60:.0f} min) "
                  f"failed: {str(e)[:60]}")
            continue
        print(f"    window {i}/{len(windows)} ({lo/60:.0f}-{hi/60:.0f} min): "
              f"{len(found)} candidate(s)")
        pooled += found

    # Drop duplicates from the overlap, keeping the stronger reading of the
    # same passage. Two windows seeing one moment is the overlap working, not a
    # fault, but shipping it twice would be.
    pooled.sort(key=lambda m: m.strength, reverse=True)
    kept: list[Moment] = []
    for m in pooled:
        if any(abs(m.start_seconds - k.start_seconds) < 10 for k in kept):
            continue
        kept.append(m)
    return kept[:count] if not auto else kept

def find_moments(
    segments: list[dict],
    count: Optional[int] = None,
    title: str = "",
    summary: str = "",
    api_key: Optional[str] = None,
    hot_windows: Optional[list[dict]] = None,
    min_total: int = MIN_TOTAL_SECONDS,
    max_total: int = MAX_TOTAL_SECONDS,
) -> list[Moment]:
    """Find the moments in a talk worth cutting.

    `count=None` means the talk decides: one moment per replay peak that stands
    alone, capped at MAX_MOMENTS. Pass a number to force exactly that many.
    """
    if not segments:
        raise ValueError("No timestamped transcript segments to search for moments.")

    auto = count is None
    # In auto mode the number is decided after the fact, by which candidates
    # clear the bar. The model is asked for the maximum so it has room to
    # return everything worth cutting.
    count = MAX_MOMENTS if auto else max(1, min(int(count), MAX_MOMENTS))

    # A talk longer than one prompt is searched in windows, not truncated.
    #
    # `segments[:MAX_SEGMENTS]` silently decided *which part of the talk
    # exists*. On a nine-hour recording that is 8% of it — the first 42 minutes
    # — and the other eight hours were never read. The failure looks like "the
    # model didn't find much" rather than "the model never saw it", which is
    # why it could sit here unnoticed.
    #
    # Windows overlap, because a moment that straddles a boundary would
    # otherwise be lost from both sides. Every window is searched, the
    # candidates are pooled, and the usual strength bar picks the winners — so a
    # nine-hour video still yields MAX_MOMENTS shorts, but chosen from all of it.
    if len(segments) > MAX_SEGMENTS:
        return _find_in_windows(
            segments, count, title, summary, api_key, hot_windows,
            min_total, max_total, auto,
        )

    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=MODEL,
        # Generous, because the answer grew a lot and the failure is total.
        #
        # Each moment now carries point, explanation, proof, three hook
        # candidates, the rejection reasoning, opens_on, ends_on, the ending
        # check, contrast and the rest — call it 300 tokens — and the ceiling
        # went to 8 moments per window. At 4000 the JSON was simply cut off
        # mid-object, which does not degrade gracefully: the parse fails and the
        # whole window returns nothing. Both windows of a talk failed this way
        # and the run produced no shorts at all.
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": _build_user_prompt(
                segments, count, title, summary, hot_windows,
                min_total, max_total, auto,
            ),
        }],
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

    moments: list[Moment] = []
    for item in data.get("moments", []):
        raw_cuts = item.get("cuts") or []
        # Tolerate the older single-window shape rather than dropping the moment.
        if not raw_cuts and item.get("start_seconds") is not None:
            raw_cuts = [{
                "start_seconds": item.get("start_seconds"),
                "end_seconds": item.get("end_seconds"),
            }]

        cuts: list[Cut] = []
        for raw_cut in raw_cuts[:MAX_CUTS]:
            try:
                start, end = _snap_cut(
                    float(raw_cut.get("start_seconds", 0)),
                    float(raw_cut.get("end_seconds", 0)),
                    segments,
                )
            except (TypeError, ValueError):
                continue
            if end - start >= MIN_CUT_SECONDS:
                cuts.append(Cut(start, end))

        if not cuts:
            continue

        cuts.sort(key=lambda c: c.start_seconds)
        cuts = _trim_to_budget(cuts, segments, max_total)

        if sum(c.duration for c in cuts) < MIN_CUT_SECONDS:
            continue

        moments.append(Moment(
            cuts=cuts,
            hook=_pick_hook(
                item.get("hook", "").strip().strip('"'),
                [h.strip().strip('"') for h in
                 (item.get("hook_candidates") or []) if h.strip()],
            ),
            hook_candidates=[h.strip().strip('"') for h in
                             (item.get("hook_candidates") or []) if h.strip()],
            hook_rejects=item.get("hook_rejects", "").strip(),
            topic=item.get("topic", "").strip(),
            completeness=item.get("completeness", "").strip(),
            opens_on=item.get("opens_on", "").strip().strip('"'),
            ends_on=item.get("ends_on", "").strip().strip('"'),
            ending_check=item.get("ending_check", "").strip(),
            payoff_at=item.get("payoff_at", "").strip().lower(),
            contrast=item.get("contrast", "").strip(),
            quote=item.get("quote", "").strip(),
            theme=item.get("theme", "").strip().lower(),
            tone=item.get("tone", "").strip(),
            visual_keywords=[k.strip() for k in item.get("visual_keywords", []) if k.strip()],
            reason=item.get("reason", "").strip(),
            stitch_reason=item.get("stitch_reason", "").strip(),
            peak_rank=_peak_for(cuts, hot_windows),
            strength=float(item.get("strength", 0) or 0),
            heat=_heat_for(cuts, hot_windows),
        ))

    # This is where the count is actually decided: not by a number handed to
    # the model, but by how many passages cleared the bar. Only in auto mode —
    # an explicit --count means the user asked for exactly that many.
    if auto:
        kept = [m for m in moments if m.strength >= STRENGTH_THRESHOLD]
        if not kept and moments:
            best = max(moments, key=lambda m: m.strength)
            print(
                f"  · Nothing cleared {STRENGTH_THRESHOLD:.0f}/10 "
                f"(best was {best.strength:.0f}) — keeping the strongest anyway. "
                f"This talk may not have much that stands alone."
            )
            kept = [best]
        elif len(kept) < len(moments):
            print(
                f"  · {len(moments) - len(kept)} passage(s) scored below "
                f"{STRENGTH_THRESHOLD:.0f}/10 and were dropped"
            )
        moments = kept

    # Strongest first, with replay evidence breaking ties.
    moments.sort(key=lambda m: (m.strength, m.heat), reverse=True)
    return moments


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python moment_finder.py <path_to_transcript.json> [count]")
        sys.exit(1)

    # No count given means auto — the peaks decide how many.
    count = int(sys.argv[2]) if len(sys.argv) > 2 else None
    with open(sys.argv[1], "r", encoding="utf-8") as f:
        data = json.load(f)

    found = find_moments(
        data.get("transcript_segments", []),
        count=count,
        title=data.get("title", ""),
    )
    for i, m in enumerate(found, 1):
        cuts = " + ".join(
            f"{c.start_seconds:.0f}-{c.end_seconds:.0f}s ({c.duration:.0f}s)" for c in m.cuts
        )
        print(f"\n{i}. {cuts}  = {m.duration:.0f}s total  [{m.theme}]")
        print(f"   HOOK: {m.hook}")
        print(f'   "{m.quote}"')
        if m.stitch_reason:
            print(f"   stitch: {m.stitch_reason}")
        print(f"   {m.reason}")
