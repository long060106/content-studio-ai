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

# The finished short, all cuts added together. Overridable per run via
# make_shorts.py's --min-seconds / --max-seconds.
MIN_TOTAL_SECONDS = 7
MAX_TOTAL_SECONDS = 25

# More than a couple of jumps stops reading as an edit and starts reading as a
# supercut of unrelated fragments.
MAX_CUTS = 3

# When nobody asks for a specific number, the talk decides: one moment per
# replay peak worth cutting. The cap stops a noisy retention curve turning into
# a twenty-short run, and the fallback covers videos YouTube publishes no
# heatmap for (it needs a view threshold), where there are no peaks to count.
MAX_MOMENTS = 8
DEFAULT_MOMENTS = 5

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
      "hook": string,                // <= 60 chars, the on-screen title card. Punchy, no hashtags,
                                     // no quotation marks. This is the scroll-stopper.
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
    # Which replay peak this moment's cuts actually overlap (0 = none).
    # Measured from the finished cuts rather than taken from the model's own
    # answer — see `_peak_for` for why that self-report can't be trusted.
    peak_rank: int = 0
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

    def to_dict(self) -> dict:
        return {
            "cuts": [asdict(c) for c in self.cuts],
            "stitch_reason": self.stitch_reason,
            "peak_rank": self.peak_rank,
            "hook": self.hook,
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
        # No target number. The talk has as many moments as it has peaks worth
        # cutting, and saying so explicitly stops the model padding the list
        # out with weak passages to reach a count it was given.
        instruction = (
            f"Work through the replay peaks above and return one moment for each "
            f"peak that genuinely stands alone — up to {count}. There is no "
            f"target number: if only three of those peaks survive the test, "
            f"return three. A short list of strong moments is the goal; padding "
            f"it with passages that need context ruins the batch.\n"
            f"For every peak you skip, you do not need to explain yourself — "
            f"just leave it out.\n\n"
            f"For each one you keep, apply STAGE 2 before writing timestamps: "
            f"expand outwards from the peak to the complete statement, then read "
            f"the boundaries off the segment times."
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


def _snap_cut(start: float, end: float, segments: list[dict]) -> tuple[float, float]:
    """Nudge one cut onto real segment boundaries.

    The model works from printed timestamps and is usually close but rarely
    exact. Snapping keeps cuts off the middle of a word.
    """
    if not segments:
        return round(start, 2), round(end, 2)

    starts = [float(s["start"]) for s in segments]
    snapped_start = min(starts, key=lambda t: abs(t - start))

    ends = [
        float(s["start"]) + float(s.get("duration", 0.0))
        for s in segments
        if float(s["start"]) >= snapped_start
    ]
    snapped_end = min(ends, key=lambda t: abs(t - end)) if ends else end
    return round(snapped_start, 2), round(snapped_end, 2)


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

    boundaries = sorted(
        float(s["start"]) + float(s.get("duration", 0.0))
        for s in segments
        if last.start_seconds < float(s["start"]) + float(s.get("duration", 0.0)) < last.end_seconds
    )
    target = last.end_seconds - overrun
    usable = [b for b in boundaries if b <= target and b - last.start_seconds >= MIN_CUT_SECONDS]

    if usable:
        cuts[-1] = Cut(last.start_seconds, round(usable[-1], 2))
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
    if auto:
        count = min(len(hot_windows), MAX_MOMENTS) if hot_windows else DEFAULT_MOMENTS
    count = max(1, min(int(count), MAX_MOMENTS))

    client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
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
            hook=item.get("hook", "").strip().strip('"'),
            quote=item.get("quote", "").strip(),
            theme=item.get("theme", "").strip().lower(),
            tone=item.get("tone", "").strip(),
            visual_keywords=[k.strip() for k in item.get("visual_keywords", []) if k.strip()],
            reason=item.get("reason", "").strip(),
            stitch_reason=item.get("stitch_reason", "").strip(),
            peak_rank=_peak_for(cuts, hot_windows),
            heat=_heat_for(cuts, hot_windows),
        ))

    # Moments the audience actually rewatched lead the list.
    moments.sort(key=lambda m: m.heat, reverse=True)
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
