"""
make_shorts.py

One talk in, a batch of motivational shorts out.

    python make_shorts.py "https://www.youtube.com/watch?v=VIDEO_ID"

What happens:

  1. Transcript      Reuses output/<video_id>/transcript.json if it's already
                     there, otherwise pulls YouTube's captions. The full talk
                     is never sent through Whisper — that's the slow, expensive
                     way to do this, and the captions are good enough to *find*
                     moments with.
  2. Moments         One Claude call works through the replay peaks and returns
                     one moment per peak that stands alone — however many that
                     is, up to 8 — each expanded outwards from the peak to the
                     complete statement, with a hook and visual search terms.
  3. Clips           Only those windows are downloaded, via the same ranged
                     yt-dlp path the main pipeline uses.
  4. Brief           brief.md next to each clip: where it came from, what is
                     said, why it was chosen, and what must not be trimmed.
                     The clip alone doesn't carry that, and an editor without
                     it cuts the setup off the front.
  5. Assets          B-roll comes from the local library, topped up from
                     stock APIs when a key is configured. No music — see
                     pick_assets for why.
  6. Render          ffmpeg assembles 1080x1920. Captions are off by default:
                     these clips get re-edited, and burned-in text fights that.
                     Pass --captions to burn them in anyway.
  7. Carousel        Optional: the card copy as text, for designing by hand.

Everything lands in output/<video_id>/shorts/, one folder per short, and the
downloaded clip is kept so you can re-render in a different style without
touching the network again.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import zlib
import sys
import threading

import asset_library
from caption_timing import captions_for_clip
from moment_finder import MAX_TOTAL_SECONDS, Moment, find_moments
from shorts_builder import (
    ShortSpec,
    build_plain_cut,
    build_rough_cut,
    build_short,
    join_clips,
    probe_duration,
)


# How many clips are built at once. Three is a deliberate floor rather than
# the core count: each worker keeps its own Whisper model in memory, and the
# ffmpeg encodes already use the GPU, so more lanes buy contention rather than
# speed. Override with --workers.
DEFAULT_WORKERS = 3

# Squeeze the long silences out of each cut before rendering.
#
# On the test talk this took a 37.7s clip to 27.8s, and almost all of it came
# from three holes — 4.9s, 2.4s and 1.5s — rather than from the speaker's
# natural rhythm. Short pauses are left alone deliberately, and the breath at
# the end that `ending_finder` works to create is never touched.
TIGHTEN_SILENCE = True

# Split the speaker's voice out of the backing music before rendering.
#
# Costs roughly twice real time on CPU — about eight minutes across a batch of
# eight — which is the price of removing something a filter cannot touch. The
# denoiser in shorts_builder handles hiss and room tone; this handles the score
# playing under the talk.
ISOLATE_VOICE = True

# Whether a run may top up its b-roll from the stock APIs.
#
# Off. Every cutaway comes from the film library in assets/broll/film, which is
# cut from films on this machine and named for what each shot shows. Stock was
# the reason that folder filled up with footage that reads as advertising.
STOCK_BROLL = False

# Ship the cut, not the edit.
#
# short.mp4 is the speaker's own picture at its native 16:9 — no window, no
# burned-in captions, no cutaways composited in. The b-roll is still chosen and
# still copied into each clip's broll/ folder, and shotlist.md still says which
# clip belongs under which line; it is a recommendation to drag onto a timeline
# rather than a decision baked into the pixels.
#
# The reasoning is that undoing a choice made here is expensive and undoing one
# made in an editor is not. A window, a caption or a cutaway that is already in
# the frame can only be removed by rendering the whole thing again.
PLAIN_ONLY = False

# Off. The user adds music himself, at upload.
#
# It was switched on once, briefly, and switched straight back off: the only
# beds available were synthesised pads, because neither Pexels nor Pixabay
# serves music through its API, and a generated drone is not music. That is the
# lesson rather than the flag — **do not put synthesised audio under a talk and
# call it a music bed.** If music is ever wanted here again it needs real
# tracks, chosen by him.
#
# The mixing itself works and is tested: drop audio files into `assets/music/`,
# set this to True, and every short gets one ducked under the voice. With the
# folder empty nothing is mixed even when this is True, so turning it on by
# accident cannot put a drone under a video.
#
# The original reasoning for having no music here still stands on its own:
# these shorts are short and typically get stitched two or three at a time into
# one upload, where a bed baked into each piece has to be re-cut, and a track
# added from TikTok's own library at upload is licensed for the platform in a
# way a mixed-in one is not.
MUSIC_ON = False

# Where beds come from. Any audio file dropped in here is a candidate; the
# choice is per-short so a batch does not come out sounding like one long
# track. With the folder empty the pipeline synthesises a plain pad rather than
# failing, which keeps a render honest but is not something to publish.
MUSIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "assets", "music")


def _pick_music(seed: str) -> str | None:
    """A bed for one short, or None to leave it dry.

    Chosen by hashing the short's own name rather than at random, so a re-run
    of the same talk keeps the same track under the same clip. A batch that
    re-rendered with the music reshuffled would make every short sound subtly
    different from the version already reviewed.

    `zlib.crc32` rather than the built-in `hash`, which is salted per process
    for strings — using it here would have given a different track on every run
    and quietly broken the only property this function promises.
    """
    if not MUSIC_ON:
        return None
    try:
        tracks = sorted(
            os.path.join(MUSIC_DIR, f) for f in os.listdir(MUSIC_DIR)
            if os.path.splitext(f)[1].lower() in {".mp3", ".m4a", ".wav", ".aac", ".ogg"}
        )
    except OSError:
        tracks = []
    if not tracks:
        return None
    return tracks[zlib.crc32(seed.encode("utf-8")) % len(tracks)]


def _slug(text: str, limit: int = 28) -> str:
    out = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return (out[:limit].rstrip("_") or "moment")


def load_or_fetch_transcript(url: str, output_dir: str) -> dict:
    """Prefer the transcript we already have; only hit the network if we must."""
    from youtube_extractor import extract, extract_video_id

    video_id = extract_video_id(url)
    path = os.path.join(output_dir, video_id, "transcript.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("transcript_segments"):
            print(f"  ✓ Reusing existing transcript ({len(data['transcript_segments'])} segments)")
            return data

    print("  · No local transcript, pulling captions...")
    video = extract(url)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(video.to_json())
    return json.loads(video.to_json())


def pick_assets(moment: Moment, fetch: bool) -> tuple[str | None, dict]:
    """Choose b-roll for one moment, fetching stock if allowed.

    No music. These shorts are published without a bed: they're deliberately
    short, so a publisher typically stitches two or three together into one
    post and lays music over that finished sequence. Music baked into each
    short would only have to be cut apart and re-matched at that point.
    """
    keywords = list(moment.visual_keywords) + [moment.theme]
    credits: dict = {}

    broll = asset_library.pick("video", keywords, min_duration=moment.duration)
    if broll is None and fetch and moment.visual_keywords:
        query = moment.visual_keywords[0]
        print(f"    · library has no matching b-roll, fetching \"{query}\"...")
        got = asset_library.fetch_stock(query, kind="video", count=2)
        if got:
            broll = asset_library.pick("video", keywords, min_duration=moment.duration)

    if broll:
        credits["broll"] = {"path": broll.path, "source": broll.source,
                            "credit": broll.credit, "licence": broll.licence}
    return (broll.abs_path if broll else None, credits)


def _remove_tree(path: str) -> bool:
    """Delete a folder, clearing read-only flags that would otherwise stop it.

    OneDrive marks synced folders ReadOnly with a reparse point, and plain
    rmtree fails on those with "Access is denied" even when nothing holds the
    files open. Clearing the attribute and retrying is what Explorer does.
    """
    import shutil
    import stat

    def on_error(func, target, _exc):
        try:
            os.chmod(target, stat.S_IWRITE)
            func(target)
        except OSError:
            pass

    shutil.rmtree(path, onerror=on_error)
    return not os.path.exists(path)


def _clear_stale_clips(shorts_dir: str, keep: set[str]) -> list[str]:
    """Remove clip folders left behind by an earlier run of the same talk.

    Folder names are built from the hook, and the hook is written fresh by the
    model each run, so re-cutting a talk leaves the previous folders sitting
    beside the new ones — same clip, different name, its own copy of the video.
    The library then shows one clip twice.

    Deliberately called *after* the clips are built rather than before. Clearing
    up front would mean a run that fails halfway destroys the clips you already
    had, which is the wrong direction to fail in. Nothing here runs unless the
    run actually produced something.
    """
    removed: list[str] = []
    if not os.path.isdir(shorts_dir):
        return removed

    for name in sorted(os.listdir(shorts_dir)):
        folder = os.path.join(shorts_dir, name)
        if not os.path.isdir(folder) or os.path.abspath(folder) in keep:
            continue
        # Only ever touch numbered clip folders: never `carousel/`, and never
        # anything a future version of this script might put here.
        if not re.fullmatch(r"\d+_.*", name):
            continue
        if _remove_tree(folder):
            removed.append(name)
    return removed


def _spoken_text(moment: Moment, segments: list[dict]) -> str:
    """The words actually said inside this moment's cuts.

    Taken from the transcript rather than from Whisper, so the brief is
    available whether or not captions were timed.
    """
    parts: list[str] = []
    for cut in moment.cuts:
        said = " ".join(
            s["text"] for s in segments
            if cut.start_seconds <= float(s["start"]) < cut.end_seconds
        ).strip()
        if said:
            parts.append(said)
    return "\n\n[hard cut]\n\n".join(parts)


def _words_from_segments(moment: Moment, segments: list[dict]) -> list:
    """Approximate word timings from the talk's own transcript.

    The fallback for when Whisper cannot run. It was referenced here for months
    and never actually written: every time the safety net deployed it raised
    NameError, the broad `except` around it swallowed that, and the run reported
    "Transcript timings failed too" — a message describing a data problem when
    the truth was that the code did not exist. The day Windows blocked both
    Whisper engines, this was the thing that should have kept captions and the
    shot plan alive, and it took the whole batch down instead.

    Precision is line-level, not word-level: each segment's words are spread
    evenly across its span, so a word's time is right to within a syllable or
    two rather than exact. That is enough for captions to track the voice and
    for the shot planner to cut on sentence ends, which is the whole point of
    having it.

    **Times come back clip-relative**, because everything downstream measures
    from the start of the cut and not from the start of the talk. Where a moment
    stitches several cuts, each one's words are shifted by the running length of
    the cuts before it — the clip is their concatenation.

    YouTube's caption segments overlap by design: they roll, so one starts
    before the last has finished. Each segment's end is therefore clamped to the
    next one's start, or words would be handed timings that run past the words
    after them.
    """
    from caption_timing import Word

    words: list = []
    offset = 0.0

    for cut in moment.cuts:
        inside = sorted(
            (s for s in segments
             if cut.start_seconds <= float(s["start"]) < cut.end_seconds),
            key=lambda s: float(s["start"]),
        )
        for i, seg in enumerate(inside):
            text = (seg.get("text") or "").strip()
            if not text:
                continue
            start = float(seg["start"])
            end = start + float(seg.get("duration") or 0.0)
            if i + 1 < len(inside):
                end = min(end, float(inside[i + 1]["start"]))
            end = min(end, cut.end_seconds)
            span = max(0.12, end - start)

            # Drop the transcriber's own stage directions. YouTube writes
            # "[Music]", "[Applause]" and "[Laughter]" into the caption stream,
            # and they are not spoken — burning one onto the screen as though
            # the speaker said it is the sort of mistake nobody looks for.
            parts = [w for w in text.split()
                     if not (w.startswith("[") and w.endswith("]"))]
            if not parts:
                continue
            step = span / len(parts)
            for j, word in enumerate(parts):
                a = offset + (start - cut.start_seconds) + j * step
                words.append(Word(text=word, start=a, end=a + step * 0.92))

        offset += max(0.0, cut.end_seconds - cut.start_seconds)

    return words


# The cutting rhythm. Short holds on the speaker, one or two quick b-roll
# shots, then back to the speaker — the pattern the reference videos use.
#
# It always opens on the speaker. Establishing who is talking before cutting
# away is what stops the clip reading as a stock-footage montage with a voice
# over it; once the face is established, the b-roll illustrates rather than
# replaces.
SPEAKER_HOLD_SECONDS = 2.0
BROLL_SHOT_SECONDS = 1.5
BROLL_RUN = 2          # b-roll shots before cutting back to the speaker

# The cut follows the sentence, not a stopwatch.
#
# A cutaway that changes mid-clause fights the speech: the viewer is still
# holding the first half of a thought when the picture tells them a new one has
# started. Cutting where the speaker actually stops means the edit and the
# sentence agree, which is what makes the b-roll read as illustration rather
# than as decoration laid over the top.
#
# Sentences that run past the upper bound are split at their longest internal
# pause, and ones under the lower bound are folded into their neighbour — a
# two-word sentence is a beat, not a shot.
SENTENCE_SHOTS = True

# Choose each cutaway by reading the line it sits under, rather than by counting
# words shared with a filename. See broll_picker for what this fixes and the
# measurements behind it. Costs one Claude call per short; turn it off to fall
# back to word scoring.
SEMANTIC_BROLL = True

# The fewest cutaways a short may ship with before the declines get overridden.
#
# The picker declining a line is a real answer and worth honouring — a cutaway
# that means nothing reads as a mistake. But honouring every decline produced a
# 17-second short with two cutaways and five speaker shots, which is not an
# edit, it is a talking head with interruptions. So the declines stand until
# they would take a short below this, and then the plan is rebuilt letting the
# word scoring fill the gaps it left.
# Four, not three, so the floor matches the standing target rather than sitting
# one under it. The intent is three or four cutaways in every short, and a floor
# of three made "the fewest that is still acceptable" the thing the pipeline
# aimed at whenever the picker was cautious. A short landing exactly on the
# floor should be at the bottom of the intended range, not below it.
BROLL_FLOOR = 4
SENTENCE_MIN = 1.2

# The longest a single picture may hold. Lowered from 8s: a shot that runs
# that long stops reading as a cutaway and starts reading as a scene, and it
# is also long enough that most library clips have to loop to cover it.
SENTENCE_MAX = 4.5

# Break inside a sentence as well as at the end of one.
#
# A comma marks a place the speaker actually paused — it is the transcriber's
# record of the breath between clauses — so it is exactly where the picture can
# change without cutting across a thought. The comma never reaches the screen;
# captions strip it. It earns its keep here instead, by doubling the number of
# places a cutaway can land and so halving how long each one has to hold.
CLAUSE_BREAKS = (",", ";", ":", "—")

# How many sentences show the speaker. Everything else is b-roll.
#
# Not an alternation. Cutting speaker-b-roll-speaker-b-roll all the way down
# gives the speaker half the running time and makes the edit feel metronomic —
# the viewer learns the rhythm and stops being surprised by the picture. A
# couple of appearances is enough: one at the top so the face is established
# and the voice has an owner, one late so the person is still there at the end.
# The rest of the clip belongs to the footage.
SPEAKER_SHOTS = 2

# Where the later speaker shot falls, as a fraction of the way through. Near
# the end rather than at it — closing on footage lets the last line land over a
# picture instead of over a talking head.
SPEAKER_LATE_AT = 0.72

# How a cutaway is chosen, rather than merely taken next off the pile.
#
# The planner used to cycle the pool — broll_paths[index % len] — so a clip's
# position in a list decided which sentence it illustrated. The pool is already
# matched to the moment as a whole, but nothing matched a shot to the line it
# would actually sit under, and nothing stopped two near-identical shots landing
# back to back.

# Every clip starts here, before matching or penalties.
#
# The score ranks candidates; it is not an entrance exam. Written first as a
# hard gate at 0.18 with no base, it cut one cutaway into an eight-shot clip
# and left the rest on the speaker — because the speech is abstract and the
# descriptions are concrete, so most pairs share no words at all and scored
# zero. A shot that merely fails to match a line is still a fine shot.
BASE_SCORE = 0.25

# Below this a clip is refused and the next is tried. With the base above, only
# a clip the penalties have pushed down — the same film again, or nearly the
# same description — can fall this far.
MIN_MATCH = 0.05

# Charged per description word a clip shares with the one before it.
# `corridor-fluorescent` after `corridor-industrial` reads as one shot that
# jumped rather than as two shots.
SIMILARITY_PENALTY = 0.22

# Charged for following a clip from the same film. Three shots from one film in
# a row look like the library ran out, which it has not.
SAME_FILM_PENALTY = 0.30

# Under this, a cutaway has no room to land and reads as a flicker.
MIN_CUTAWAY_SECONDS = 1.4

# The shortest a shot may be. Anything under this is not a shot, it is a flash.
#
# The remainder of a span used to become its own shot whenever it exceeded a
# 0.35s guard, so a 2.69s span covered by a 2.3s clip left 0.39s — twelve
# frames of unrelated footage between two cutaways, which is what the flashing
# between transitions was.
#
# A leftover shorter than this is absorbed into the shot before it. That can
# mean a fraction of a second of the clip repeating, which is invisible; a
# twelve-frame cut to something else is not.
MIN_SHOT_SECONDS = 1.0
# One clip per cutaway, never reused inside a short. A repeat is noticeable:
# the same footage returning twenty seconds later reads as running out of
# material rather than as a deliberate callback. The ceiling exists only to
# stop a pathological case downloading a hundred files.
MAX_DISTINCT_BROLL = 24

# How many more clips to request than the trial plan counted. See
# `_broll_slots` for why the count is a floor rather than a total.
BROLL_REQUEST_MARGIN = 1.5


def _text_between(words: list, start: float, end: float) -> str:
    """What is being said across a span, for the shot list."""
    said = [w.text for w in words if w.start < end and w.end > start]
    return " ".join(said).strip()


def _sentences(words: list, total: float) -> list[tuple[float, float]]:
    """The clip split into spans that each end where a sentence ends.

    Punctuation first — it is what the transcriber already knows about the
    grammar and it costs nothing to read. A long silence closes a span too,
    because a speaker who stops for most of a second has finished a thought
    whether or not the transcript put a full stop on it.
    """
    if not words:
        return []

    spans: list[tuple[float, float]] = []
    start = max(0.0, float(words[0].start))

    for i, w in enumerate(words):
        text = (w.text or "").strip()
        last = i + 1 >= len(words)
        ends_sentence = text.endswith((".", "!", "?"))
        ends_clause = text.endswith(CLAUSE_BREAKS)
        long_pause = (not last
                      and float(words[i + 1].start) - float(w.end) >= 0.45)
        if last or ends_sentence or ends_clause or long_pause:
            end = float(w.end) if last else min(float(w.end) + 0.06,
                                                float(words[i + 1].start))
            if end - start > 0.05:
                spans.append((start, end))
            start = end

    if not spans:
        return []

    # Fold anything too short into its neighbour: a two-word sentence is a
    # beat, not a shot, and cutting on it produces a flicker.
    merged: list[list[float]] = [list(spans[0])]
    for a, b in spans[1:]:
        if b - a < SENTENCE_MIN and merged:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    if len(merged) > 1 and merged[0][1] - merged[0][0] < SENTENCE_MIN:
        merged[1][0] = merged[0][0]
        merged.pop(0)

    # And split anything too long, so one rambling sentence does not hold a
    # single picture for a third of the clip.
    out: list[tuple[float, float]] = []
    for a, b in merged:
        while b - a > SENTENCE_MAX:
            out.append((a, a + SENTENCE_MAX))
            a += SENTENCE_MAX
        out.append((a, b))

    # Cover the whole clip: the last span runs to the end, whatever the word
    # timings said, or the final shot would be missing.
    if out:
        out[0] = (0.0, out[0][1])
        out[-1] = (out[-1][0], max(total, out[-1][1]))
    return [(a, b) for a, b in out if b - a > 0.05]


def _usable_spans(total: float, words: list | None) -> list[tuple[float, float]]:
    """The sentence spans a plan is actually built from.

    Shared with the b-roll picker on purpose. The planner indexes *these*, not
    the raw output of `_sentences`, and building the picker's line list from the
    raw spans put every pick after a dropped one under the wrong line — a clip
    chosen for "I buried a mother, father, sister" landing on "I went to
    college", silently, with nothing in the output to show it. Two places
    deriving "the same" list separately is how that happens; one function is the
    fix.
    """
    epsilon = 0.2
    spans = _sentences(words or [], total) if SENTENCE_SHOTS else []
    return [(max(0.0, a), min(total, b)) for a, b in spans
            if min(total, b) - max(0.0, a) > epsilon]


def _shot_plan(
    total: float,
    speech_path: str,
    broll_paths: list[str],
    words: list | None = None,
    picks: dict | None = None,
    _backfilling: bool = False,
) -> list[tuple[str, float, float, str]]:
    """Alternate speaker and b-roll across the clip.

    Returns (path, source_start, seconds, kind) in playing order.

    Speaker shots carry their position in the clip as `source_start`, because
    cutting back to the speaker only works if their picture matches the audio
    being heard at that moment.

    With no b-roll available this degrades to a single continuous speaker shot,
    which is exactly the old behaviour.
    """
    plan: list[tuple[str, float, float, str]] = []
    index = 0
    chosen: set = set()
    previous: str | None = None
    backfilling = _backfilling
    # Anything shorter than this isn't a shot, it's a flicker.
    epsilon = 0.2

    usable = _usable_spans(total, words)
    if usable and broll_paths:
        # One picture per sentence. The speaker takes a couple of them and the
        # footage takes the rest.

        # The first sentence is always the speaker: a short that opens on stock
        # footage with a disembodied voice reads as a montage, and the face is
        # what says who is talking.
        speaker_at = {0}
        if SPEAKER_SHOTS >= 2 and len(usable) >= 3:
            late = min(len(usable) - 2, round(len(usable) * SPEAKER_LATE_AT))
            # Only if it leaves footage in between. On a three-sentence clip the
            # arithmetic lands on index 1, which puts two speaker shots back to
            # back — the picture then does not change on a sentence boundary at
            # all, which looks like the cut failed rather than like a choice.
            if late >= 2:
                speaker_at.add(late)

        for i, (a, b) in enumerate(usable):
            if i in speaker_at:
                plan.append((speech_path, a, b - a, "original"))
                continue

            # Cover a long span with several clips rather than one on repeat.
            #
            # B-roll inputs are opened with `-stream_loop -1`, so a two-second
            # clip asked to hold a six-second span simply plays three times.
            # That is what the looping-back looks like on screen, and it reads
            # as running out of material — which is the opposite of true here,
            # with over a thousand clips available.
            #
            # The clip is also slowed to BROLL_SPEED, so it covers more than its
            # own length: a 3s clip at 0.7x fills 4.3s before it would repeat.
            # What is said under this span, and whether it wants a picture.
            spoken_here = " ".join(
                w.text for w in (words or [])
                if a <= float(getattr(w, "start", 0.0)) < b
            )
            if not _deserves_cutaway(spoken_here, b - a):
                plan.append((speech_path, a, b - a, "original"))
                continue

            spoken_set = _content_words(spoken_here)
            remaining, at = b - a, 0.0
            # A pick made by reading the line beats one made by counting shared
            # words, so it is used when there is one — including its refusals.
            # `picks` holds an entry for every span the picker considered; a
            # span it declined has no entry and stays on the speaker, which is
            # the answer it meant rather than a gap to fill by other means.
            while remaining > MIN_SHOT_SECONDS / 2:
                if picks is not None and i in picks and not at:
                    # A clip chosen by reading this line.
                    clip = picks[i]
                    if clip in chosen:
                        clip = None
                elif picks is not None and i not in picks and not backfilling:
                    # The picker looked at this line and declined it. That is a
                    # real answer — a cutaway meaning nothing is worse than the
                    # speaker's face — so it is honoured, unless honouring every
                    # decline has left the short with too few cutaways to read
                    # as an edit at all. See BROLL_FLOOR.
                    clip = None
                else:
                    clip = _choose_broll(spoken_set, broll_paths, chosen, previous)
                if clip is None:
                    # Nothing suits this line. The speaker is never the wrong
                    # shot, so the rest of the span stays on the face.
                    plan.append((speech_path, a + at, remaining, "original"))
                    break
                chosen.add(clip)
                previous = clip
                covers = _broll_cover(clip)
                take = remaining if covers <= 0 else min(remaining, covers)
                # Never leave behind a piece too short to read as a shot.
                if remaining - take < MIN_SHOT_SECONDS:
                    take = remaining
                plan.append((clip, 0.0, take, "b-roll"))
                remaining -= take
                at += take
        if plan:
            return _merge_flashes(plan)

    t = 0.0

    while t < total - epsilon:
        span = min(SPEAKER_HOLD_SECONDS, total - t)
        plan.append((speech_path, t, span, "original"))
        t += span

        if not broll_paths:
            continue
        for _ in range(BROLL_RUN):
            if t >= total - epsilon:
                break
            span = min(BROLL_SHOT_SECONDS, total - t)
            plan.append((broll_paths[index % len(broll_paths)], 0.0, span, "b-roll"))
            index += 1
            t += span

    if not plan:
        plan = [(speech_path, 0.0, total, "original")]
    return plan


# Words that carry no picture. A caption made of these describes nothing, and a
# clip matched on them is matched on noise.
_STOPWORDS = {
    "that", "this", "with", "your", "from", "have", "will", "what", "when",
    "they", "them", "then", "than", "there", "here", "been", "being", "just",
    "like", "into", "about", "would", "could", "should", "because", "which",
    "their", "these", "those", "some", "more", "most", "very", "much", "only",
    "even", "also", "know", "think", "want", "make", "made", "does", "done",
    "going", "gonna", "really", "actually", "something", "anything", "everything",
}


def _content_words(text: str) -> set:
    """The words in a line that could plausibly describe a picture."""
    return {w for w in re.findall(r"[a-z]+", text.lower())
            if len(w) > 3 and w not in _STOPWORDS}


def _clip_tokens(path: str) -> set:
    """The description a clip carries, as words.

    The filename is `<film>-<NN>-<description>.mp4`, so the first two parts are
    dropped: the film code and the index describe provenance, not the picture,
    and matching on them would quietly group every shot from one film together.
    """
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    parts = stem.split("-")
    body = parts[2:] if len(parts) > 2 else parts
    return {w for w in body if len(w) > 2}


def _film_of(path: str) -> str:
    return os.path.basename(path).split("-")[0]


def _score_clip(clip: str, spoken: set, previous: str | None) -> float:
    """How well this clip suits this line, penalised for repeating the last one.

    Two parts, and the second is what stops a sequence reading as a montage of
    one thing. Three shots of the same film in a row look like the library ran
    out even when it did not, and two shots sharing most of their description —
    `corridor-fluorescent` after `corridor-industrial` — read as one shot that
    jumped rather than as an edit.
    """
    tokens = _clip_tokens(clip)
    if not tokens:
        return 0.0

    overlap = len(tokens & spoken)
    score = BASE_SCORE
    if spoken:
        score += overlap / (len(spoken) ** 0.5)

    if previous:
        prev_tokens = _clip_tokens(previous)
        # Charged on the *proportion* of the description shared, not the count.
        #
        # Per shared word it was 0.22 against a base of 0.25, so a single word
        # in common sank a clip to 0.03 and under the 0.05 floor — and in this
        # library the word in common is almost always "man" or "night", which
        # nearly every description carries. The effect was that the second
        # cutaway in a short was refused, then every cutaway after it, and the
        # shorts came out with one piece of b-roll and a talking head for the
        # rest. Nothing was wrong with the footage; the shots were rejected for
        # both containing a man.
        #
        # The floor is meant to catch "nearly the same description", which is a
        # ratio and not a tally. Sharing one word out of five is now a fifth of
        # the penalty and the clip still ranks and still plays; sharing all of
        # them spends the whole penalty and is refused, which is the case the
        # rule was written for.
        shared = len(tokens & prev_tokens)
        overlap_ratio = shared / max(1, min(len(tokens), len(prev_tokens)))
        score -= SIMILARITY_PENALTY * overlap_ratio
        if _film_of(clip) == _film_of(previous):
            score -= SAME_FILM_PENALTY
    return score


def _choose_broll(spoken: set, pool: list, used: set, previous: str | None):
    """The best unused clip for this line, or None to stay on the speaker.

    Returning None is the point of this function rather than a failure of it.
    The spec this follows is explicit: if no suitable footage exists, keep the
    original shot. A cutaway that does not belong is worse than no cutaway,
    because the viewer reads it as a mistake and the speaker's face is never a
    mistake.
    """
    # No clip twice in one short, full stop.
    #
    # This used to fall back to the whole pool once every clip had been used,
    # which is how a short ended up cutting to the same shot twice — four of
    # six in one batch did, one of them three times. Coming back to footage the
    # viewer saw twenty seconds ago reads as running out of material, and with
    # a 732-clip library that is not what happened; the planner simply asked
    # for fewer clips than the plan turned out to need.
    #
    # Running out now means staying on the speaker, which is the same answer
    # this function gives to every other kind of "nothing suitable". The real
    # remedy is upstream, in how many clips get requested — see `_broll_slots`.
    candidates = [c for c in pool if c not in used]
    if not candidates:
        return None

    ranked = sorted(candidates, key=lambda c: _score_clip(c, spoken, previous),
                    reverse=True)
    best = ranked[0]
    if _score_clip(best, spoken, previous) < MIN_MATCH:
        return None
    return best


def _deserves_cutaway(text: str, seconds: float) -> bool:
    """Whether this line is better served by footage than by the speaker.

    Three cases stay on the face, taken from how these edits actually read:

    - **A short line.** Under a beat and a half there is no room for a cutaway
      to land; it registers as a flicker rather than as a picture.
    - **A question.** The speaker is addressing the viewer, and the answer is in
      their face — cutting away throws away the only thing being offered.
    A line like "You have to decide" describes nothing filmable and ideally
    would stay on the face too, but this function does not catch it: it asks
    whether there is *any* content word, not whether that word is concrete, and
    "decide" passes. What actually protects that case is the match threshold in
    `_choose_broll` — an abstract line scores nothing against a library of
    concrete descriptions, so no clip clears MIN_MATCH and the speaker is kept.
    Two loose gates in series, rather than one strict one.
    """
    if seconds < MIN_CUTAWAY_SECONDS:
        return False
    stripped = text.strip()
    if stripped.endswith("?"):
        return False
    return bool(_content_words(stripped))


def _merge_flashes(plan: list) -> list:
    """Fold any shot too short to read into the one before it.

    A guarantee applied to the finished plan rather than a rule enforced in
    three places while it is being built. Short shots arrived from two
    directions — a leftover slice of a span that no clip quite covered, and a
    span too brief to earn a cutaway which then became its own shot of the
    speaker — and each was fixed separately while the other kept producing
    them.

    Merging into the *previous* shot keeps the total length identical, so the
    picture still ends where the speech does. The absorbed time simply extends
    a shot that was already on screen, which is invisible; a twelve-frame cut
    to something else is not.
    """
    if not plan:
        return plan

    merged: list = []
    for shot in plan:
        path, source_start, seconds, kind = shot
        if merged and seconds < MIN_SHOT_SECONDS:
            p0, s0, d0, k0 = merged[-1]
            merged[-1] = (p0, s0, d0 + seconds, k0)
            continue
        merged.append((path, source_start, seconds, kind))

    # A first shot that is itself too short has nothing before it to join, so
    # it takes from the one after instead.
    while len(merged) > 1 and merged[0][2] < MIN_SHOT_SECONDS:
        p0, s0, d0, k0 = merged.pop(0)
        p1, s1, d1, k1 = merged[0]
        merged[0] = (p1, s1, d1 + d0, k1)
    return merged


def _broll_cover(path: str) -> float:
    """How many seconds of screen time one b-roll clip can fill without looping.

    Its own length divided by the playback rate, because the cutaways are
    slowed: a three-second clip at 0.7x covers 4.3 seconds before it would have
    to start again.

    Returns 0.0 when the length cannot be read, which the caller treats as "no
    limit" — a shot that is slightly too long is a much smaller problem than a
    plan that refuses to cover the clip.
    """
    try:
        from asset_library import probe
        from shorts_builder import BROLL_SPEED

        duration = probe(path)[0]
        if duration <= 0:
            return 0.0
        return duration / max(0.05, BROLL_SPEED)
    except Exception:
        return 0.0


def _broll_slots(total: float, words: list | None = None) -> int:
    """How many distinct b-roll clips a short of this length needs.

    Counted from the plan itself rather than estimated, so the fetch asks for
    exactly as many as there are cutaways and nothing has to repeat.

    **The placeholders must be distinct, and that is the whole function.** This
    asked for the plan with a pool of one — `["placeholder"]` — and a pool of
    one cannot produce more than one cutaway: the second span finds every
    candidate used, falls back to the same clip, and is then refused by the
    similarity and same-film penalties for repeating itself. So this returned 1
    for every short of any length, exactly one clip was fetched, and the real
    plan was then built against that one clip and refused a second cutaway for
    the same reason. A self-fulfilling count — the shorts were not choosing one
    cutaway, they were being told only one existed.

    The names are shaped like real clips (`ph3-00-slot3.mp4`) because the
    scorer reads them like real clips: it takes the film from the leading token
    and the description from everything after the number, so each placeholder
    has to differ in both or the penalties reappear here instead.
    """
    pool = [f"ph{i}-00-slot{i}.mp4" for i in range(MAX_DISTINCT_BROLL)]
    plan = _shot_plan(total, "SPEECH", pool, words)
    cutaways = sum(1 for _p, _s, _d, kind in plan if kind == "b-roll")
    # Ask for half again as many, because this count is a floor and not a total.
    #
    # A placeholder has no duration, so `_broll_cover` reads it as "no limit"
    # and the trial plan gives every span a single clip. Real clips are finite:
    # a 2s clip cannot hold a 4.5s span, so that span becomes two shots wanting
    # two clips, and the count comes back short of what the real plan needs.
    #
    # Being short used to be invisible because the planner quietly reused a
    # clip. Now that it refuses to, being short costs cutaways instead — so the
    # margin is what actually keeps them. Clips that go unused are pruned from
    # the short's `broll/` folder afterwards, so over-asking costs a copy and
    # not a cluttered folder.
    wanted = int(cutaways * BROLL_REQUEST_MARGIN + 0.999)
    return min(max(1, wanted), MAX_DISTINCT_BROLL)


def _shotlist(
    plan: list[tuple[str, float, float, str]],
    words: list,
    hook: str,
) -> str:
    """The edit, written out: what is on screen when, and what is being said.

    This is what makes the folder assemblable rather than a pile of files, and
    what lets one shot be swapped without rebuilding the edit.
    """
    lines: list[str] = [f"# Shot list - {hook}", ""]
    lines.append("The speaker's audio runs underneath the whole way; only the")
    lines.append("picture changes. Swap any b-roll shot you don't like and keep")
    lines.append("the timings.")
    lines.append("")

    t = 0.0
    for path, _source_start, seconds, kind in plan:
        end = t + seconds
        said = _text_between(words, t, end)
        label = "ORIGINAL (the speaker)" if kind == "original" else f"broll/{os.path.basename(path)}"
        lines.append(f"### {t:.1f}s - {end:.1f}s   ({seconds:.1f}s)")
        lines.append("")
        lines.append(f"**On screen:** {label}")
        if said:
            lines.append("")
            lines.append(f"> {said}")
        lines.append("")
        t = end

    return "\n".join(lines)


def _clip_brief(
    moment: Moment,
    index: int,
    url: str,
    segments: list[dict],
    talk_title: str,
    channel: str,
    tags=None,
) -> str:
    """An editing brief for whoever cuts the final video.

    The clip on its own doesn't say why it was chosen, where it sits in the
    talk, or what would break if it were trimmed further. An editor working
    without that will cut the setup off the front — which is exactly the
    mistake the peak-expansion logic exists to avoid — so the reasoning ships
    alongside the file.
    """
    lines: list[str] = []
    add = lines.append

    add(f"# Clip {index:02d} — {moment.hook}")
    add("")
    if talk_title:
        add(f"**Source:** {talk_title}" + (f" ({channel})" if channel else ""))
    add(f"**Watch the original:** {url}&t={int(moment.cuts[0].start_seconds)}s")
    add(f"**Length:** {moment.duration:.1f}s across {len(moment.cuts)} cut(s)")
    add(f"**Theme:** {moment.theme}   **Tone:** {moment.tone}")
    if moment.peak_rank:
        add(f"**Replay peak #{moment.peak_rank}** — this is a stretch the audience "
            f"rewatched, not a passage picked from reading the transcript.")
    add("")

    add("## Where it comes from")
    add("")
    for n, cut in enumerate(moment.cuts, 1):
        add(f"- Cut {n}: `{cut.start_seconds:.1f}s` to `{cut.end_seconds:.1f}s` "
            f"({cut.duration:.1f}s) — {url}&t={int(cut.start_seconds)}s")
    if moment.stitch_reason:
        add("")
        add(f"**Why these join:** {moment.stitch_reason}")
    add("")

    add("## What is said")
    add("")
    spoken = _spoken_text(moment, segments)
    for para in spoken.split("\n\n"):
        add(f"> {para}" if not para.startswith("[") else para)
    add("")

    add("## Why this one")
    add("")
    add(moment.reason or "—")
    add("")

    add("## Constraints — read before editing")
    add("")
    add("- **Do not trim the opening.** The cut already starts where the thought "
        "begins, not where the strong line lands. Cutting into it leaves the "
        "payoff sounding like a non-sequitur.")
    add("- **Do not trim the ending.** It stops on a finished sentence. Ending "
        "early is the most common way a clip stops feeling complete.")
    if len(moment.cuts) > 1:
        joined = moment.cuts[0].duration
        add(f"- **There is a hard cut at {joined:.1f}s.** The audio jumps to a "
            "different part of the talk. Change the picture on that beat so the "
            "jump reads as an edit rather than a mistake.")
    add("- **Keep the speaker's audio continuous.** It carries the whole clip; "
        "any music sits under it, never over it.")
    add("- Aim to stay under 25s. Past that, completion rate falls and the "
        "platform stops pushing it.")
    add("")

    add("## Footage to cut over the top")
    add("")
    add("The speaker's voice runs underneath; the picture is yours to choose.")
    add("")
    queries = list(moment.visual_keywords)
    if tags is not None:
        for q in getattr(tags, "visual_queries", []):
            if q not in queries:
                queries.append(q)
    for q in queries:
        add(f"- {q}")
    add("")

    if tags is not None and getattr(tags, "hashtags", None):
        add("## Hashtags")
        add("")
        add(" ".join(f"#{t}" for t in tags.hashtags))
        add("")

    return "\n".join(lines)


def _publishing_set(data: dict, video_dir: str) -> list[str]:
    """Write the post-everywhere formats for the whole talk.

    The clips are only half the job. A short goes out with a caption on
    TikTok, a description on YouTube, a thread on X and a carousel on
    Instagram, and writing those by hand for every talk is most of the work
    that is left after the video is cut.

    All of it hangs off one `ContentBrief`, which is what every generator in
    this project already expects — so the brief is produced first and the rest
    read from it. That is also why they agree with each other: the blog post
    and the thread are not independent takes on the same video, they are the
    same understanding of it in different shapes.

    The VideoData is rebuilt from transcript.json rather than re-fetched. It
    was written from exactly that object, so the round trip is lossless and
    costs nothing.

    Every step is caught separately: a thread that fails should not take the
    blog post down with it, and whatever did work stays on disk.
    """
    from youtube_extractor import VideoData

    written: list[str] = []

    def save(name: str, text: str) -> None:
        path = os.path.join(video_dir, name)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
        written.append(name)

    import dataclasses

    fields = {f.name for f in dataclasses.fields(VideoData)}
    video = VideoData(**{k: data[k] for k in fields if k in data})
    title = data.get("title", "")

    from content_brief import generate_brief

    print("  · Content brief...")
    brief = generate_brief(video)
    save("brief.json", brief.to_json())

    print("  · Blog post...")
    try:
        from blog_generator import generate_blog_post

        save("blog_post.md", generate_blog_post(brief, video_title=title))
    except Exception as e:
        print(f"    ⚠ blog post failed: {e}")

    print("  · X / Twitter thread...")
    try:
        from twitter_thread_generator import generate_twitter_thread

        thread = generate_twitter_thread(brief)
        save("twitter_thread.json", thread.to_json())
        save("twitter_thread.txt", thread.to_text())
    except Exception as e:
        print(f"    ⚠ thread failed: {e}")

    print("  · Captions and descriptions...")
    try:
        from captions_generator import generate_captions

        captions = generate_captions(brief)
        save("captions.json", captions.to_json())
        save("captions.txt", captions.to_text())
    except Exception as e:
        print(f"    ⚠ captions failed: {e}")

    return written


def build_quote_carousel(
    moments: list[Moment],
    out_dir: str,
    talk_title: str = "",
    channel: str = "",
    fetch: bool = True,
) -> list[str]:
    """Write the carousel as words, not pictures.

    This used to search the asset library for a background photo per card and
    render finished PNGs. It no longer does either: the designing happens by
    hand afterwards, so rendering images here produced work that was thrown
    away, and the image search was one of the slowest steps in a run — several
    seconds per card, sometimes a stock API call.

    What comes out is the copy for each card, ready to paste into whatever
    design tool is being used.
    """
    if not moments:
        return []

    os.makedirs(out_dir, exist_ok=True)

    cards: list[dict] = [{
        "card": 1,
        "role": "hook",
        "headline": moments[0].hook,
        "subtext": talk_title,
        "source": channel,
    }]

    for moment in moments:
        cards.append({
            "card": len(cards) + 1,
            "role": "quote",
            "headline": moment.quote or moment.hook,
            "subtext": moment.hook if moment.quote else "",
            "source": channel,
        })

    cards.append({
        "card": len(cards) + 1,
        "role": "close",
        "headline": "Save this.",
        "subtext": "Then go do the work.",
        "source": channel,
    })

    json_path = os.path.join(out_dir, "carousel.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"cards": cards}, f, indent=2, ensure_ascii=False)

    lines: list[str] = []
    for card in cards:
        lines.append(f"--- Card {card['card']} ({card['role']}) ---")
        lines.append(card["headline"])
        if card["subtext"]:
            lines.append(f"    {card['subtext']}")
        lines.append("")
    text_path = os.path.join(out_dir, "carousel.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return [text_path, json_path]


def make_shorts(
    url: str,
    count: int | None = None,
    style: str = "broll",
    output_dir: str = "output",
    model_size: str = "base",
    fetch: bool = True,
    carousel: bool = True,
    formats: bool = True,
    min_seconds: int | None = None,
    max_seconds: int | None = None,
    source_file: str | None = None,
    workers: int = DEFAULT_WORKERS,
    captions: bool = False,
    keep_old: bool = False,
    moments_file: str | None = None,
) -> list[dict]:
    from video_clipper import download_clip, download_full, cut_from_file, YouTubeBlockedDownload

    # Footage already claimed by another clip in this run. Two shorts from the
    # same talk opening on the same stock runner looks like a template, so the
    # set is shared and the lock keeps three workers from claiming one clip.
    used_broll: set = set()
    BROLL_LOCK = threading.Lock()
    from youtube_extractor import extract_video_id

    video_id = extract_video_id(url)
    print(f"→ Building shorts for {video_id}")

    data = load_or_fetch_transcript(url, output_dir)
    segments = data.get("transcript_segments") or []
    if not segments:
        print("✗ This video has no timestamped transcript, so moments can't be located.")
        sys.exit(1)

    # Ask YouTube where the audience actually rewatched. Not every video has a
    # heatmap (it needs a view threshold), so this is best-effort — without it
    # the moments are chosen from the transcript alone, as before.
    hot_windows = None
    try:
        import heatmap as heatmap_module

        hot_windows = heatmap_module.for_video(url, segments=segments, count=8)
        if hot_windows:
            print(f"  ✓ Most-replayed data: {len(hot_windows)} peak(s) to cut around")
            for w in hot_windows[:3]:
                print(f"     {w['start']:.0f}s-{w['end']:.0f}s (intensity {w['value']:.2f})")
        else:
            print("  · No most-replayed data for this video — using the transcript alone")
    except Exception as e:
        print(f"  ⚠ Couldn't read most-replayed data ({e}); using the transcript alone")

    if count is None:
        peaks = len(hot_windows) if hot_windows else 0
        print(
            f"→ Finding moments with Claude ({peaks} replay peak(s) to work through)..."
            if peaks else
            "→ Finding moments with Claude (no replay data — reading the transcript)..."
        )
    else:
        print(f"→ Finding the {count} strongest moments with Claude...")
    from moment_finder import MAX_TOTAL_SECONDS, MIN_TOTAL_SECONDS

    if moments_file:
        # A pinned set, rendered as given.
        #
        # The model returns different hooks on every call — the same four
        # passages come back, worded four different ways — so without a way to
        # save one there is no way to keep a set you liked. Review a few passes,
        # save the best, edit the wording by hand if you want, render from that.
        # Nothing here is re-judged: the strength bar and the ceiling have
        # already done their work by the time a set is saved.
        from moment_finder import Moment
        with open(moments_file, encoding="utf-8") as f:
            saved = json.load(f)
        if isinstance(saved, dict):
            saved = saved.get("moments", [])
        moments = [Moment.from_dict(d) for d in saved]
        print(f"→ Using {len(moments)} pinned moment(s) from "
              f"{os.path.basename(moments_file)}")
    else:
        moments = find_moments(
            segments,
            count=count,
            title=data.get("title", ""),
            hot_windows=hot_windows,
            min_total=min_seconds or MIN_TOTAL_SECONDS,
            max_total=max_seconds or MAX_TOTAL_SECONDS,
        )
    if not moments:
        print("✗ No usable moments came back.")
        sys.exit(1)
    print(f"  ✓ {len(moments)} moments found")
    for i, m in enumerate(moments, 1):
        heat = f" · replay {m.heat:.2f}" if m.heat else ""
        shape = f"{len(m.cuts)} cuts" if len(m.cuts) > 1 else "1 cut"
        print(f"     {i}. {m.duration:.0f}s ({shape}) {m.theme}{heat}: {m.hook}")

    asset_library.ensure_dirs()
    asset_library.scan()

    shorts_dir = os.path.join(output_dir, video_id, "shorts")
    os.makedirs(shorts_dir, exist_ok=True)
    results: list[dict] = []

    # Fetch the talk once, then take every cut from the local file.
    #
    # A ranged download costs 20-30s here, of which only ~7s is transfer: the
    # rest is yt-dlp re-extracting the video and re-solving YouTube's JS
    # challenge, paid again per cut. One full download pays that once, and it's
    # cached, so re-running a talk later costs no network at all.
    total_cuts = sum(len(m.cuts) for m in moments)
    # The source is now fetched even for a single cut, because the ending
    # refinement below listens to the audio around each cut and needs the file
    # to do it. It is cached, so a re-run costs nothing either way.
    if source_file is None:
        candidate = os.path.join(output_dir, video_id, "source.mp4")
        cached = os.path.isfile(candidate)
        print(
            f"  ✓ Reusing the downloaded talk"
            if cached else
            f"→ Downloading the talk once, for all {total_cuts} cuts..."
        )
        try:
            source_file = download_full(url, candidate)
            if not cached:
                size = os.path.getsize(source_file) / 1e6
                print(f"  ✓ Got the source ({size:.0f} MB) — cuts are local from here")
        except Exception as e:
            # Not fatal: fall back to the per-cut path, just slower.
            print(f"  ⚠ Couldn't fetch the whole talk ({e}); cutting per clip instead")
            source_file = None

    # Put every ending on the moment the speaker actually finishes.
    #
    # This has to happen here — after the moments are chosen, before anything
    # is cut — because it changes where the cuts are. It is also the first
    # point where both things it needs exist at once: the chosen moments, and
    # the audio to check them against.
    #
    # It runs in sequence rather than inside the parallel builders. Whisper is
    # already serialised by a lock, so there is nothing to gain from spreading
    # it out, and doing it up front means the printed cut list matches what is
    # actually built.
    if source_file:
        from ending_finder import refine_moment

        print("→ Checking every ending against the audio...")
        for i, moment in enumerate(moments, 1):
            before = moment.duration
            try:
                refine_moment(
                    moment,
                    source_file,
                    model_size=model_size,
                    max_total=max_seconds or MAX_TOTAL_SECONDS,
                    log=lambda msg, n=i: print(f"     {n}. {msg}"),
                )
            except Exception as e:
                # Refining an ending is an improvement, not a requirement: the
                # moment already has an end time from the transcript, and a
                # slightly loose one costs a beat of silence. Letting this
                # abort the run costs every short instead.
                #
                # It went unguarded until Windows Application Control blocked
                # both Whisper engines at once, at which point this line — a
                # tidy-up that runs before a single clip is cut — took three
                # whole talks down with it.
                print(f"  ⚠ Endings left as found ({type(e).__name__}); "
                      f"continuing without refinement.")
                break
            after = moment.duration
            if abs(after - before) >= 0.05:
                print(f"        now {after:.1f}s (was {before:.1f}s)")

    def build_one(i: int, moment: Moment) -> tuple[dict | None, list[str]]:
        """Produce one short. Returns (record, its log lines).

        The log is collected instead of printed because several of these run at
        the same time, and interleaved output from three clips at once is
        unreadable. Each clip's story is buffered and printed whole when it
        finishes, so the transcript still reads top to bottom.
        """
        log: list[str] = []
        say = log.append

        folder = os.path.join(shorts_dir, f"{i:02d}_{_slug(moment.theme + '_' + moment.hook)}")
        os.makedirs(folder, exist_ok=True)
        say(f"→ Short {i}/{len(moments)}: {moment.hook}")

        raw_clip = os.path.join(folder, "clip_raw.mp4")

        # A cached clip is only reusable if it was cut at the timings this run
        # is asking for. That used to be a safe assumption and no longer is:
        # the endings are refined against the audio before anything is cut, so
        # the same moment can come back with a different end and the file on
        # disk would silently keep the old one — the exact clipped ending this
        # is meant to fix, preserved by the cache.
        wanted = [[c.start_seconds, c.end_seconds] for c in moment.cuts]
        record_path = os.path.join(folder, "moment.json")
        cached_cuts = None
        if os.path.isfile(record_path):
            try:
                with open(record_path, encoding="utf-8") as f:
                    previous = json.load(f)
                cached_cuts = [
                    [c["start_seconds"], c["end_seconds"]]
                    for c in previous.get("cuts", [])
                ]
            except (OSError, ValueError, KeyError, TypeError):
                cached_cuts = None

        if os.path.isfile(raw_clip) and cached_cuts == wanted:
            say("  ✓ Clip already cut at these timings")
        else:
            if os.path.isfile(raw_clip):
                say("  · Timings changed — recutting")
                try:
                    os.remove(raw_clip)
                except OSError:
                    pass
            pieces: list[str] = []
            failed = None
            for n, cut in enumerate(moment.cuts, start=1):
                label = f"cut {n}/{len(moment.cuts)}" if len(moment.cuts) > 1 else "clip"
                piece = (
                    raw_clip if len(moment.cuts) == 1
                    else os.path.join(folder, f"cut_{n:02d}.mp4")
                )
                if source_file:
                    # Everything comes out of the one local file — no network,
                    # so no retry loop and nothing to be blocked.
                    say(f"  · Cutting {label}: {cut.duration:.0f}s from {cut.start_seconds:.0f}s...")
                    try:
                        cut_from_file(source_file, cut.start_seconds, cut.end_seconds, piece)
                    except Exception as e:
                        failed = e
                        break
                    pieces.append(piece)
                    continue

                say(f"  · Downloading {label}: {cut.duration:.0f}s from {cut.start_seconds:.0f}s...")

                # The ranged download shells out to ffmpeg through yt-dlp and
                # occasionally crashes on a window that downloads fine on a
                # second attempt, so one retry rather than losing the moment.
                last_error = None
                for attempt in (1, 2):
                    try:
                        download_clip(url, cut.start_seconds, cut.end_seconds, piece)
                        last_error = None
                        break
                    except YouTubeBlockedDownload as e:
                        # An access block is deterministic — a second attempt
                        # fails the same way, and every remaining cut will too.
                        last_error = e
                        break
                    except Exception as e:
                        last_error = e
                        if attempt == 1:
                            say(f"    · failed ({e}); retrying once...")
                if last_error is not None:
                    failed = last_error
                    break
                pieces.append(piece)

            if failed is not None:
                say(f"  ⚠ Couldn't download this moment: {failed}")
                return None, log

            if len(pieces) > 1:
                try:
                    join_clips(pieces, raw_clip)
                    say(f"  ✓ {len(pieces)} cuts stitched")
                    for piece in pieces:
                        try:
                            os.remove(piece)
                        except OSError:
                            pass
                except Exception as e:
                    say(f"  ⚠ Couldn't stitch the cuts together: {e}")
                    return None, log
            else:
                say("  ✓ Clip ready")

        # Render against what actually landed on disk, not the length planned
        # from the transcript. Ranged downloads snap to keyframes, so a cut
        # asked for as 18.0s often arrives as 17.6s, and rendering the planned
        # length against a shorter file freezes the last frame.
        actual = probe_duration(raw_clip)
        render_duration = actual if actual > 0.5 else moment.duration
        if actual > 0.5 and abs(actual - moment.duration) > 0.4:
            say(f"  · Clip is {actual:.1f}s (planned {moment.duration:.0f}s) — rendering to the real length")

        # Whisper always runs now. The SRT is a deliverable in its own right —
        # it is what saves retyping every caption by hand in CapCut — and the
        # word timings are also what the shot list cuts the picture on. The
        # `captions` flag only decides whether they are ALSO burned into the
        # video, which is off by default because baked-in text fights the edit.
        words: list = []
        captions_path = None
        try:
            from caption_timing import build_ass, transcribe_words, words_to_srt

            words = transcribe_words(raw_clip, model_size=model_size)

            # Squeeze out the dead air before anything else uses these timings.
            #
            # It has to happen here, between transcribing and everything
            # downstream: the SRT, the shot plan and the render all read these
            # word times, and they must all describe the same file. The word
            # timings are remapped arithmetically rather than by transcribing
            # the tightened clip again, which would cost a second Whisper pass
            # to rediscover something already known exactly.
            if TIGHTEN_SILENCE:
                try:
                    from silence_trimmer import plan_keep_ranges, remap_words, tighten
                    from shorts_builder import video_encoder_args

                    ranges = plan_keep_ranges(words, render_duration)
                    tight_path = os.path.join(folder, "clip_tight.mp4")
                    used, saved = tighten(
                        raw_clip, words, render_duration, tight_path,
                        encoder_args=video_encoder_args(),
                    )
                    if saved > 0:
                        raw_clip = used
                        words = remap_words(words, ranges)
                        render_duration = max(0.5, render_duration - saved)
                        say(f"  ✓ Tightened {saved:.1f}s of dead air → {render_duration:.1f}s")
                except Exception as e:
                    say(f"  ⚠ Couldn't tighten silences ({str(e)[:60]}) — using the full cut")

            # Split the speaker's voice out of the mix.
            #
            # These talks are scored: music plays under the speaking, and no
            # amount of filtering removes it, because music occupies the same
            # frequencies as the voice. A separation model does, and measured
            # on this footage it takes 28 dB out of the gaps between sentences
            # while costing the speech 0.4 dB.
            #
            # After the silence trim on purpose. Separation runs at roughly
            # twice real time on this machine, so it is much cheaper on the
            # shortened clip, and the word timings are unaffected either way.
            if ISOLATE_VOICE:
                try:
                    import subprocess as _sp

                    from voice_isolator import isolate_vocals

                    vocal_wav = os.path.join(folder, "_vocals.wav")
                    if isolate_vocals(raw_clip, vocal_wav):
                        voiced = os.path.join(folder, "clip_voice.mp4")
                        muxed = _sp.run(
                            ["ffmpeg", "-y", "-nostdin", "-hide_banner",
                             "-loglevel", "error",
                             "-i", raw_clip, "-i", vocal_wav,
                             "-map", "0:v:0", "-map", "1:a:0",
                             "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                             "-shortest", voiced],
                            capture_output=True,
                        )
                        if muxed.returncode == 0 and os.path.isfile(voiced):
                            raw_clip = voiced
                            say("  ✓ Voice separated from the music")
                        try:
                            os.remove(vocal_wav)
                        except OSError:
                            pass
                    else:
                        say("  · Voice separation unavailable — keeping the mix")
                except Exception as e:
                    say(f"  ⚠ Voice separation failed ({str(e)[:60]}) — keeping the mix")

            words_to_srt(words, os.path.join(folder, "captions.srt"))

            # The spoken words as plain prose, to be copied into whatever
            # subtitles the clip gets by hand.
            #
            # Not a duplicate of the SRT beside it: an SRT carries timings and
            # cue numbers, which is what an editor imports, while this is what
            # gets pasted into a caption box or a post. Selecting the words out
            # of an SRT means dragging past a timestamp every second line.
            try:
                spoken = " ".join(w.text.strip() for w in words if w.text.strip())
                # Break after a sentence ends so it reads as paragraphs rather
                # than one unbroken wall of text.
                spoken = re.sub(r"(?<=[.!?]) +", "\n", spoken)
                with open(os.path.join(folder, "transcript.txt"), "w",
                          encoding="utf-8") as f:
                    f.write(spoken.strip() + "\n")
            except Exception as e:
                say(f"  ⚠ Couldn't write transcript.txt: {e}")

            say(f"  ✓ {len(words)} words timed → captions.srt, transcript.txt")
            if captions:
                captions_path = os.path.join(folder, "captions.ass")
                build_ass(words, captions_path, hook=moment.hook)
        except Exception as e:
            # Whisper is the most fragile piece here — it pulls in numba, which
            # Windows Application Control has blocked outright. Falling back to
            # transcript timings keeps the shot list and a usable SRT; only the
            # word-level precision is lost.
            say(f"  ⚠ Whisper unavailable ({str(e)[:60]}) — using transcript timings")
            try:
                from caption_timing import words_to_srt

                words = _words_from_segments(moment, segments)
                if words:
                    words_to_srt(words, os.path.join(folder, "captions.srt"))
                    say(f"  ✓ {len(words)} caption lines → captions.srt (line-level, not word-level)")
            except Exception as inner:
                say(f"  ⚠ Transcript timings failed too: {inner}")

        # Hashtags for the caption, visual queries for the footage search — two
        # different jobs, one call. Failing here shouldn't cost you the short.
        tags = None
        try:
            import topic_tags
            # The talk's own title and channel tell the query writer what world
            # this lives in. Without it, a line about consistency from a
            # basketball player produced "athlete meal prep" and returned stock
            # footage of vegetables being chopped.
            talk_context = " — ".join(
                part for part in (data.get("title", ""), data.get("channel", "")) if part
            )
            tags = topic_tags.for_moment(moment, context=talk_context)
            with open(os.path.join(folder, "hashtags.json"), "w", encoding="utf-8") as f:
                json.dump(tags.to_dict(), f, indent=2, ensure_ascii=False)
            say(f"  ✓ {len(tags.hashtags)} hashtags, {len(tags.visual_queries)} visual queries")
        except Exception as e:
            say(f"  ⚠ Tag generation failed: {e}")

        try:
            with open(os.path.join(folder, "brief.md"), "w", encoding="utf-8") as f:
                f.write(_clip_brief(
                    moment, i, url, segments,
                    data.get("title", ""), data.get("channel", ""), tags,
                ))
            say("  ✓ brief.md")
        except Exception as e:
            say(f"  ⚠ Couldn't write the brief: {e}")

        # The post caption, ready to paste.
        try:
            caption_lines = [moment.hook, ""]
            if moment.quote:
                caption_lines += [f"“{moment.quote}”", ""]
            if tags is not None and tags.hashtags:
                caption_lines.append(" ".join(f"#{t}" for t in tags.hashtags))
            with open(os.path.join(folder, "caption.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(caption_lines).strip() + "\n")
        except Exception as e:
            say(f"  ⚠ Couldn't write the caption: {e}")

        # The comment to post under your own short, and pin.
        #
        # A short lands on a feed with no thread under it, and the first comment
        # decides whether one forms. Left alone it is usually praise for the
        # speaker, which reads well and generates nothing, because there is no
        # reply to praise. An open question gives every later viewer something
        # obvious to do.
        #
        # Written per clip rather than once per talk: the question has to be
        # about *this* statement. One that would fit any motivational video is
        # a wasted comment.
        try:
            import comment_prompt

            prompt = comment_prompt.for_moment(
                moment,
                means=(tags.means if tags is not None else ""),
                context=talk_context if tags is not None else "",
            )
            with open(os.path.join(folder, "comment.txt"), "w", encoding="utf-8") as f:
                f.write(prompt.to_text())
            with open(os.path.join(folder, "comment.json"), "w", encoding="utf-8") as f:
                json.dump(prompt.to_dict(), f, indent=2, ensure_ascii=False)
            say(f"  ✓ comment to pin: {prompt.question}")
        except Exception as e:
            say(f"  ⚠ Couldn't write the comment: {e}")

        # --- b-roll, one distinct shot per beat --------------------------------
        broll_local: list[str] = []
        credits: dict = {}

        if fetch:
            # Context-aware queries lead. `moment.visual_keywords` is written
            # while reading the transcript alone, before anything knows who is
            # speaking — that is where "athlete meal prep" came from, and why
            # it returned footage of vegetables being chopped for a basketball
            # interview. Kept only as a fallback when the tag call failed.
            queries: list[str] = []
            if tags is not None:
                queries = list(getattr(tags, "visual_queries", []))
            if not queries:
                queries = list(moment.visual_keywords)
            picked = []
            wanted = _broll_slots(render_duration, words)
            try:
                # Serialised: three workers hitting the same stock API at once
                # gets rate-limited, and the shared "already used" set has to be
                # read and written without two clips claiming the same footage.
                with BROLL_LOCK:
                    # Hand-picked footage always wins. Stock only fills the gap
                    # it leaves, so the library gets better every time you add
                    # to assets/broll/ without anything else changing.
                    picked = asset_library.curated_broll(
                        queries, count=wanted, exclude=used_broll,
                        theme=moment.theme,
                    )
                    # The library is finite. Once earlier shorts have claimed
                    # most of it, insisting on unused clips pushes the rest of
                    # the run onto stock — which is worse than a clip appearing
                    # in two different videos that get posted days apart.
                    # Repeats within a single short are what actually look
                    # cheap, and those are prevented by asking for `wanted`
                    # distinct clips here.
                    if len(picked) < wanted:
                        already = {a.id for a in picked}
                        picked += asset_library.curated_broll(
                            queries, count=wanted - len(picked), exclude=already,
                            theme=moment.theme,
                        )
                    # Stock is not a fallback any more, it is excluded.
                    #
                    # The library used to top up from Pexels whenever the
                    # curated folder ran short, which is why 104 stock clips
                    # accumulated there and were eventually deleted by hand.
                    # Leaving the top-up in would quietly refill the folder on
                    # the next run and undo that.
                    #
                    # Running short now means fewer cutaways, which is the right
                    # failure: a shot from the films is the point, and a stock
                    # clip that reads as an advert costs more than a repeat.
                    if STOCK_BROLL and len(picked) < wanted:
                        for asset in picked:
                            used_broll.add(asset.id)
                        picked += asset_library.fetch_broll_set(
                            queries, count=wanted - len(picked), exclude=used_broll
                        )
                    for asset in picked:
                        used_broll.add(asset.id)
            except Exception as e:
                say(f"  ⚠ B-roll fetch failed: {e}")

            curated_count = sum(1 for a in picked if a.source == "curated")
            if curated_count:
                say(f"  ✓ {curated_count} curated + {len(picked) - curated_count} stock")

            if picked:
                import shutil as _shutil

                broll_dir = os.path.join(folder, "broll")
                os.makedirs(broll_dir, exist_ok=True)

                # Clear what a previous run left behind.
                #
                # Selection changes between runs — the library gets pruned, the
                # tags change, the model picks differently — so the old files
                # are not merely redundant, they are footage this edit does not
                # use. Left in place they accumulate: eighty-one stale clips
                # across one batch, so a folder opened in CapCut showed
                # thirty-four options when eleven were used. That folder is
                # meant to be the shot list made physical.
                for old in os.listdir(broll_dir):
                    if os.path.splitext(old)[1].lower() in asset_library.VIDEO_EXTS:
                        try:
                            os.remove(os.path.join(broll_dir, old))
                        except OSError:
                            pass

                for n, asset in enumerate(picked, 1):
                    target = os.path.join(
                        broll_dir, f"{n:02d}_{os.path.basename(asset.path)}"
                    )
                    try:
                        if not os.path.isfile(target):
                            _shutil.copyfile(asset.abs_path, target)
                        broll_local.append(target)
                    except OSError:
                        continue
                credits["broll"] = [
                    {"path": a.path, "source": a.source, "credit": a.credit,
                     "licence": a.licence}
                    for a in picked
                ]

        # Ask a model which clip belongs under which line, before planning.
        #
        # Word overlap picks the pool; meaning picks the shot. Measured on one
        # batch, 65-90% of the library scored zero shared words against a given
        # moment, so once the few real matches ran out the rest of the plan was
        # arbitrary — a retirement short cut over hands holding a fish. See
        # broll_picker for the numbers.
        #
        # Failure here is not fatal: an empty result falls back to the word
        # scoring, which is what shipped before.
        picks = None
        if SEMANTIC_BROLL and broll_local:
            try:
                import broll_picker
                spans = _usable_spans(render_duration, words)
                lines = [
                    " ".join(w.text for w in (words or [])
                             if a <= float(getattr(w, "start", 0.0)) < b).strip()
                    for a, b in spans
                ]
                by_name = {os.path.basename(p): p for p in broll_local}
                chosen_names = broll_picker.choose(
                    lines, list(by_name), hook=moment.hook, theme=moment.theme,
                )
                if chosen_names:
                    picks = {i: by_name[n] for i, n in chosen_names.items()}
                    say(f"  ✓ {len(picks)} cutaway(s) chosen by meaning, "
                        f"{len(lines) - len(picks)} line(s) left on the speaker")
                else:
                    # An empty answer is a failure, not a set of declines, and
                    # it used to be indistinguishable from success: one short in
                    # a batch quietly reverted to word matching with nothing in
                    # the log to say so. The whole point of the picker is that
                    # its absence is visible.
                    say("  ⚠ The b-roll picker returned nothing — "
                        "falling back to word matching for this short")
            except Exception as e:
                say(f"  ⚠ B-roll picker failed ({str(e)[:60]}) — "
                    f"falling back to word matching")

        # Opens on the speaker, cuts away for a second or two, comes back.
        plan = _shot_plan(render_duration, raw_clip, broll_local, words, picks)
        cutaways = sum(1 for _p, _s, _d, kind in plan if kind == "b-roll")
        if picks is not None and cutaways < BROLL_FLOOR:
            plan = _shot_plan(render_duration, raw_clip, broll_local, words,
                              picks, _backfilling=True)
            filled = sum(1 for _p, _s, _d, kind in plan if kind == "b-roll")
            say(f"  · {cutaways} cutaway(s) after the declines — "
                f"backfilled to {filled}")
            cutaways = filled

        # Drop the clips the plan did not use.
        #
        # More are fetched than the trial count asks for, because that count is
        # a floor — see `_broll_slots`. The surplus is what keeps the cutaways
        # up now that a clip can never be used twice, and it has to be cleared
        # afterwards or this folder stops being the shot list made physical,
        # which is the one thing it is for.
        used_paths = {p for p, _s, _d, kind in plan if kind == "b-roll"}
        for path in list(broll_local):
            if path in used_paths:
                continue
            try:
                os.remove(path)
                broll_local.remove(path)
            except OSError:
                pass

        if broll_local:
            say(f"  ✓ {len(broll_local)} b-roll clip(s), {len(plan)} shots ({cutaways} cutaways)")

        try:
            with open(os.path.join(folder, "shotlist.md"), "w", encoding="utf-8") as f:
                f.write(_shotlist(plan, words, moment.hook))
        except Exception as e:
            say(f"  ⚠ Couldn't write the shot list: {e}")

        out_path = os.path.join(folder, "short.mp4")

        # A run that cannot time words produces a lesser short: no captions, and
        # a shot plan that falls back to a stopwatch instead of cutting on
        # sentences. That is worth having when there is nothing else, and it is
        # not worth writing over a good version of the same clip.
        #
        # This happened: both Whisper engines were blocked, eighteen shorts
        # rendered silently without captions, and each overwrote the captioned
        # version already sitting in its folder. The run reported success
        # because it had produced files.
        if not words and os.path.isfile(out_path):
            say("  ⚠ No word timings, and a previous version of this short "
                "exists — keeping it rather than overwriting with an "
                "uncaptioned one.")
            return None, log
        try:
            if PLAIN_ONLY:
                # The cut, and nothing else: the speaker's own picture at its
                # native shape, voice separated, silences tightened. No window,
                # no captions, no cutaways burned in.
                #
                # The b-roll is still chosen, still copied into broll/, and
                # still written into shotlist.md against the line it belongs
                # under. It is a recommendation now rather than a decision —
                # dragging a named clip onto a timeline takes seconds, and
                # undoing one that is already in the pixels does not.
                build_plain_cut(raw_clip, out_path, render_duration)
                say(f"  ✓ {out_path} (the cut — no frame, no captions, "
                    f"{len(broll_local)} b-roll suggested in shotlist.md)")
            elif broll_local:
                shots = [(path, src, dur) for path, src, dur, _kind in plan]
                bed = _pick_music(os.path.basename(folder))
                build_rough_cut(raw_clip, shots, out_path, render_duration,
                                words=words, music_path=bed)
                bed_note = (f", music: {os.path.basename(bed)}" if bed
                            else ", no music" if MUSIC_ON else "")
                say(f"  ✓ {out_path} (speaker, cutting to b-roll and back"
                    f"{bed_note})")
            else:
                # No footage available — fall back to the speaker's own picture
                # so the run still produces something usable.
                build_short(ShortSpec(
                    speech_source=raw_clip,
                    duration=render_duration,
                    out_path=out_path,
                    captions_path=captions_path,
                    style="speaker",
                ))
                say(f"  ✓ {out_path} (speaker footage — no b-roll available)")
        except Exception as e:
            say(f"  ⚠ Render failed: {e}")
            return None, log

        # The same moment again, stripped: same cut, same separated voice, no
        # b-roll, no window, no captions. It is the raw material of the short
        # rather than an alternative edit of it, which is why it takes the same
        # duration and the same audio chain and differs only in the picture.
        #
        # Rendered after the styled version and never allowed to fail the run:
        # if this one breaks, the finished short still exists, and that is the
        # right way round to lose something.
        plain_path = os.path.join(folder, "short_plain.mp4")
        try:
            build_plain_cut(raw_clip, plain_path, render_duration)
            say(f"  ✓ {plain_path} (plain — cut and voice only)")
        except Exception as e:
            say(f"  ⚠ Plain version failed, keeping the styled one: {e}")

        record = moment.to_dict()
        record.update({
            "index": i,
            "video_id": video_id,
            "source_url": url,
            "style": style,
            "folder": folder,
            "short": out_path,
            "assets": credits,
            "caption_words": len(words) if captions_path else 0,
        })
        with open(os.path.join(folder, "moment.json"), "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        return record, log

    # Clips are independent of each other, so they overlap. The work is a mix
    # of local ffmpeg cutting, Whisper on the CPU, one API call and a GPU
    # encode, which is exactly the shape that benefits: while one clip waits on
    # the network, another is encoding.
    #
    # Kept deliberately low. Each worker holds its own Whisper model in memory,
    # and pushing past the core count turns a speed-up into contention.
    lanes = max(1, min(workers, len(moments)))

    if lanes == 1:
        for i, moment in enumerate(moments, 1):
            print()
            record, log = build_one(i, moment)
            print("\n".join(log))
            if record is not None:
                results.append(record)
    else:
        from concurrent.futures import ThreadPoolExecutor

        print(f"\n→ Building {len(moments)} shorts, {lanes} at a time...")
        with ThreadPoolExecutor(max_workers=lanes) as pool:
            # Submitted all at once so they genuinely overlap; read back in
            # order so the output still reads as short 1, 2, 3.
            pending = [pool.submit(build_one, i, m) for i, m in enumerate(moments, 1)]
            for future in pending:
                record, log = future.result()
                print()
                print("\n".join(log))
                if record is not None:
                    results.append(record)

    # Re-cutting a talk renames the folders (they are built from the hook), so
    # the previous run's clips would otherwise sit beside the new ones — same
    # moment, different name, its own copy of the video.
    if results and not keep_old:
        keep = {os.path.abspath(r["folder"]) for r in results}
        stale = _clear_stale_clips(shorts_dir, keep)
        if stale:
            print()
            print(f"  · Removed {len(stale)} folder(s) from an earlier run of this talk:")
            for name in stale:
                print(f"      {name}")

    # Everything needed to actually post: brief, blog, thread, captions and
    # descriptions. Written once per talk rather than once per clip, because
    # they describe the talk, not any single cut of it.
    format_files: list[str] = []
    if formats:
        print()
        print("→ Writing the publishing set...")
        try:
            format_files = _publishing_set(data, os.path.join(output_dir, video_id))
            print(f"  ✓ {len(format_files)} file(s): {', '.join(format_files)}")
        except Exception as e:
            print(f"  ⚠ Publishing set failed: {e}")

    carousel_paths: list[str] = []
    if carousel:
        print("\n→ Writing carousel copy from the same moments...")
        try:
            carousel_paths = build_quote_carousel(
                moments,
                os.path.join(shorts_dir, "carousel"),
                talk_title=data.get("title", ""),
                channel=data.get("channel", ""),
                fetch=fetch,
            )
            print(f"  ✓ carousel copy written ({len(carousel_paths)} file(s))")
        except Exception as e:
            print(f"  ⚠ Carousel copy failed: {e}")

    with open(os.path.join(shorts_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({
            "video_id": video_id,
            "source_url": url,
            "shorts": results,
            "carousel": carousel_paths,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Done. {len(results)} short(s) in {shorts_dir}")
    for r in results:
        print(f"    {r['short']}")
    for p in carousel_paths:
        print(f"    {p}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cut a long talk into vertical motivational shorts."
    )
    parser.add_argument("url", help="YouTube URL or bare video ID")
    parser.add_argument("--count", type=int, default=None,
                        help="how many moments to cut. Omit to let the talk decide — "
                             "one per replay peak worth cutting, up to 8")
    parser.add_argument("--style", default="broll", choices=["broll", "speaker", "split"],
                        help="broll: stock footage; speaker: the talk's own video; split: both")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--whisper-model", default="base",
                        help="tiny/base/small — larger is more accurate, slower")
    parser.add_argument("--no-fetch", action="store_true",
                        help="use only what's already in assets/, never call a stock API")
    parser.add_argument("--no-carousel", action="store_true",
                        help="skip the carousel copy (text only, no images)")
    parser.add_argument("--no-formats", action="store_true",
                        help="skip the publishing set — brief, blog post, X thread, "
                             "captions and descriptions. Saves about four Claude "
                             "calls per talk when you only want the clips")
    parser.add_argument("--min-seconds", type=int, default=None,
                        help="shortest finished short (default 7)")
    parser.add_argument("--max-seconds", type=int, default=None,
                        help=f"longest finished short (default {MAX_TOTAL_SECONDS})")
    parser.add_argument("--keep-old", action="store_true",
                        help="don't remove clip folders left by an earlier run of "
                             "the same talk (they show up as duplicates)")
    parser.add_argument("--captions", action="store_true",
                        help="burn karaoke captions into the clip. Off by default, "
                             "because these clips are usually re-edited afterwards")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help=f"how many clips to build at once (default {DEFAULT_WORKERS}; "
                             "1 makes the log easier to read when debugging)")
    parser.add_argument("--moments-file", default=None,
                        help="render this saved set of moments instead of "
                             "asking the model for new ones — see Moment.to_dict")
    parser.add_argument("--source-file", default=None,
                        help="cut from this local video instead of downloading; "
                             "the URL is still used for the transcript and heatmap")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        from dotenv import load_dotenv
        load_dotenv()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("✗ ANTHROPIC_API_KEY not set. Add it to .env first.")
        sys.exit(1)

    if args.source_file and not os.path.isfile(args.source_file):
        print(f"✗ Source file not found: {args.source_file}")
        sys.exit(1)

    make_shorts(
        args.url,
        count=args.count,
        style=args.style,
        output_dir=args.output_dir,
        model_size=args.whisper_model,
        fetch=not args.no_fetch,
        carousel=not args.no_carousel,
        formats=not args.no_formats,
        min_seconds=args.min_seconds,
        max_seconds=args.max_seconds,
        source_file=args.source_file,
        workers=args.workers,
        captions=args.captions,
        keep_old=args.keep_old,
        moments_file=args.moments_file,
    )


if __name__ == "__main__":
    main()
