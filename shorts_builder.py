"""
shorts_builder.py

Assembles the finished vertical short: footage + speech + music bed + burned-in
captions, at 1080x1920.

Three layouts:

  broll    Full-frame stock footage, the speaker heard but not seen. The
           default, and what most motivational accounts run.
  speaker  The speaker's own footage, uncropped and centred on black. The
           bands above and below are where the caption sits.
  split    Speaker on top, b-roll underneath.

The audio is the part that's easy to get wrong. Music sits under speech via
`sidechaincompress`, so the bed ducks automatically whenever the speaker
talks and swells back in the gaps, then everything is normalised to roughly
-14 LUFS, which is what the social platforms target. Without ducking the
music fights the voice and the short sounds amateur.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

VIDEO_W = 1080
VIDEO_H = 1920
FPS = 30

RENDER_TIMEOUT = 1800  # a 35s short encodes in well under a minute

# The canvas stitched cuts are joined on, before any layout decision is made.
# Landscape on purpose: these are pieces of a filmed talk, and joining is not
# the place to decide how they sit in a vertical frame.
JOIN_W = 1920
JOIN_H = 1080

MUSIC_GAIN = 0.22       # bed level before ducking
FADE_OUT = 1.2


class RenderError(RuntimeError):
    pass


# Chosen once per process, then reused. The probe costs about a second; doing
# it per encode would give back much of what hardware encoding saves.
_ENCODER: Optional[list[str]] = None

_SOFTWARE_ENCODER = [
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
]
_QSV_ENCODER = [
    "-c:v", "h264_qsv", "-global_quality", "24", "-pix_fmt", "nv12",
]


def video_encoder_args() -> list[str]:
    """ffmpeg arguments for encoding video, hardware if the machine has it.

    Intel Quick Sync (h264_qsv) hands the work to the GPU's dedicated encoder.
    Measured on this project: a short that took 9.1s with libx264 at -preset
    medium takes 4.1s with QSV, and the file is smaller (6.0 MB vs 9.0 MB).

    Falling back matters as much as the speedup: QSV is listed by ffmpeg on
    machines that cannot actually run it — the encoder is compiled in whether
    or not the hardware and drivers are there — so availability is settled by
    encoding one real frame rather than by reading the list.
    """
    global _ENCODER
    if _ENCODER is not None:
        return _ENCODER

    probe = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "color=c=black:s=256x256:d=0.1",
        *_QSV_ENCODER, "-f", "null", "-",
    ]
    try:
        result = subprocess.run(probe, capture_output=True, text=True, timeout=60)
        _ENCODER = _QSV_ENCODER if result.returncode == 0 else _SOFTWARE_ENCODER
    except (OSError, subprocess.TimeoutExpired):
        _ENCODER = _SOFTWARE_ENCODER
    return _ENCODER


@dataclass
class ShortSpec:
    """Everything needed to render one short."""
    speech_source: str              # clip holding the speaker's audio
    duration: float
    out_path: str
    captions_path: Optional[str] = None
    broll_path: Optional[str] = None
    music_path: Optional[str] = None
    style: str = "broll"
    speaker_video: Optional[str] = None   # defaults to speech_source


def _run(cmd: list[str], cwd: Optional[str] = None) -> None:
    try:
        result = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=RENDER_TIMEOUT
        )
    except subprocess.TimeoutExpired:
        raise RenderError(f"ffmpeg didn't finish within {RENDER_TIMEOUT}s.") from None
    if result.returncode != 0:
        raise RenderError(f"ffmpeg failed:\n{result.stderr[-2500:]}")


def _fill(label_in: str, label_out: str, duration: float) -> str:
    """Scale/crop any source to fill the vertical frame."""
    return (
        f"[{label_in}]scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=increase,"
        f"crop={VIDEO_W}:{VIDEO_H},setsar=1,fps={FPS},"
        f"trim=duration={duration:.3f},setpts=PTS-STARTPTS[{label_out}]"
    )


# The window. Every shot in every short from every talk sits in a band of
# exactly this height, centred on black.
#
# It is a constant, and an earlier version measured it from each talk instead.
# That was a mistake worth spelling out, because measuring looks more careful.
# It kept the speaker and the b-roll matched *within* one short — which is what
# it was written for, and it did that correctly — but it made the window a
# different size for every talk, and for a source that was already vertical it
# produced a band the full height of the frame: no black at all, the whole
# format gone. One talk in a batch came out looking like a different account.
#
# The window belongs to the account, not to the footage. Whatever comes in gets
# fitted to it.
BAND_H = (VIDEO_W * 9 // 16) // 2 * 2  # 608

# Where to take the crop from when a source is taller than the band.
#
# Dead centre is wrong for people. A talking head shot vertically has the face
# in the upper half and empty room below, so a centred crop takes the chin and
# the chest and loses the eyes. Biasing upward keeps the face.
CROP_BIAS = 0.35


def _to_band(label_in: str, label_out: str, duration: float, extra: str = "") -> str:
    """Fit any source into the band: full width, cropped to height if taller.

    A 16:9 source lands exactly on the band with nothing cropped, which is the
    common case and unchanged from before. Anything taller — 4:3, 4:5, a
    vertical phone recording — is cropped rather than given a taller band, so
    the window stays the same size no matter what came in.

    `extra` carries filters that belong to one kind of shot only, such as the
    b-roll grade and slow-down.
    """
    return (
        f"[{label_in}]scale={VIDEO_W}:{BAND_H}:force_original_aspect_ratio=increase,"
        f"crop={VIDEO_W}:{BAND_H}:0:(ih-{BAND_H})*{CROP_BIAS},"
        f"setsar=1,fps={FPS},{extra}"
        f"pad={VIDEO_W}:{VIDEO_H}:0:{(VIDEO_H - BAND_H) // 2}:black,"
        f"trim=duration={duration:.3f},setpts=PTS-STARTPTS[{label_out}]"
    )


def _on_black(label_in: str, label_out: str, duration: float) -> str:
    """Speaker footage uncropped and centred on solid black.

    The counterpart to `_fill` above, and deliberately not the same treatment.
    B-roll fills the frame because it is backdrop; the speaker must not, because
    cropping 16:9 to 9:16 discards two thirds of the width and upscales the
    rest.

    This used to lay the frame over a blurred, zoomed copy of itself instead of
    black. That was rejected on sight: on dark footage the blurred copy puts a
    large out-of-focus head above the speaker. Black is what the accounts in
    this format actually use, and the bands are where the caption goes.

    It delegates to `_to_band` so this path — the fallback taken when no
    footage is available — produces the same window as the main one. It
    previously scaled to whatever height the source's aspect ratio gave, which
    is how a vertical source came out filling the frame with no bands at all.
    """
    return _to_band(label_in, label_out, duration)


def build_short(spec: ShortSpec) -> str:
    """Render one short. Returns the output path."""
    if not os.path.isfile(spec.speech_source):
        raise RenderError(f"Speech source not found: {spec.speech_source}")

    duration = max(1.0, float(spec.duration))
    speaker_video = spec.speaker_video or spec.speech_source

    has_broll = bool(spec.broll_path and os.path.isfile(spec.broll_path))
    has_music = bool(spec.music_path and os.path.isfile(spec.music_path))
    style = spec.style if (has_broll or spec.style == "speaker") else "speaker"

    # libass chokes on Windows drive letters inside a filtergraph, so the
    # subtitle file is copied next to the render and referenced by bare name
    # with ffmpeg's cwd set to that folder.
    work_dir = tempfile.mkdtemp(prefix="short_render_")
    try:
        subs_name = None
        if spec.captions_path and os.path.isfile(spec.captions_path):
            subs_name = "captions.ass"
            shutil.copyfile(spec.captions_path, os.path.join(work_dir, subs_name))

        inputs: list[str] = []
        filters: list[str] = []
        idx = {}

        if style in ("broll", "split") and has_broll:
            inputs += ["-stream_loop", "-1", "-i", os.path.abspath(spec.broll_path)]
            idx["broll"] = len(idx)
        if style in ("speaker", "split"):
            inputs += ["-i", os.path.abspath(speaker_video)]
            idx["speaker"] = len(idx)

        # The speech track always comes from the extracted clip.
        inputs += ["-i", os.path.abspath(spec.speech_source)]
        idx["speech"] = len(idx)

        if has_music:
            inputs += ["-stream_loop", "-1", "-i", os.path.abspath(spec.music_path)]
            idx["music"] = len(idx)

        # ---- video ----
        if style == "broll":
            filters.append(_fill(f"{idx['broll']}:v", "base", duration))
            # Slight darkening keeps white captions readable over bright footage.
            filters.append("[base]eq=brightness=-0.05:saturation=1.05[vbase]")
        elif style == "speaker":
            filters.append(_on_black(f"{idx['speaker']}:v", "base", duration))
            filters.append("[base]null[vbase]")
        else:  # split
            half = VIDEO_H // 2
            filters.append(
                f"[{idx['speaker']}:v]scale={VIDEO_W}:{half}:force_original_aspect_ratio=increase,"
                f"crop={VIDEO_W}:{half},setsar=1,fps={FPS},"
                f"trim=duration={duration:.3f},setpts=PTS-STARTPTS[top]"
            )
            filters.append(
                f"[{idx['broll']}:v]scale={VIDEO_W}:{half}:force_original_aspect_ratio=increase,"
                f"crop={VIDEO_W}:{half},setsar=1,fps={FPS},"
                f"trim=duration={duration:.3f},setpts=PTS-STARTPTS[bottom]"
            )
            filters.append(f"[top][bottom]vstack=inputs=2,setsar=1[vbase]")

        if subs_name:
            filters.append(f"[vbase]subtitles={subs_name}[v]")
        else:
            filters.append("[vbase]null[v]")

        # ---- audio ----
        filters.append(
            f"[{idx['speech']}:a]atrim=duration={duration:.3f},asetpts=PTS-STARTPTS,"
            f"highpass=f=85,dynaudnorm=f=200:g=5[speech]"
        )
        if has_music:
            filters.append(
                f"[{idx['music']}:a]atrim=duration={duration:.3f},asetpts=PTS-STARTPTS,"
                f"volume={MUSIC_GAIN},"
                f"afade=t=out:st={max(0.0, duration - FADE_OUT):.3f}:d={FADE_OUT}[musicraw]"
            )
            filters.append("[speech]asplit=2[speech_out][speech_key]")
            filters.append(
                "[musicraw][speech_key]sidechaincompress="
                "threshold=0.05:ratio=12:attack=8:release=350:makeup=1[ducked]"
            )
            filters.append(
                "[speech_out][ducked]amix=inputs=2:duration=first:dropout_transition=0,"
                "loudnorm=I=-14:TP=-1.5:LRA=11[a]"
            )
        else:
            filters.append("[speech]loudnorm=I=-14:TP=-1.5:LRA=11[a]")

        os.makedirs(os.path.dirname(os.path.abspath(spec.out_path)) or ".", exist_ok=True)

        cmd = (
            ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error"]
            + inputs
            + [
                "-filter_complex", ";".join(filters),
                "-map", "[v]", "-map", "[a]",
                "-t", f"{duration:.3f}",
                *video_encoder_args(),
                "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
                "-movflags", "+faststart",
                os.path.abspath(spec.out_path),
            ]
        )
        _run(cmd, cwd=work_dir)
        return spec.out_path
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


def join_clips(paths: list[str], out_path: str) -> str:
    """Stitch several downloaded cuts into one continuous clip.

    Re-encodes through the concat filter rather than using the concat demuxer.
    The demuxer is faster but requires every input to share codec, resolution
    and timebase exactly — yt-dlp's ranged downloads don't reliably do that, and
    when they don't the result is silent audio desync rather than an error.

    The join is a hard cut, which is the convention in this format: the jump
    between passages is the edit, and fading it would only make it feel like a
    mistake being smoothed over.
    """
    if not paths:
        raise RenderError("Nothing to join.")
    if len(paths) == 1:
        if os.path.abspath(paths[0]) != os.path.abspath(out_path):
            shutil.copyfile(paths[0], out_path)
        return out_path

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)

    inputs: list[str] = []
    for path in paths:
        inputs += ["-i", os.path.abspath(path)]

    # Normalise each piece before concatenating so mismatched sources can't
    # desync: common frame rate, sample rate, channel layout and PTS reset.
    #
    # **Fit, never crop.** This used to scale each piece up and crop it to
    # 1080x1920, which quietly turned a stitched clip into vertical footage
    # before the layout stage ever saw it — the exact crop-and-zoom that was
    # rejected on quality grounds, surviving in the one path nobody looked at.
    # The result was a batch where seven shorts were letterboxed and the
    # stitched one filled the frame, zoomed and soft, looking like a different
    # account.
    #
    # This is a joining step. Its only job is to make the pieces compatible
    # enough to concatenate; deciding how footage sits in the frame belongs to
    # `_to_band`, further down the pipeline, and doing it here as well meant
    # doing it twice and disagreeing.
    parts = []
    for i in range(len(paths)):
        parts.append(
            f"[{i}:v]scale={JOIN_W}:{JOIN_H}:force_original_aspect_ratio=decrease,"
            f"pad={JOIN_W}:{JOIN_H}:(ow-iw)/2:(oh-ih)/2:black,"
            f"setsar=1,fps={FPS},setpts=PTS-STARTPTS[v{i}];"
            f"[{i}:a]aresample=48000,asetpts=PTS-STARTPTS[a{i}]"
        )
    streams = "".join(f"[v{i}][a{i}]" for i in range(len(paths)))
    filtergraph = ";".join(parts) + f";{streams}concat=n={len(paths)}:v=1:a=1[v][a]"

    cmd = (
        ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error"]
        + inputs
        + [
            "-filter_complex", filtergraph,
            "-map", "[v]", "-map", "[a]",
            *video_encoder_args(),
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            os.path.abspath(out_path),
        ]
    )
    _run(cmd)
    return out_path


def build_rough_cut(
    speech_source: str,
    shots: list[tuple[str, float, float]],
    out_path: str,
    duration: float,
) -> str:
    """Cut between the speaker and b-roll over one continuous speech track.

    `shots` is (path, source_start, seconds) in playing order. The audio runs
    underneath unbroken; only the picture changes.

    The `source_start` is what makes cutting back to the speaker work. A shot
    of the speaker at 8s into the clip has to show the speaker *at 8s* — their
    mouth has to match the words being heard. Starting it from zero would put
    the picture out of sync with its own audio, which is instantly obvious and
    looks broken. B-roll has no such constraint and starts wherever.

    Alternating back to the speaker is the format, not a fallback: holding
    b-roll for the whole clip loses the person saying it, and the cut back to a
    face is what makes the words feel said rather than narrated.

    B-roll inputs are opened with `-stream_loop -1` so a four-second stock clip
    can still fill its slot. Captions are deliberately not burned in — the SRT
    beside the file covers that, and text baked into the picture fights the
    edit this file exists to start.
    """
    if not shots:
        raise RenderError("No shots to assemble.")
    if not os.path.isfile(speech_source):
        raise RenderError(f"Speech source not found: {speech_source}")

    duration = max(1.0, float(duration))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    speech_abs = os.path.abspath(speech_source)

    inputs: list[str] = []
    for path, source_start, _seconds in shots:
        if os.path.abspath(path) == speech_abs:
            # Seek to the matching moment so the picture stays in sync with
            # the audio it belongs to.
            inputs += ["-ss", f"{max(0.0, float(source_start)):.3f}", "-i", speech_abs]
        else:
            inputs += ["-stream_loop", "-1", "-i", os.path.abspath(path)]
    speech_index = len(shots)
    inputs += ["-i", speech_abs]

    parts: list[str] = []
    for i, (path, _source_start, seconds) in enumerate(shots):
        span = max(0.2, float(seconds))
        is_speaker = os.path.abspath(path) == speech_abs

        if is_speaker:
            # The whole 16:9 frame, uncropped, centred on solid black.
            #
            # Nothing is cropped and nothing is enlarged. Cropping 16:9 footage
            # to fill a 9:16 frame throws away two thirds of the width and
            # upscales what is left, which is what made the speaker look soft
            # and over-zoomed.
            #
            # The bands above and below are deliberately black rather than
            # filled. An earlier version put a blurred, zoomed copy of the same
            # frame back there; on dark footage that puts a large out-of-focus
            # head above the speaker and reads as a smudge rather than as
            # design. The empty space is not a gap to be patched — it is where
            # the caption sits, which is the whole reason this layout works on
            # the accounts using it.
            parts.append(_to_band(f"{i}:v", f"v{i}", span))
        else:
            # B-roll sits in the same band as the speaker, at the same size and
            # the same position, so the window never changes shape across a cut.
            #
            # This costs real picture. The library is vertical — 87 portrait
            # clips, no landscape — so cropping into a wide band throws away
            # most of each frame's height, and the argument against doing it is
            # genuinely strong: nothing here needs rescuing from the wrong
            # aspect ratio the way the speaker does.
            #
            # It is done anyway, because the letterbox is not a repair. It is
            # the format's identity: a fixed window that footage changes inside
            # while the frame stays put. A cutaway that expands to fill the
            # screen breaks that every couple of seconds, and side by side the
            # consistent version wins clearly enough that the lost height is
            # worth paying. If this is ever reverted, revert the library toward
            # footage that reads in a wide strip at the same time.
            #
            # Slowed down because these cuts sit under a voice — footage moving
            # at normal speed pulls attention off the words, and slow motion is
            # the register the format uses.
            #
            # Graded per shot, not over the finished video. See BROLL_GRADE.
            parts.append(_to_band(
                f"{i}:v", f"v{i}", span,
                extra=f"setpts={1.0 / BROLL_SPEED:.3f}*PTS,{BROLL_GRADE},",
            ))
    streams = "".join(f"[v{i}]" for i in range(len(shots)))
    parts.append(f"{streams}concat=n={len(shots)}:v=1:a=0[v]")
    parts.append(
        f"[{speech_index}:a]atrim=duration={duration:.3f},asetpts=PTS-STARTPTS,"
        f"highpass=f=85,loudnorm=I=-14:TP=-1.5:LRA=11[a]"
    )

    cmd = (
        ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error"]
        + inputs
        + [
            "-filter_complex", ";".join(parts),
            "-map", "[v]", "-map", "[a]",
            "-t", f"{duration:.3f}",
            *video_encoder_args(),
            "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
            "-movflags", "+faststart",
            os.path.abspath(out_path),
        ]
    )
    _run(cmd)
    return out_path


# B-roll playback rate. Below 1.0 is slow motion.
#
# These cuts sit under a voice, and footage moving at normal speed competes
# with the words for attention. Slowing it makes the picture read as mood
# rather than as action, which is the register this format uses throughout.
BROLL_SPEED = 0.7

# Grading applied to b-roll, and only to b-roll.
#
# Stock footage and a filmed talk are lit nothing alike. The talks this runs on
# are dark studio interiors; stock clips of rain, forests and streets are shot
# in bright overcast daylight. Cutting straight between them makes the screen
# jump from near-black to mid-grey every couple of seconds, and the cutaway
# stops reading as part of the same film — it reads as a stock clip dropped in,
# which is exactly what it is and exactly what should not be visible.
#
# Darkening and pulling the colour back closes most of that gap. It cannot be
# closed completely without crushing the footage into mud, and it should not
# be: some lift on the cutaway is what makes it a cutaway.
#
# **Per shot, not over the finished video.** An earlier version graded the
# concatenated result, which meant the speaker got darkened too — and the
# speaker is the one thing already near black — while saturation was pushed
# *up*, the opposite of what the mismatch needs. A grade meant for one kind of
# footage has to be applied to that footage, not to everything.
BROLL_GRADE = "eq=brightness=-0.10:saturation=0.80:contrast=1.05"


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)")
_VIDEO_SIZE_RE = re.compile(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})")


def ffmpeg_probe(path: str) -> tuple[float, int, int]:
    """(duration, width, height) read out of ffmpeg's own banner.

    The fallback for when `ffprobe` can't be run. They ship together, but they
    are separate executables and can be permitted separately: on this machine
    Windows Application Control blocks `ffprobe.exe` outright (WinError 4551)
    while `ffmpeg.exe` runs fine. Since ffmpeg prints the same Duration and
    Stream lines to stderr when given no output file, everything needed is
    already there for the reading.

    Returns zeroes if ffmpeg can't be run either.
    """
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-i", path],
            capture_output=True, text=True, timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0.0, 0, 0

    # ffmpeg exits non-zero here because no output was requested; the metadata
    # it printed on the way is still valid.
    text = out.stderr or ""
    duration = 0.0
    match = _DURATION_RE.search(text)
    if match:
        hours, minutes, seconds = match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)

    width = height = 0
    size = _VIDEO_SIZE_RE.search(text)
    if size:
        width, height = int(size.group(1)), int(size.group(2))
    return duration, width, height


def probe_duration(path: str) -> float:
    """Actual playing time of a file, or 0.0 if it can't be read.

    Worth measuring rather than trusting the planned length: yt-dlp's ranged
    downloads are keyframe-aligned, so a cut asked for as 18.0s routinely comes
    back as 17.6s. Rendering the planned duration against a shorter source
    leaves a frozen frame on the end of the short.
    """
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration", "-of", "json", path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        import json as _json

        found = float(_json.loads(out.stdout or "{}").get("format", {}).get("duration", 0) or 0)
        if found > 0:
            return found
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass

    return ffmpeg_probe(path)[0]


def make_music_bed(out_path: str, duration: float) -> str:
    """Generate a plain synth pad.

    A placeholder so the audio chain is testable (and a render never fails)
    before any real music is in the library. Swap in a real track from
    assets/music/ for anything you actually publish.
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"sine=frequency=110:duration={duration:.3f}",
        "-f", "lavfi", "-i", f"sine=frequency=164.81:duration={duration:.3f}",
        "-filter_complex",
        "[0:a][1:a]amix=inputs=2,tremolo=f=0.4:d=0.3,aformat=sample_rates=48000[a]",
        "-map", "[a]", "-c:a", "libmp3lame", "-q:a", "5", os.path.abspath(out_path),
    ]
    _run(cmd)
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render one vertical short.")
    parser.add_argument("speech_source", help="clip holding the speaker audio")
    parser.add_argument("out_path")
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--captions")
    parser.add_argument("--broll")
    parser.add_argument("--music")
    parser.add_argument("--style", default="broll", choices=["broll", "speaker", "split"])
    args = parser.parse_args()

    path = build_short(ShortSpec(
        speech_source=args.speech_source,
        duration=args.duration,
        out_path=args.out_path,
        captions_path=args.captions,
        broll_path=args.broll,
        music_path=args.music,
        style=args.style,
    ))
    print(f"Saved {path}")
