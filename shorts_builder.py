"""
shorts_builder.py

Assembles the finished vertical short: footage + speech + music bed + burned-in
captions, at 1080x1920.

Three layouts:

  broll    Full-frame stock footage, the speaker heard but not seen. The
           default, and what most motivational accounts run.
  speaker  The speaker's own footage, scaled to fill with a blurred copy of
           itself behind — same treatment the existing clip pipeline uses.
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


def _blurred_fill(label_in: str, label_out: str, duration: float) -> str:
    """Speaker footage centred over a blurred, zoomed copy of itself."""
    return (
        f"[{label_in}]split=2[sp_bg][sp_fg];"
        f"[sp_bg]scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=increase,"
        f"crop={VIDEO_W}:{VIDEO_H},gblur=sigma=22[sp_bgb];"
        f"[sp_fg]scale={VIDEO_W}:-2[sp_fgs];"
        f"[sp_bgb][sp_fgs]overlay=(W-w)/2:(H-h)/2,setsar=1,fps={FPS},"
        f"trim=duration={duration:.3f},setpts=PTS-STARTPTS[{label_out}]"
    )


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
            filters.append(_blurred_fill(f"{idx['speaker']}:v", "base", duration))
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
    parts = []
    for i in range(len(paths)):
        parts.append(
            f"[{i}:v]scale={VIDEO_W}:{VIDEO_H}:force_original_aspect_ratio=increase,"
            f"crop={VIDEO_W}:{VIDEO_H},setsar=1,fps={FPS},setpts=PTS-STARTPTS[v{i}];"
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
