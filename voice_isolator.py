"""
voice_isolator.py

Strips the backing music out of a talk, leaving the speaker's voice.

The talks this pipeline cuts frequently have a score playing under the
speaking. A denoiser cannot touch that: hiss and room tone are broadband and
statistically stationary, which is what an FFT denoiser models, while music is
structured sound sitting in the same frequencies as the voice. Removing it is
source separation — a model that knows what a voice sounds like — not
filtering.

Demucs does that. It splits a mix into vocals, drums, bass and other; this
keeps the vocals and discards the rest.

**Why it does its own audio I/O.** Demucs ships `demucs.api`, which is the
obvious way in, and importing it fails on this machine: it pulls
`demucs.audio`, which imports `lameenc` for MP3 output, and Windows Smart App
Control blocks that binary. It is the third native package it has blocked here,
after numba and PyAV. The model itself loads fine, so this reaches
`demucs.pretrained` and `demucs.apply` directly and moves audio in and out with
ffmpeg, which has never been blocked.

The model is downloaded once on first use and cached by torch, and it is held
in memory afterwards because separation runs once per clip.
"""

from __future__ import annotations

import os
import subprocess
import threading

SAMPLE_RATE = 44100          # what the pretrained models expect
MODEL_NAME = "htdemucs"

# How much consecutive analysis windows overlap.
#
# Demucs works in chunks and cross-fades them; more overlap means smoother
# joins and more compute. The default of 0.25 costs 36 seconds on a 36-second
# clip here, 0.10 costs 29, and 0.02 costs 25.
#
# 0.10 is the useful point on that curve — a fifth off the time, and the joins
# it affects are between analysis windows rather than anything audible in
# speech. Dropping to 0.02 saves another four seconds and starts risking
# artefacts at the seams for a return that is not worth it.
OVERLAP = 0.10

_LOCK = threading.Lock()
_MODEL = None


def _load_model():
    """The separation model, loaded once and reused."""
    global _MODEL
    if _MODEL is None:
        from demucs.pretrained import get_model

        _MODEL = get_model(MODEL_NAME)
        _MODEL.eval()
    return _MODEL


def _read_audio(path: str):
    """Decode to a (2, samples) float tensor via ffmpeg.

    ffmpeg rather than torchaudio or demucs.audio: both route through native
    extensions this machine has blocked, and ffmpeg is the one component here
    that never has been.
    """
    import torch

    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
         "-i", path, "-f", "f32le", "-acodec", "pcm_f32le",
         "-ac", "2", "-ar", str(SAMPLE_RATE), "-"],
        capture_output=True, check=True)
    audio = torch.frombuffer(bytearray(r.stdout), dtype=torch.float32)
    return audio.view(-1, 2).t().contiguous()


def _write_audio(tensor, path: str) -> None:
    import torch

    data = tensor.t().contiguous().to(torch.float32).numpy().tobytes()
    subprocess.run(
        ["ffmpeg", "-y", "-nostdin", "-hide_banner", "-loglevel", "error",
         "-f", "f32le", "-ar", str(SAMPLE_RATE), "-ac", "2", "-i", "-",
         "-c:a", "pcm_s16le", path],
        input=data, capture_output=True, check=True)


# How far below the voice the music has to sit before separating is not worth
# the minute it costs. Measured across this project's talks, a clip with no
# real backing music lands around -16 dB or lower and separating it changes
# nothing audible.
MUSIC_FLOOR_DB = -14.0

# Seconds of audio used to decide whether there is music at all. A tenth of the
# work of separating the whole clip, and enough to tell a scored talk from a
# dry one.
PROBE_SECONDS = 8.0


def music_level_db(media_path: str, seconds: float = PROBE_SECONDS) -> float | None:
    """How loud the non-vocal content is, relative to the voice, in dB.

    Runs the separator over a short sample rather than the whole clip. Returns
    a negative number: -30 means the music is thirty decibels under the voice
    and effectively absent, 0 would mean they are equally loud.

    This exists because separation costs about twice real time and most talks
    do not need it. Measuring first turns a minute of work into a few seconds
    for every clip that has nothing to remove.
    """
    try:
        import torch
        from demucs.apply import apply_model
    except Exception:
        return None

    try:
        with _LOCK:
            model = _load_model()
            wav = _read_audio(media_path)
            take = int(seconds * SAMPLE_RATE)
            if wav.shape[1] > take:
                start = max(0, (wav.shape[1] - take) // 2)
                wav = wav[:, start:start + take]

            ref = wav.mean(0)
            wav_n = (wav - ref.mean()) / (ref.std() + 1e-8)
            with torch.no_grad():
                sources = apply_model(model, wav_n[None], device="cpu",
                                      progress=False)[0]

            def rms(x):
                return float(x.pow(2).mean().sqrt()) + 1e-12

            vocals = rms(sources[model.sources.index("vocals")])
            rest = max(
                rms(sources[i])
                for i, name in enumerate(model.sources) if name != "vocals"
            )
            import math

            return 20 * math.log10(rest / vocals)
    except Exception:
        return None


def isolate_vocals(media_path: str, out_wav: str) -> str | None:
    """Write the vocal track of `media_path` to `out_wav`.

    Returns the path, or None if separation could not run — the caller should
    fall back to the original audio rather than lose the clip. Music under a
    voice is a flaw; no audio at all is a failure.
    """
    try:
        import torch
        from demucs.apply import apply_model
    except Exception:
        return None

    try:
        with _LOCK:
            model = _load_model()
            wav = _read_audio(media_path)

            # Demucs expects a batch and works in its own normalisation.
            ref = wav.mean(0)
            wav_n = (wav - ref.mean()) / (ref.std() + 1e-8)

            with torch.no_grad():
                sources = apply_model(
                    model, wav_n[None], device="cpu", progress=False,
                    overlap=OVERLAP,
                )[0]
            sources = sources * (ref.std() + 1e-8) + ref.mean()

            vocals = sources[model.sources.index("vocals")]
            _write_audio(vocals, out_wav)
    except Exception:
        return None

    return out_wav if os.path.isfile(out_wav) else None


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Keep only the voice.")
    parser.add_argument("media")
    parser.add_argument("out")
    args = parser.parse_args()

    start = time.time()
    path = isolate_vocals(args.media, args.out)
    if path:
        print(f"vocals -> {path}  ({time.time()-start:.0f}s)")
    else:
        print("separation unavailable; the original audio should be used")
