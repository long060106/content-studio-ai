# AI Content Studio — Extraction, Brief, Blog Post, Twitter Thread, LinkedIn Post, Captions

Takes a single YouTube URL and produces:

1. **`transcript.json`** — the raw transcript + video metadata (title, channel,
   duration, description).
2. **`brief.json`** — a structured "content brief" (summary, key points with
   supporting quotes, hooks, tone, audience, notable quotes, topics). This is
   the input every generator reads from.
3. **`blog_post.md`** — a ~500-800 word blog post in Markdown, ready to publish.
4. **`twitter_thread.json`** / **`twitter_thread.txt`** — an 8-12 tweet thread,
   each tweet under 280 characters. The `.txt` version is numbered and
   formatted for easy copy-pasting tweet by tweet.
5. **`linkedin_post.md`** — a 150-250 word LinkedIn post with a hook-first
   structure suited to LinkedIn's "see more" truncation.
6. **`captions.json`** / **`captions.txt`** — 3 short-form caption variants
   (hook-led, story-led, question-led) for Instagram/TikTok/Shorts, each with
   hashtags.
7. **`voiceover_script.txt`** / **`voiceover.mp3`** — a short (100-150 word)
   narration script and its spoken-audio version, generated via ElevenLabs.
   **Optional**: only runs if `ELEVENLABS_API_KEY` is set in `.env`. Everything
   else in the pipeline runs regardless.
8. **`clip_info.json`** / **`short_form_clip.mp4`** — Claude picks the best
   20-60 second self-contained window from the timestamped transcript, then
   that segment is downloaded and converted to vertical 9:16 video (with a
   blurred-background fill) for Shorts/Reels/TikTok.

## Why a two-step pipeline?

We don't generate the blog post / tweet thread / etc. directly from the raw
transcript. Instead we go: **transcript → brief → each format**. This keeps
every output consistent with each other, produces higher-quality results
(the model works from a distilled outline instead of a noisy transcript),
and is cheaper (the brief is generated once and reused).

## Setup

```bash
cd ai-content-studio
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and add your ANTHROPIC_API_KEY
# get one at https://console.anthropic.com/
```

### Optional: voice-over (ElevenLabs)

The voice-over step is optional — skip this and everything else still works.
To enable it:

1. Create a free account at https://elevenlabs.io
2. Get an API key from https://elevenlabs.io/app/settings/api-keys
3. Add it to `.env`:
   ```
   ELEVENLABS_API_KEY=your_key_here
   ```

The default script length (100-150 words) is intentionally short to keep
free-tier character usage low while testing.

## Usage

### The web UI (easiest)

```bash
python webapp.py
```

Opens http://127.0.0.1:8420 in your browser. Paste a YouTube URL, hit
**Generate**, and watch the pipeline work through its checklist live. When it
finishes, every format is browsable in one place — blog post rendered from
Markdown, thread as numbered tweet cards with character counts, carousel
slides as images, the short-form clip in a player, the voice-over as audio —
each with a copy or download button.

Everything already in `./output/` shows up in the left-hand library, so past
runs stay one click away.

Notes:

- It's the same pipeline: the server shells out to `cli.py` and streams its
  output. Anything the CLI can do, the UI does.
- Standard library only — no extra `pip install`, and it runs with or without
  the venv activated (it uses `venv/`'s interpreter for the pipeline itself
  when that venv exists).
- Binds to `127.0.0.1` only, and reports *whether* your API keys are set
  without ever sending the values to the browser.
- One run at a time; a second request is rejected while one is in flight.
- Change the port with `CONTENT_STUDIO_UI_PORT=9000 python webapp.py`, or set
  `CONTENT_STUDIO_UI_NO_BROWSER=1` to stop it from opening a browser tab.

### The CLI

```bash
python cli.py "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
```

This will print progress and save results to `./output/<video_id>/`.

You can also run each stage independently:

```bash
# Just get the transcript + metadata, print as JSON
python youtube_extractor.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Get transcript AND generate the brief, print brief as JSON
python content_brief.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Generate a blog post from an already-saved brief.json
python blog_generator.py output/VIDEO_ID/brief.json

# Generate a Twitter/X thread from an already-saved brief.json
python twitter_thread_generator.py output/VIDEO_ID/brief.json

# Generate a LinkedIn post from an already-saved brief.json
python linkedin_generator.py output/VIDEO_ID/brief.json

# Generate social captions from an already-saved brief.json
python captions_generator.py output/VIDEO_ID/brief.json
```

## Whisper fallback (videos without captions)

If a video has no YouTube captions available, `youtube_extractor.py`
automatically falls back to `audio_transcriber.py`: it downloads the video's
audio via `yt-dlp` and transcribes it locally using OpenAI's Whisper model.
This is slower (audio download + local ML inference) but works on any video
regardless of caption availability.

**Extra setup required for this path:**

1. **ffmpeg** must be installed and on PATH (used to extract/decode audio).
   On Windows: `winget install ffmpeg`, then restart your terminal.
   Verify with `ffmpeg -version`.
2. **`openai-whisper`** (in `requirements.txt`) pulls in PyTorch — a much
   larger install than the other dependencies. First transcription run also
   downloads the model weights (~150MB for the default "base" model),
   cached afterward.

You don't need to do anything extra to trigger this — it happens
automatically whenever `get_transcript()` raises `NoTranscriptAvailable`.

## Known limitations (v1)

- **Very long videos get truncated** to ~12,000 words of transcript before
  being sent to the brief generator, to control cost/latency. Fine for most
  content (talks, tutorials, podcasts under ~90 min); long-form (3hr+) videos
  will lose tail content in the brief. Chunked summarization is the fix if you
  need that later.
- **Non-English content**: transcript extraction handles other languages fine,
  but the brief prompt is currently written to respond in English regardless
  of source language. Easy to adjust if you want brief output to match the
  source language.

## File structure

```
ai-content-studio/
├── youtube_extractor.py         # URL → transcript + metadata (with Whisper fallback)
├── audio_transcriber.py         # Whisper-based fallback for videos without captions
├── content_brief.py             # transcript → structured brief (Claude API)
├── blog_generator.py            # brief → blog post (Claude API)
├── twitter_thread_generator.py  # brief → Twitter/X thread (Claude API)
├── linkedin_generator.py        # brief → LinkedIn post (Claude API)
├── captions_generator.py        # brief → social captions (Claude API)
├── voiceover_script_generator.py # brief → narration script (Claude API)
├── text_to_speech.py            # script → MP3 audio (ElevenLabs API)
├── carousel_generator.py        # brief → carousel slide text (Claude API)
├── carousel_renderer.py         # slide text → PNG images (Pillow)
├── clip_selector.py             # timestamped transcript → best clip window (Claude API)
├── video_clipper.py             # time range → vertical MP4 (yt-dlp + ffmpeg)
├── cli.py                       # orchestrates all of the above, saves output
├── webapp.py                    # local web UI: runs cli.py, browses ./output (stdlib only)
├── ui/                          # the UI's static files
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── requirements.txt
├── .env.example
└── output/                      # created at runtime, one folder per video_id
    └── <video_id>/
        ├── transcript.json
        ├── brief.json
        ├── blog_post.md
        ├── twitter_thread.json
        ├── twitter_thread.txt
        ├── linkedin_post.md
        ├── captions.json
        ├── captions.txt
        ├── voiceover_script.txt   (if ElevenLabs configured)
        ├── voiceover.mp3          (if ElevenLabs configured)
        ├── carousel/
        │   ├── carousel.json
        │   └── slide_01.png, slide_02.png, ...
        ├── clip_info.json
        └── short_form_clip.mp4
```

## Status

All 8 original target formats are implemented: blog post, Twitter/X thread,
LinkedIn post, Instagram carousel, TikTok/social captions, voice-over,
short-form video, and captions. Natural next steps beyond this: TikTok-specific
script formatting (currently covered by the general captions generator),
and publishing integrations to push finished content to each platform
automatically.
