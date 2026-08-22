# AI Content Studio — Extraction, Brief, Blog Post, Twitter Thread, Captions

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
5. **`captions.json`** / **`captions.txt`** — 3 short-form caption variants
   (hook-led, story-led, question-led) for Instagram/TikTok/Shorts, each with
   hashtags.
6. **`voiceover_script.txt`** / **`voiceover.mp3`** — a short (100-150 word)
   narration script and its spoken-audio version, via ElevenLabs. **Off by
   default** — pass `--voiceover` and set `ELEVENLABS_API_KEY`. AI narration
   over a real speaker is rarely what you want, so it's there if you need it
   and out of the way if you don't.
7. **`clip_info.json`** / **`short_form_clip.mp4`** — Claude picks the best
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

Off by default and rarely needed — the pipeline cuts real speakers, and a
synthetic voice on top of that usually makes things worse. If you do want it,
pass `--voiceover` and:

1. Create a free account at https://elevenlabs.io
2. Get an API key from https://elevenlabs.io/app/settings/api-keys
3. Add it to `.env`:
   ```
   ELEVENLABS_API_KEY=your_key_here
   ```

Both are required: without `--voiceover` the step is skipped even with a key
set. The default script length (100-150 words) keeps free-tier character usage
low while testing.

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
- **Stopping a run**: while anything is running, a red **Stop** appears in the
  header on every view — you don't have to navigate back to the run to cancel
  it, and clicking the label next to it jumps to that run. Stopping kills the
  whole process tree (yt-dlp, ffmpeg, Whisper), not just the Python parent, so
  nothing keeps encoding in the background. Anything already finished stays on
  disk: `cli.py` and `make_shorts.py` both save each artifact as it's produced.
- Stopping the server (Ctrl+C) also stops a run in progress rather than
  leaving it orphaned.
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

# Generate social captions from an already-saved brief.json
python captions_generator.py output/VIDEO_ID/brief.json
```

## Motivational shorts (`make_shorts.py`)

A second pipeline aimed at one thing: turning a long talk into a batch of
vertical motivational shorts — speaker audio, matched b-roll, word-by-word
captions, music bed.

```bash
python make_shorts.py "https://www.youtube.com/watch?v=VIDEO_ID" --count 3 --carousel
```

Or in the web UI: switch the run bar to **Motivational shorts**, which exposes
the moment count, layout, and whether to render the quote carousel. Finished
shorts appear under a **Shorts** tab on that video, each with a player, its
hook, the quote it was cut around, and where in the talk it came from.

Each short lands in `output/<video_id>/shorts/<nn>_<theme>_<hook>/` with the
finished `short.mp4`, the raw downloaded clip, the caption file, and a
`moment.json` recording the hook, quote, theme and asset credits.

### How it differs from the main pipeline

The main `cli.py` describes one video eight ways. This one hunts for the
*several* moments in a talk that stand alone, and cuts each into its own video.

1. **Moments, not a summary.** `moment_finder.py` reads the timestamped
   transcript and picks the strongest 15-35 second beats — one complete
   thought with a turn in it, starting and ending on sentence boundaries.
   Each moment comes back with a hook line, the strongest quote, a theme
   (discipline, ownership, fear…) and concrete visual search terms.
2. **The full talk is never re-transcribed.** Moment-finding runs off
   YouTube's existing captions, and only the chosen 15-35s windows are
   downloaded. One Claude call per talk.
3. **Captions are timed per clip.** `caption_timing.py` runs Whisper locally
   on each short clip alone (a few seconds on CPU, no API cost) to get
   word-level timings, then writes an ASS subtitle file where the spoken word
   pops in accent colour. YouTube's captions are far too coarse for this.
4. **Music ducks under the voice.** `shorts_builder.py` mixes the bed through
   `sidechaincompress` keyed on the speech, then normalises to ~-14 LUFS.

### Cutting where the audience actually rewatched

Picking moments by reading a transcript produces clips that read well and land
flat. YouTube publishes a **most replayed** curve for videos above a view
threshold, and yt-dlp already exposes it — so `heatmap.py` uses it to anchor the
cuts in real viewer behaviour.

```bash
python heatmap.py "https://www.youtube.com/watch?v=VIDEO_ID" --transcript output/VIDEO_ID/transcript.json
```

**The opening spike is not a signal.** Every curve peaks in its first seconds
because viewers start there and drift away. On the Lara Boyd talk the 0s point
scores 0.87 and the transcript there reads *"Transcriber: Jessica Lee"* — cutting
the hottest raw window gives you the credits, every time. So peaks are found by
**local prominence** — how far a point rises above its own neighbourhood — with
the intro excluded outright. That correctly surfaces the 285s peak where she
explains Braille readers have larger hand sensory areas.

The heatmap says *where* attention spiked; Claude still decides the exact cut,
because a peak marks roughly where interest rose, not where the sentence starts,
and a hot passage that only makes sense in context is still a bad clip. Each
moment carries a `heat` score and the shortlist leads with the ones the audience
returned to.

Videos without a heatmap (below the view threshold) fall back to transcript-only
selection automatically.

### Layouts

`--style broll` (default) puts stock footage full-frame with the speaker heard
but not seen. `--style speaker` uses the talk's own footage over a blurred
fill. `--style split` stacks speaker over b-roll. The raw clip is kept, so
re-rendering in another style costs no network.

### The asset library

`assets/` is a reusable, tagged pool shared by every talk you process:

```
assets/video/    b-roll     assets/music/    background beds
assets/image/    stills     assets/library.json   tags, durations, licences
```

```bash
python asset_library.py scan                      # index whatever you dropped in
python asset_library.py fetch video "runner at dawn"
python asset_library.py collect motivation "runner at dawn" "boxer training gym"
python asset_library.py validate                  # drop broken or unusable files
python asset_library.py credits                   # regenerate assets/CREDITS.md
```

`collect` fills the library for a topic from several searches at once, tagging
everything with the topic so the carousel and shorts can find it later.

**Images need no API key.** `fetch_stock` falls through four sources — Pexels
and Pixabay if a key is set, then **Openverse** (Creative Commons, filtered to
licences that allow both commercial use *and* modification) and **Wikimedia
Commons**, neither of which needs one. Wikimedia in particular is the only
practical way to get licensed photographs of specific, named people. Video
b-roll still needs a Pexels or Pixabay key.

Every fetched file records its creator, licence and source page, and
`assets/CREDITS.md` collects them in one place — CC BY and BY-SA both require
crediting the photographer wherever the image appears. `validate` removes
truncated downloads and anything whose licence forbids derivative works, which
matters here because every image gets cropped and has text laid over it.

The same library feeds the **quote carousel** (`--carousel`): one card per
moment, the hook and quote set over a matching photo with a gradient scrim so
white text stays readable over any image. It reuses the moments already found,
so it costs no extra Claude call.

### Finding images that actually match

Two things decide whether the images fit, and neither is licence filtering:

**Query shape.** These providers match fairly literally. `"boxer in empty gym"`
returns **zero** results on Openverse at any setting, while `"boxing gym"`
returns hundreds — descriptive four-word phrases read best in a prompt and are
the worst possible search input. So every query is tried in full first, then
progressively broadened (`boxer throwing punches heavy bag` → `boxer throwing`
→ `heavy bag` → `boxer`) until results appear. This was the single largest
cause of bad images: empty result sets, not bad ranking.

**Two kinds of keyword.** `topic_tags.py` produces both from one call, because
they do different jobs — hashtags (`#neuroplasticity`, `#growthmindset`) are
for the caption, and concrete visual queries (`runner at dawn`) are for the
image search. Feeding hashtags to an image API returns junk: nobody
photographs an abstraction.

```bash
python topic_tags.py "Struggle makes your brain grow stronger" --theme discipline
```

`asset_library.best_for_topic()` then fetches a small candidate pool and can
have Claude look at thumbnails to pick the best one — that catches the failure
keyword scoring can't, where an image matches the words but not the meaning (a
stadium gate for "gym", a product box for an athlete's name).

Licences are not filtered on beyond one check: **no-derivatives images are
excluded**, because every image here gets cropped and has text laid over it,
which makes it a derivative work. Attribution isn't tracked as a workflow step.

`scan` needs no key — drop files in, and tags are read from the filename
(`sunrise-runner_discipline.mp4` → `sunrise`, `runner`, `discipline`).
`fetch` pulls from **Pexels** or **Pixabay**, whose free APIs licence content
for commercial use. Add either key to `.env`:

```
PEXELS_API_KEY=your_key_here      # https://www.pexels.com/api/
PIXABAY_API_KEY=your_key_here     # https://pixabay.com/api/docs/
```

Without a key the pipeline still runs — it falls back to the speaker's own
footage and skips the music bed (`--placeholder-music` synthesises a plain
one for testing).

**On Pinterest**: it has no public API for this, its terms forbid scraping,
and its images are largely other people's copyrighted work re-pinned without
licence — which becomes your problem once they're in a video you publish.
Pexels and Pixabay give the same look with a licence attached. If you've
curated boards of images you do have rights to, save them into `assets/image/`
and `scan` will index them.

## Briefs from photos (`project_brief.py`)

For physical work — furniture, joinery, anything made by hand — where the
source material is a folder of photographs rather than a talk.

```bash
python project_brief.py path/to/photos --goal commissions
python project_brief.py path/to/photos --goal community --notes "the shelves are ash, the table is oak"
```

Goals: `commissions` (craft credibility, every post ends with a way to
enquire), `portfolio`, `community` (technical, maker-to-maker), `neutral`.

**Why it plugs into everything else**: it emits the same `brief.json` that the
transcript pipeline produces, so every existing generator works on it
unchanged:

```bash
python captions_generator.py output/projects/<name>/brief.json
python carousel_generator.py output/projects/<name>/brief.json
```

It also writes `project_brief.json` (per-piece material, construction, finish,
and a shot-by-shot video order) and a readable `brief.md`.

### It won't guess

Specifics are what sell craft work — "through-tenons wedged in contrasting
walnut" earns an enquiry where "beautiful handmade table" earns nothing. But a
confident wrong claim about timber or joinery destroys credibility with exactly
the audience worth having, and he'll be asked about it in the comments.

So anything it can't determine from the photograph goes in an `uncertain` list
to confirm before posting, rather than being filled in. On a test set it
correctly refused to treat a painting, a museum scale model and a watermarked
digital graphic as real furniture, and flagged that photos filed under
"joinery" showed no joinery at all.

Build-progress shots matter: the shot order puts process before reveal, which
holds attention far better than a gallery of finished pieces.

## Statement posters (`make_posters.py`)

The still-image counterpart to the shorts: a photo, two lines of type, nothing
else.

```bash
python make_posters.py --count 8 --signature "mindset."
python make_posters.py --count 6 --theme discipline --layout stack
python make_posters.py --from-talk VIDEO_ID     # reuse a talk's own moments
```

### The writing

`statement_writer.py` writes in one fixed shape — the antithesis:

```
setup    what to do, be, or choose      (<= 4 words)
payoff   what to reject                 (<= 4 words, usually starts with "not")
```

The prompt is built around what makes these land: the two halves must genuinely
oppose each other, and the payoff has to name a *specific* temptation. "not
your mood" works; "not negativity" doesn't, so words like negativity, excuses
and obstacles are banned outright, along with hustle-speak and any word that
could be dropped into a different statement unnoticed.

### The rendering

`poster_renderer.py` has two layouts. **bleed** runs the setup edge to edge in
enormous lowercase across the upper third with the payoff in serif italic under
its right end; **stack** puts a quiet setup over a much heavier payoff,
left-aligned. Either can be `portrait` (1080x1350), `story` (1080x1920) or
`square`.

Three things decide whether these look designed rather than stamped:

- **Placement.** Horizontal bands are scored on busyness, brightness and how
  much *skin* they contain, and the text goes in the calmest one. Type across
  someone's eyes is the placement that always looks wrong, so on a tight face
  crop — where the whole upper third is face — the headline drops below the
  chin instead.
- **Contrast.** The scrim under the text is sized from the photo *and* the text
  colour. Red is a mid-brightness colour (luma ~88), so red type on a
  background of luma 80 is invisible however different the two look described
  in words; it needs a genuinely dark backdrop. `--colour auto` picks red only
  where that's achievable and white everywhere else.
- **Margins.** The headline is lifted by a fraction of its ascent to sit
  optically right, and the top margin allows for that so nothing clips.

Output goes to `output/posters/<run>/` with a `statements.json` recording each
statement, the image behind it and that image's licence.

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
├── captions_generator.py        # brief → social captions (Claude API)
├── voiceover_script_generator.py # brief → narration script (Claude API)
├── text_to_speech.py            # script → MP3 audio (ElevenLabs API)
├── statement_writer.py          # antithesis poster statements (Claude API)
├── poster_renderer.py           # statement + photo → poster PNG (Pillow)
├── make_posters.py              # orchestrates a batch of posters
├── carousel_generator.py        # brief → carousel slide text (Claude API)
├── carousel_renderer.py         # slide text → PNG images (Pillow)
├── clip_selector.py             # timestamped transcript → best clip window (Claude API)
├── video_clipper.py             # time range → vertical MP4 (yt-dlp + ffmpeg)
├── cli.py                       # orchestrates all of the above, saves output
│
│   # sourcing
├── project_brief.py             # photos of physical work → brief.json (Claude vision)
├── topic_tags.py                # idea → hashtags (captions) + visual queries (image search)
│
│   # motivational shorts pipeline
├── heatmap.py                   # YouTube most-replayed curve → hot windows (yt-dlp)
├── moment_finder.py             # transcript → several standalone 15-35s moments (Claude API)
├── caption_timing.py            # clip → word-level karaoke captions (.ass) via local Whisper
├── asset_library.py             # tagged b-roll/stills/music library (+ Pexels/Pixabay fetch)
├── shorts_builder.py            # footage + speech + music + captions → 1080x1920 MP4 (ffmpeg)
├── make_shorts.py               # orchestrates the above, one talk → many shorts
│
├── webapp.py                    # local web UI: runs cli.py, browses ./output (stdlib only)
├── ui/                          # the UI's static files
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── requirements.txt
├── .env.example
├── assets/                      # reusable media library (media gitignored, manifest tracked)
│   ├── video/  image/  music/
│   └── library.json
└── output/                      # created at runtime
    ├── posters/                 # statement poster runs
    ├── projects/<name>/         # photo-driven briefs: project_brief.json, brief.json, brief.md
    └── <video_id>/
        ├── transcript.json
        ├── brief.json
        ├── blog_post.md
        ├── twitter_thread.json
        ├── twitter_thread.txt
        ├── captions.json
        ├── captions.txt
        ├── voiceover_script.txt   (if ElevenLabs configured)
        ├── voiceover.mp3          (if ElevenLabs configured)
        ├── carousel/
        │   ├── carousel.json
        │   └── slide_01.png, slide_02.png, ...
        ├── clip_info.json
        ├── short_form_clip.mp4
        └── shorts/                    # motivational shorts pipeline
            ├── index.json
            └── 01_<theme>_<hook>/
                ├── short.mp4          # the finished vertical short
                ├── clip_raw.mp4       # downloaded window, kept for re-renders
                ├── captions.ass
                └── moment.json        # hook, quote, theme, asset credits
```

## Status

All 8 original target formats are implemented: blog post, Twitter/X thread,
Instagram carousel, TikTok/social captions, optional voice-over,
short-form video, and captions. Natural next steps beyond this: TikTok-specific
script formatting (currently covered by the general captions generator),
and publishing integrations to push finished content to each platform
automatically.
