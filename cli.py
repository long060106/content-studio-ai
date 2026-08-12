"""
cli.py

AI Content Studio — Step 1: Transcript Extraction + Content Brief

Usage:
    python cli.py "https://www.youtube.com/watch?v=VIDEO_ID"

Output:
    Saves two files into ./output/<video_id>/
        - transcript.json   (raw transcript + metadata)
        - brief.json        (structured content brief for downstream generators)
"""

import argparse
import os
import sys

from youtube_extractor import extract, InvalidYouTubeURL, NoTranscriptAvailable
from content_brief import generate_brief
from blog_generator import generate_blog_post
from twitter_thread_generator import generate_twitter_thread
from linkedin_generator import generate_linkedin_post
from captions_generator import generate_captions
from voiceover_script_generator import generate_voiceover_script
from text_to_speech import synthesize_speech
from carousel_generator import generate_carousel
from carousel_renderer import render_carousel
from clip_selector import select_clip
from video_clipper import create_short_form_clip


def run(url: str, output_dir: str = "output") -> None:
    print(f"→ Extracting transcript for: {url}")
    try:
        video = extract(url)
    except InvalidYouTubeURL as e:
        print(f"✗ {e}")
        sys.exit(1)
    except NoTranscriptAvailable as e:
        # Only reachable if the Whisper fallback itself couldn't run
        # (e.g. audio_transcriber import failed) rather than a caption issue.
        print(f"✗ No transcript available for this video: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Transcript extraction failed: {e}")
        print("  If this is a Whisper/ffmpeg error, confirm 'ffmpeg -version' works in this terminal.")
        sys.exit(1)

    print(f"  ✓ Got transcript ({len(video.transcript_text.split())} words, "
          f"language: {video.transcript_language})")
    print(f"  ✓ Title: {video.title}")

    print("→ Generating content brief with Claude...")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("✗ ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key.")
        sys.exit(1)

    try:
        brief = generate_brief(video)
    except ValueError as e:
        print(f"✗ Failed to generate brief: {e}")
        sys.exit(1)

    print(f"  ✓ Brief generated ({len(brief.key_points)} key points, "
          f"{len(brief.hooks)} hooks)")

    print("→ Generating blog post...")
    blog_post = generate_blog_post(brief, video_title=video.title)
    print(f"  ✓ Blog post generated ({len(blog_post.split())} words)")

    print("→ Generating Twitter/X thread...")
    thread = generate_twitter_thread(brief)
    print(f"  ✓ Thread generated ({len(thread.tweets)} tweets)")

    print("→ Generating LinkedIn post...")
    linkedin_result = generate_linkedin_post(brief)
    print(f"  ✓ LinkedIn post generated ({len(linkedin_result.post.split())} words, "
          f"{len(linkedin_result.claims_to_verify)} claim(s) flagged to verify)")

    print("→ Generating captions...")
    caption_set = generate_captions(brief)
    print(f"  ✓ Captions generated ({len(caption_set.captions)} variants)")

    voiceover_script = None
    voiceover_audio_path = None
    if os.environ.get("ELEVENLABS_API_KEY"):
        print("→ Generating voice-over script...")
        voiceover_script = generate_voiceover_script(brief)
        print(f"  ✓ Script generated ({len(voiceover_script.split())} words)")

        print("→ Synthesizing voice-over audio with ElevenLabs...")
        try:
            voiceover_audio_path_tmp = os.path.join(output_dir, video.video_id, "voiceover.mp3")
            synthesize_speech(voiceover_script, voiceover_audio_path_tmp)
            voiceover_audio_path = voiceover_audio_path_tmp
            print(f"  ✓ Audio saved")
        except Exception as e:
            print(f"  ⚠ Voice-over audio generation failed: {e}")
            print("  (Script was still generated and saved as text.)")
    else:
        print("→ Skipping voice-over (ELEVENLABS_API_KEY not set in .env — everything else still runs)")

    print("→ Generating Instagram carousel...")
    carousel = generate_carousel(brief, video_title=video.title)
    print(f"  ✓ Carousel text generated ({len(carousel.slides)} slides)")

    print("→ Rendering carousel slide images...")
    carousel_dir = os.path.join(output_dir, video.video_id, "carousel")
    slide_paths = render_carousel(
        [
            {"headline": s.headline, "subtext": s.subtext, "flow": s.flow, "source": s.source}
            for s in carousel.slides
        ],
        carousel_dir,
    )
    print(f"  ✓ Rendered {len(slide_paths)} slide images")

    short_clip_path = None
    clip_selection = None
    print("→ Selecting best short-form video clip...")
    try:
        clip_selection = select_clip(brief, video.transcript_segments)
        print(f"  ✓ Clip selected: {clip_selection.start_seconds:.1f}s–{clip_selection.end_seconds:.1f}s "
              f"({clip_selection.duration:.1f}s) — {clip_selection.reason}")

        print("→ Downloading and converting to vertical short-form video (this can take a minute or two)...")
        short_clip_path_tmp = os.path.join(output_dir, video.video_id, "short_form_clip.mp4")
        create_short_form_clip(
            url, clip_selection.start_seconds, clip_selection.end_seconds, short_clip_path_tmp
        )
        short_clip_path = short_clip_path_tmp
        print(f"  ✓ Short-form video saved")
    except Exception as e:
        print(f"  ⚠ Short-form video generation failed: {e}")
        print("  (Everything else in the pipeline still completed successfully.)")

    video_dir = os.path.join(output_dir, video.video_id)
    os.makedirs(video_dir, exist_ok=True)

    transcript_path = os.path.join(video_dir, "transcript.json")
    brief_path = os.path.join(video_dir, "brief.json")
    blog_path = os.path.join(video_dir, "blog_post.md")
    thread_json_path = os.path.join(video_dir, "twitter_thread.json")
    thread_text_path = os.path.join(video_dir, "twitter_thread.txt")
    linkedin_path = os.path.join(video_dir, "linkedin_post.md")
    linkedin_claims_path = os.path.join(video_dir, "linkedin_claims_to_verify.txt")
    captions_json_path = os.path.join(video_dir, "captions.json")
    captions_text_path = os.path.join(video_dir, "captions.txt")
    voiceover_script_path = os.path.join(video_dir, "voiceover_script.txt")
    carousel_json_path = os.path.join(video_dir, "carousel", "carousel.json")
    clip_info_path = os.path.join(video_dir, "clip_info.json")

    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(video.to_json())

    with open(brief_path, "w", encoding="utf-8") as f:
        f.write(brief.to_json())

    with open(blog_path, "w", encoding="utf-8") as f:
        f.write(blog_post)

    with open(thread_json_path, "w", encoding="utf-8") as f:
        f.write(thread.to_json())

    with open(thread_text_path, "w", encoding="utf-8") as f:
        f.write(thread.to_text())

    with open(linkedin_path, "w", encoding="utf-8") as f:
        f.write(linkedin_result.post)

    if linkedin_result.claims_to_verify:
        with open(linkedin_claims_path, "w", encoding="utf-8") as f:
            f.write("Claims/quotes to verify against the original video before publishing:\n\n")
            f.write("\n".join(f"- {c}" for c in linkedin_result.claims_to_verify))

    with open(captions_json_path, "w", encoding="utf-8") as f:
        f.write(caption_set.to_json())

    with open(captions_text_path, "w", encoding="utf-8") as f:
        f.write(caption_set.to_text())

    if voiceover_script:
        with open(voiceover_script_path, "w", encoding="utf-8") as f:
            f.write(voiceover_script)

    with open(carousel_json_path, "w", encoding="utf-8") as f:
        f.write(carousel.to_json())

    if clip_selection:
        import json as _json
        with open(clip_info_path, "w", encoding="utf-8") as f:
            f.write(_json.dumps({
                "start_seconds": clip_selection.start_seconds,
                "end_seconds": clip_selection.end_seconds,
                "duration_seconds": clip_selection.duration,
                "reason": clip_selection.reason,
            }, indent=2))

    print(f"\n✓ Done. Saved to:")
    print(f"    {transcript_path}")
    print(f"    {brief_path}")
    print(f"    {blog_path}")
    print(f"    {thread_json_path}")
    print(f"    {thread_text_path}")
    print(f"    {linkedin_path}")
    if linkedin_result.claims_to_verify:
        print(f"    {linkedin_claims_path}  ⚠ review before publishing")
    print(f"    {captions_json_path}")
    print(f"    {captions_text_path}")
    if voiceover_script:
        print(f"    {voiceover_script_path}")
    if voiceover_audio_path:
        print(f"    {voiceover_audio_path}")
    print(f"    {carousel_json_path}")
    for p in slide_paths:
        print(f"    {p}")
    if clip_selection:
        print(f"    {clip_info_path}")
    if short_clip_path:
        print(f"    {short_clip_path}")


def main():
    parser = argparse.ArgumentParser(
        description="AI Content Studio — extract a YouTube transcript and generate a content brief."
    )
    parser.add_argument("url", help="YouTube video URL (or bare 11-char video ID)")
    parser.add_argument(
        "--output-dir", default="output", help="Directory to save results into (default: ./output)"
    )
    args = parser.parse_args()
    run(args.url, args.output_dir)


if __name__ == "__main__":
    main()
