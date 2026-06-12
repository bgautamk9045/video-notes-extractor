#!/usr/bin/env python3
"""
Video Note Extractor — FREE LOCAL CLI
======================================
Runs 100% on your machine. No API keys. No internet (except YouTube download).

Quick start:
  1. Install Ollama:  https://ollama.com/download
  2. Pull a model:    ollama pull llama3
  3. Start Ollama:    ollama serve          (keep this running in another terminal)
  4. Run this tool:   python cli.py "https://youtu.be/VIDEO_ID"

Examples:
  # YouTube video (default settings from .env)
  python cli.py "https://www.youtube.com/watch?v=VIDEO_ID"

  # Local video file
  python cli.py ./my_lecture.mp4

  # Use a faster/smaller model
  python cli.py "https://youtu.be/..." --model phi3

  # Higher accuracy Whisper
  python cli.py ./meeting.mp4 --whisper small

  # Print output to terminal as well as saving to file
  python cli.py "https://youtu.be/..." --print

  # JSON output only
  python cli.py ./lecture.mp4 --format json
"""

import argparse
import json
import sys
import os
from pathlib import Path

# Make sure imports work from project root
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()   # load .env before anything else

from backend.core.pipeline import VideoNotePipeline


def main():
    parser = argparse.ArgumentParser(
        description="Convert videos to notes, timestamps and action items — 100% local and free.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("source",
        help="YouTube URL or path to a local video/audio file")

    parser.add_argument("--whisper", default=None,
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: from .env or 'base'). "
             "Bigger = slower but more accurate.")

    parser.add_argument("--model", default=None,
        help="Ollama model name (default: from .env or 'llama3'). "
             "Must be pulled first with: ollama pull <model>")

    parser.add_argument("--language", default=None,
        help="Audio language, e.g. 'en', 'hi', 'es'. Default: auto-detect.")

    parser.add_argument("--output", default=None,
        help="Output folder (default: ./outputs)")

    parser.add_argument("--format", default="both",
        choices=["markdown", "json", "both"],
        help="What to save: markdown, json, or both (default: both)")

    parser.add_argument("--print", action="store_true",
        help="Also print the Markdown output to terminal after saving")

    parser.add_argument("--verbose", action="store_true",
        help="Show detailed debug logging")

    args = parser.parse_args()

    # Banner
    print("\n" + "═" * 58)
    print("  🎬  Video Note Extractor  —  Local & Free")
    print("═" * 58)
    print(f"  Source : {args.source}")
    print(f"  Whisper: {args.whisper or os.getenv('WHISPER_MODEL', 'base')}")
    print(f"  LLM    : {args.model   or os.getenv('OLLAMA_MODEL',  'llama3')} (Ollama)")
    print(f"  Output : {args.output  or './outputs'}")
    print("═" * 58 + "\n")

    try:
        pipeline = VideoNotePipeline(
            whisper_model    = args.whisper,
            whisper_language = args.language,
            ollama_model     = args.model,
            output_dir       = args.output,
            verbose          = args.verbose,
        )
        result = pipeline.run(args.source)
    except RuntimeError as e:
        # Friendly error for common problems
        msg = str(e)
        if "Ollama" in msg or "Cannot reach" in msg:
            print("\n❌  Ollama is not running!")
            print("    Fix: open a NEW terminal and run:  ollama serve")
            print("    Then try this command again.\n")
        else:
            print(f"\n❌  Error: {e}\n")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"\n❌  File not found: {e}\n")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠  Interrupted.\n")
        sys.exit(1)

    # Summary
    print("\n" + "═" * 58)
    print(f"  ✅  Done!  '{result.video_title}'")
    print(f"  ⏱  Duration  : {_fmt(result.duration_seconds)}")
    print(f"  📝  Sections  : {len(result.notes)}")
    print(f"  ⏰  Timestamps: {len(result.timestamps)}")
    print(f"  ✔  Actions   : {len(result.action_items)}")
    print(f"  ⚡  Elapsed   : {result.metadata.get('elapsed_sec', '?')}s")
    print(f"  📁  Saved to  : {args.output or './outputs'}/")
    print("═" * 58 + "\n")

    # Optional terminal output
    if args.print or args.format == "markdown":
        print(result.to_markdown())
    if args.format == "json":
        print(json.dumps(result.to_dict(), indent=2))


def _fmt(secs):
    secs = int(secs or 0)
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


if __name__ == "__main__":
    main()