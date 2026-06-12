"""
Pipeline — FREE LOCAL VERSION
==============================
Reads all settings from .env file.
No API keys. No paid services. Everything on your machine.

Usage:
    from backend.core.pipeline import VideoNotePipeline
    pipeline = VideoNotePipeline()   # reads .env automatically
    result   = pipeline.run("https://youtu.be/...")
"""

import os
import json
import time
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()   # reads .env file from project root

from .audio_extractor  import AudioExtractor
from .transcriber      import Transcriber
from .llm_processor    import LLMProcessor
from .output_formatter import OutputFormatter

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Data models (unchanged from paid version)
# ─────────────────────────────────────────────────────────────

@dataclass
class Timestamp:
    time_seconds:  float
    time_label:    str
    title:         str
    description:   str
    importance:    str   # "high" | "medium" | "low"

@dataclass
class ActionItem:
    task:          str
    owner:         Optional[str]
    due_date:      Optional[str]
    priority:      str   # "urgent" | "normal" | "low"
    context:       str

@dataclass
class NoteSection:
    heading:       str
    content:       str
    subsections:   list
    timestamp_ref: Optional[float] = None

@dataclass
class ExtractionResult:
    video_title:        str
    duration_seconds:   float
    processed_at:       str
    summary:            str
    notes:              list
    timestamps:         list
    action_items:       list
    key_topics:         list
    transcript_excerpt: str
    metadata:           dict = field(default_factory=dict)

    def to_dict(self):  return asdict(self)
    def to_markdown(self): return OutputFormatter.to_markdown(self)


# ─────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────

class VideoNotePipeline:
    """
    End-to-end free/local pipeline.

    All settings read from .env — just call VideoNotePipeline()
    and everything is configured automatically.

    You can still override any setting by passing arguments:
        pipeline = VideoNotePipeline(whisper_model="medium")
    """

    def __init__(
        self,
        whisper_model:   str = None,   # default: from .env or "base"
        whisper_language:str = None,   # default: from .env or "auto"
        ollama_model:    str = None,   # default: from .env or "llama3"
        ollama_base_url: str = None,   # default: from .env or localhost:11434
        output_dir:      str = None,   # default: from .env or "./outputs"
        verbose:         bool = False,
    ):
        # Read from .env, with fallback defaults
        self.whisper_model    = whisper_model    or os.getenv("WHISPER_MODEL",    "tiny")
        self.whisper_language = whisper_language or os.getenv("WHISPER_LANGUAGE", "auto")
        self.ollama_model     = ollama_model     or os.getenv("OLLAMA_MODEL",     "llama3")
        self.ollama_base_url  = ollama_base_url  or os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
        self.output_dir       = Path(output_dir  or os.getenv("OUTPUT_DIR",       "./outputs"))
        self.temp_dir         = Path(os.getenv("TEMP_DIR", "./tmp"))
        self.verbose          = verbose

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            format="%(asctime)s  %(levelname)s  %(message)s",
            datefmt="%H:%M:%S",
            level=level,
        )

        logger.info("Config — Whisper: %s | Ollama: %s (%s)",
                    self.whisper_model, self.ollama_model, self.ollama_base_url)

        # Instantiate components
        self._audio = AudioExtractor(output_dir=str(self.temp_dir))
        self._asr   = Transcriber(
            model_size=self.whisper_model,
            language=None if self.whisper_language == "auto" else self.whisper_language,
        )
        self._llm   = LLMProcessor(
            model=self.ollama_model,
            base_url=self.ollama_base_url,
        )

    # ── Public entry point ────────────────────────────────────

    def run(self, source: str) -> ExtractionResult:
        t0 = time.time()
        logger.info("=" * 55)
        logger.info("Starting pipeline for: %s", source)
        logger.info("=" * 55)

        # Step 1 — Audio
        logger.info("[1/4] Extracting audio …")
        audio_path, title, duration = self._audio.extract(source)
        logger.info("      Audio ready — %.0f seconds, '%s'", duration, title)

        # Step 2 — Transcription
        logger.info("[2/4] Transcribing with Whisper (%s) …", self.whisper_model)
        logger.info("      This may take a few minutes for long videos …")
        segments = self._asr.transcribe(audio_path)
        full_tx  = " ".join(s["text"] for s in segments)
        logger.info("      Transcript: %d words across %d segments",
                    len(full_tx.split()), len(segments))

        # Step 3 — LLM extraction
        logger.info("[3/4] Running Ollama (%s) …", self.ollama_model)
        logger.info("      This runs locally — no internet needed …")
        llm_out = self._llm.extract_all(
            transcript=full_tx,
            segments=segments,
            video_title=title,
            duration=duration,
        )

        # Step 4 — Assemble
        logger.info("[4/4] Assembling output …")
        result = ExtractionResult(
            video_title        = title,
            duration_seconds   = duration,
            processed_at       = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            summary            = llm_out["summary"],
            notes              = llm_out["notes"],
            timestamps         = llm_out["timestamps"],
            action_items       = llm_out["action_items"],
            key_topics         = llm_out["key_topics"],
            transcript_excerpt = full_tx[:500],
            metadata           = {
                "source"        : source,
                "whisper_model" : self.whisper_model,
                "ollama_model"  : self.ollama_model,
                "elapsed_sec"   : round(time.time() - t0, 1),
            },
        )

        self._save(result)
        elapsed = time.time() - t0
        logger.info("=" * 55)
        logger.info("Done in %.0fs — check the outputs/ folder!", elapsed)
        logger.info("=" * 55)
        return result

    # ── Save outputs ──────────────────────────────────────────

    def _save(self, result: ExtractionResult):
        slug = result.video_title[:40].replace(" ", "_").replace("/", "-")
        ts   = time.strftime("%Y%m%d_%H%M%S")
        base = self.output_dir / f"{ts}_{slug}"

        md_path   = base.with_suffix(".md")
        json_path = base.with_suffix(".json")
        tx_path   = base.with_name(base.name + "_transcript.txt")

        md_path.write_text(result.to_markdown(), encoding="utf-8")
        json_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
        tx_path.write_text(result.transcript_excerpt, encoding="utf-8")

        logger.info("  Saved: %s", md_path.name)
        logger.info("  Saved: %s", json_path.name)
        logger.info("  Saved: %s", tx_path.name)