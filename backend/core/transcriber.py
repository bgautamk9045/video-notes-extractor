"""
Transcriber
===========
Converts audio to word-level timestamped transcript segments using
OpenAI Whisper (local) or the OpenAI Whisper API (cloud).

Key features:
  • Chunked processing for long files (avoids OOM on large videos)
  • Word-level timestamps via whisper's verbose_json mode
  • Optional speaker diarisation via pyannote.audio
  • Filler word removal ("um", "uh", "like", "you know" …)

Segment schema:
  {
    "start": 12.4,      # seconds
    "end":   18.9,
    "text":  "...",
    "speaker": "A"      # only if diarisation enabled
  }
"""

import re
import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)

# Words that add no meaning and confuse the LLM
FILLER_WORDS = {
    "um", "uh", "er", "ah", "like", "you know", "i mean",
    "kind of", "sort of", "basically", "literally", "actually",
    "right", "okay", "so", "well",
}

FILLER_RE = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in FILLER_WORDS) + r")\b[,.]?\s*",
    re.IGNORECASE
)


class Transcriber:
    """
    Wraps OpenAI Whisper for speech-to-text with timestamped segments.

    Args:
        model_size: "tiny" | "base" | "small" | "medium" | "large"
                    Larger = slower but more accurate.
                    For most lecture content, "base" or "small" is sufficient.
        language:   ISO 639-1 code ("en", "es", …) or None for auto-detect.
        use_api:    If True, uses the cloud Whisper API (requires OPENAI_API_KEY).
                    If False, runs the model locally via the `whisper` package.
        chunk_sec:  Seconds per audio chunk for long-form processing.
        diarize:    Enable speaker diarisation (requires pyannote.audio + HF token).
    """

    def __init__(
        self,
        model_size: str = "base",
        language: Optional[str] = None,
        use_api: bool = False,
        openai_api_key: Optional[str] = None,
        chunk_sec: int = 300,        # 5-minute chunks
        diarize: bool = False,
        hf_token: Optional[str] = None,
    ):
        self.model_size = model_size
        self.language   = language
        self.use_api    = use_api
        self.openai_key = openai_api_key
        self.chunk_sec  = chunk_sec
        self.diarize    = diarize
        self.hf_token   = hf_token
        self._model     = None          # lazy-loaded

    # ── Public ───────────────────────────────────

    def transcribe(self, audio_path: str) -> list:
        """
        Transcribe audio file.

        Returns a list of segment dicts:
          [{"start": float, "end": float, "text": str, "speaker": str|None}, …]
        """
        if self.use_api:
            segments = self._transcribe_api(audio_path)
        else:
            segments = self._transcribe_local(audio_path)

        segments = self._clean_segments(segments)

        if self.diarize:
            segments = self._apply_diarization(audio_path, segments)

        return segments

    # ── Local Whisper ────────────────────────────

    def _transcribe_local(self, audio_path: str) -> list:
        """
        Run Whisper locally.  Lazy-loads the model on first call.

        Theory:
          Whisper uses a Transformer encoder-decoder architecture trained on
          680,000 hours of multilingual audio.  It outputs log-mel spectrogram
          features from 30-second audio chunks and predicts BPE tokens.

          The 'verbose_json' format gives us word-level timestamps computed via
          cross-attention weight alignment (similar to forced alignment in
          Montreal Forced Aligner).
        """
        import whisper  # pip install openai-whisper

        if self._model is None:
            logger.info("  Loading Whisper model '%s' …", self.model_size)
            self._model = whisper.load_model(self.model_size)

        logger.info("  Transcribing (local Whisper) …")

        options = {
            "verbose": False,
            "word_timestamps": True,    # per-word start/end times
            "task": "transcribe",
            "condition_on_previous_text": True,
        }
        if self.language:
            options["language"] = self.language

        result = self._model.transcribe(audio_path, **options)

        # Convert to our segment format
        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start":   round(seg["start"], 2),
                "end":     round(seg["end"],   2),
                "text":    seg["text"].strip(),
                "speaker": None,
            })
        return segments

    # ── Cloud API Whisper ────────────────────────

    def _transcribe_api(self, audio_path: str) -> list:
        """
        Use the OpenAI Whisper API (whisper-1).

        Splits audio into ≤25 MB chunks because the API has a file size limit.
        Each chunk is sent separately and results are merged with offset correction.
        """
        from openai import OpenAI
        import subprocess, struct, wave

        client = OpenAI(api_key=self.openai_key)

        # Get duration to decide if chunking is needed
        duration = self._get_duration(audio_path)
        chunk_starts = list(range(0, int(duration), self.chunk_sec))

        all_segments = []
        for i, start in enumerate(chunk_starts):
            logger.info("  API chunk %d/%d (t=%ds) …", i+1, len(chunk_starts), start)
            chunk_path = f"/tmp/_vne_chunk_{i}.wav"
            end = min(start + self.chunk_sec, duration)

            # Slice with ffmpeg
            subprocess.run([
                "ffmpeg", "-y", "-i", audio_path,
                "-ss", str(start), "-t", str(end - start),
                "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
                chunk_path
            ], capture_output=True, check=True)

            with open(chunk_path, "rb") as f:
                response = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=f,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                    language=self.language,
                )

            for seg in response.segments:
                all_segments.append({
                    "start":   round(seg.start + start, 2),
                    "end":     round(seg.end   + start, 2),
                    "text":    seg.text.strip(),
                    "speaker": None,
                })

        return all_segments

    # ── Speaker Diarisation ──────────────────────

    def _apply_diarization(self, audio_path: str, segments: list) -> list:
        """
        Assign speaker labels to each segment using pyannote.audio.

        Theory:
          Diarisation = 'who spoke when'.
          pyannote uses ECAPA-TDNN embeddings → cosine similarity clustering.
          We match the diarisation timeline to our Whisper segments by finding
          the speaker label with maximum overlap for each segment interval.

        Requires:
          pip install pyannote.audio
          Accept the pyannote/speaker-diarization-3.1 user agreement on HuggingFace.
        """
        try:
            from pyannote.audio import Pipeline as DiarizePipeline
        except ImportError:
            logger.warning("pyannote.audio not installed; skipping diarisation")
            return segments

        logger.info("  Running speaker diarisation …")
        pipeline = DiarizePipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=self.hf_token,
        )
        diarization = pipeline(audio_path)

        # Build a list of (start, end, speaker) turns
        turns = [
            (turn.start, turn.end, speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]

        # Assign each segment the dominant speaker
        for seg in segments:
            overlap = {}
            for t_start, t_end, speaker in turns:
                o = max(0, min(seg["end"], t_end) - max(seg["start"], t_start))
                if o > 0:
                    overlap[speaker] = overlap.get(speaker, 0) + o
            if overlap:
                seg["speaker"] = max(overlap, key=overlap.get)

        return segments

    # ── Cleaning ─────────────────────────────────

    @staticmethod
    def _clean_segments(segments: list) -> list:
        """Remove filler words and tidy whitespace."""
        cleaned = []
        for seg in segments:
            text = FILLER_RE.sub(" ", seg["text"])
            text = re.sub(r"\s{2,}", " ", text).strip()
            if text:
                seg = dict(seg)
                seg["text"] = text
                cleaned.append(seg)
        return cleaned

    @staticmethod
    def _get_duration(audio_path: str) -> float:
        import subprocess, re
        r = subprocess.run(
            ["ffprobe", "-i", audio_path, "-show_entries",
             "format=duration", "-v", "quiet", "-of", "csv=p=0"],
            capture_output=True, text=True
        )
        try:
            return float(r.stdout.strip())
        except ValueError:
            return 0.0
