"""
Audio Extractor
===============
Handles:
  • YouTube URLs  → yt-dlp download → audio WAV
  • Local video   → ffmpeg strip audio → WAV
  • Local audio   → ffmpeg normalise  → WAV

Output: 16 kHz, mono, 16-bit PCM WAV — optimal for Whisper.
"""

import os
import re
import subprocess
import tempfile
import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

# Regex to validate YouTube URLs
YT_PATTERN = re.compile(
    r"(https?://)?(www\.)?(youtube\.com|youtu\.be)/.+"
)


class AudioExtractor:
    """
    Extracts and preprocesses audio from any video/audio source.

    Returns:
        (audio_path: str, video_title: str, duration_seconds: float)
    """

    def __init__(self, output_dir: str = "./tmp"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public ───────────────────────────────────

    def extract(self, source: str) -> Tuple[str, str, float]:
        """Route to the correct handler based on source type."""
        source = source.strip()
        if YT_PATTERN.match(source):
            return self._from_youtube(source)
        else:
            path = Path(source)
            if not path.exists():
                raise FileNotFoundError(f"File not found: {source}")
            return self._from_local(path)

    # ── Handlers ─────────────────────────────────

    def _from_youtube(self, url: str) -> Tuple[str, str, float]:
        """
        Download best audio stream from YouTube using yt-dlp.
        Steps:
          1. yt-dlp --print title  → get video title
          2. yt-dlp -x --audio-format wav → download + convert
          3. ffmpeg post-process   → 16kHz mono normalisation
        """
        logger.info("  YT-DLP: fetching metadata …")

        # --- Get title ---
        title_cmd = [
            "yt-dlp", "--print", "title",
            "--no-playlist", url
        ]
        title_result = subprocess.run(
            title_cmd, capture_output=True, text=True, check=True
        )
        video_title = title_result.stdout.strip() or "Unknown Video"

        # --- Download audio ---
        raw_path = self.output_dir / "yt_raw_audio.%(ext)s"
        dl_cmd = [
            "yt-dlp",
            "-x",                         # extract audio only
            "--audio-format", "wav",
            "--audio-quality", "0",       # best quality
            "--no-playlist",
            "-o", str(raw_path),
            url,
        ]
        logger.info("  YT-DLP: downloading audio …")
        subprocess.run(dl_cmd, check=True, capture_output=not logger.isEnabledFor(logging.DEBUG))

        downloaded = list(self.output_dir.glob("yt_raw_audio.*"))
        if not downloaded:
            raise RuntimeError("yt-dlp download produced no file")

        raw_file = downloaded[0]

        # --- Normalise with ffmpeg ---
        out_path = self.output_dir / "audio_processed.wav"
        duration = self._ffmpeg_normalise(raw_file, out_path)
        raw_file.unlink(missing_ok=True)

        logger.info("  Audio ready: %.1fs  '%s'", duration, video_title)
        return str(out_path), video_title, duration

    def _from_local(self, path: Path) -> Tuple[str, str, float]:
        """
        Process local video or audio file.
        Uses ffmpeg to strip/convert audio to 16kHz mono WAV.
        """
        logger.info("  Local file: %s", path)
        out_path = self.output_dir / "audio_processed.wav"
        video_title = path.stem.replace("_", " ").replace("-", " ")
        duration = self._ffmpeg_normalise(path, out_path)
        logger.info("  Audio ready: %.1fs  '%s'", duration, video_title)
        return str(out_path), video_title, duration

    # ── ffmpeg helpers ───────────────────────────

    def _ffmpeg_normalise(self, input_path: Path, output_path: Path) -> float:
        """
        Convert any audio/video to 16kHz mono WAV with loudnorm filter.
        Returns duration in seconds.

        ffmpeg flags explained:
          -vn          : drop video stream
          -acodec pcm_s16le : raw PCM 16-bit little-endian
          -ac 1        : mono
          -ar 16000    : 16 kHz sample rate (Whisper optimum)
          -af loudnorm : EBU R128 loudness normalisation
        """
        cmd = [
            "ffmpeg", "-y",
            "-i", str(input_path),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ac",  "1",
            "-ar",  "16000",
            "-af",  "loudnorm=I=-16:LRA=11:TP=-1.5",
            str(output_path),
        ]
        logger.debug("ffmpeg cmd: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-2000:]}")

        # Parse duration from ffmpeg stderr
        duration = self._parse_duration(result.stderr)
        return duration

    @staticmethod
    def _parse_duration(ffmpeg_stderr: str) -> float:
        """Extract duration in seconds from ffmpeg stderr output."""
        match = re.search(r"Duration:\s*(\d+):(\d+):(\d+)\.(\d+)", ffmpeg_stderr)
        if match:
            h, m, s, cs = match.groups()
            return int(h)*3600 + int(m)*60 + int(s) + int(cs)/100
        return 0.0