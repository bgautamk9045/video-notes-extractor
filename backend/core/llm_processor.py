"""
LLM Processor — FREE LOCAL VERSION using Ollama
================================================
Replaces GPT-4o and Claude with a local model running on your machine
via Ollama (https://ollama.com).

What is Ollama?
  Ollama is a tool that lets you run open-source LLMs (like Llama 3,
  Mistral, Phi-3) entirely on your own computer — no internet, no
  API key, no cost.  It exposes a simple HTTP API on localhost:11434.

Which model should I use?
  RAM < 8GB  → phi3   or mistral  (4GB models)
  RAM  8GB+  → llama3             (8GB model, best quality)
  RAM 16GB+  → llama3:13b         (even better)
  RAM 40GB+  → llama3:70b         (near GPT-4 quality)

Install Ollama:
  https://ollama.com/download
  Then run:  ollama pull llama3

The prompts here are identical to the paid version — only the HTTP
call changes (from api.openai.com to localhost:11434).
"""

import json
import logging
import re
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ── How many characters we send per chunk ────────────────────
# Local models typically have smaller context windows than GPT-4o.
# llama3 = 8192 tokens ≈ 28,000 chars.  We stay conservative.
CHUNK_CHARS = 8000
OVERLAP_CHARS = 500   # 10% overlap


# ─────────────────────────────────────────────────────────────
# Prompts  (same as paid version — model-agnostic)
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert note-taker specialising in extracting structured
knowledge from video transcripts (lectures, meetings, podcasts).
IMPORTANT: Always respond with valid JSON only — no markdown fences, no extra text,
no explanation before or after the JSON object."""

NOTES_PROMPT = """Analyse this video transcript and produce organised notes.

VIDEO TITLE: {title}
DURATION: {duration}

TRANSCRIPT:
---
{transcript}
---

Respond with ONLY this JSON (no other text):
{{
  "notes": [
    {{
      "heading": "Main section heading",
      "content": "2-4 sentence summary. Use **bold** for key terms.\\n- Bullet point facts",
      "timestamp_ref": 0.0,
      "subsections": [
        {{
          "heading": "Sub-topic",
          "content": "Key points\\n- Point 1\\n- Point 2",
          "timestamp_ref": 60.0,
          "subsections": []
        }}
      ]
    }}
  ],
  "key_topics": ["topic1", "topic2", "topic3"]
}}

Rules:
- 3 to 6 main sections
- Each section: 2 to 4 subsections
- timestamp_ref = seconds when that topic first appears
- key_topics: 5 to 8 concise labels"""

TIMESTAMPS_PROMPT = """Find the most important moments in this video transcript.

VIDEO TITLE: {title}
DURATION: {duration}

TRANSCRIPT (times shown as [MM:SS]):
---
{transcript}
---

Respond with ONLY this JSON:
{{
  "timestamps": [
    {{
      "time_seconds": 0,
      "time_label": "00:00",
      "title": "Short title (max 7 words)",
      "description": "One sentence: why this moment matters",
      "importance": "high"
    }}
  ]
}}

Rules:
- importance: "high" | "medium" | "low"
- Include 3 to 5 timestamps
- high = major topic shift, key insight, conclusion, important definition
- Skip filler ("let me now turn to...", "any questions?")"""

ACTIONS_PROMPT = """Extract all action items and tasks from this transcript.

TRANSCRIPT:
---
{transcript}
---

Respond with ONLY this JSON:
{{
  "action_items": [
    {{
      "task": "What needs to be done (one clear sentence)",
      "owner": "Person or role mentioned, or null",
      "due_date": "Date/timeframe if mentioned, or null",
      "priority": "urgent",
      "context": "The sentence from the transcript that led to this task"
    }}
  ]
}}

priority: "urgent" (deadline/blocker) | "normal" (clearly assigned) | "low" (casually mentioned)
If no action items exist, return {{"action_items": []}}"""

SUMMARY_PROMPT = """Write an executive summary of this video for someone who has not watched it.

VIDEO TITLE: {title}
KEY TOPICS: {topics}

TRANSCRIPT:
---
{transcript}
---

Respond with ONLY this JSON:
{{
  "summary": "3 to 5 sentences. First sentence = one-line TL;DR. Remaining sentences cover the main points and why they matter. Active voice, present tense."
}}"""


# ─────────────────────────────────────────────────────────────
# Processor
# ─────────────────────────────────────────────────────────────

class LLMProcessor:
    """
    Sends transcript to a LOCAL Ollama model.
    No API key. No internet. Runs 100% on your machine.
    """

    def __init__(
        self,
        model: str = "llama3",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        summary_temperature: float = 0.3,
    ):
        self.model       = model
        self.base_url    = base_url.rstrip("/")
        self.temperature = temperature
        self.sum_temp    = summary_temperature

        # Validate Ollama is reachable at startup
        self._check_ollama()

    # ── Public ───────────────────────────────────────────────

    def extract_all(self, transcript: str, segments: list,
                    video_title: str, duration: float) -> dict:
        dur_label = _fmt_time(duration)
        ts_text   = _embed_timestamps(segments)
        chunks    = _chunk(transcript, CHUNK_CHARS, OVERLAP_CHARS)

        logger.info("  Ollama model: %s | chunks: %d", self.model, len(chunks))

        # ── Notes + topics ────────────────────────────────────
        all_notes, all_topics = [], []
        for i, chunk in enumerate(chunks):
            logger.info("  Notes chunk %d/%d …", i + 1, len(chunks))
            raw  = self._call(NOTES_PROMPT.format(
                title=video_title, duration=dur_label, transcript=chunk))
            data = _parse(raw)
            all_notes.extend(data.get("notes", []))
            all_topics.extend(data.get("key_topics", []))
        key_topics = list(dict.fromkeys(all_topics))[:8]

        # ── Timestamps ────────────────────────────────────────
        logger.info("  Extracting timestamps …")
        ts_raw  = self._call(TIMESTAMPS_PROMPT.format(
            title=video_title, duration=dur_label,
            transcript=ts_text[:CHUNK_CHARS]))
        timestamps = _parse(ts_raw).get("timestamps", [])

        # ── Action items ──────────────────────────────────────
        logger.info("  Extracting action items …")
        all_actions = []
        for chunk in chunks:
            ai_raw = self._call(ACTIONS_PROMPT.format(transcript=chunk))
            all_actions.extend(_parse(ai_raw).get("action_items", []))

        # ── Summary ───────────────────────────────────────────
        logger.info("  Generating summary …")
        sum_raw  = self._call(
            SUMMARY_PROMPT.format(
                title=video_title,
                topics=", ".join(key_topics),
                transcript=transcript[:CHUNK_CHARS * 2]),
            temperature=self.sum_temp)
        summary = _parse(sum_raw).get("summary", "No summary generated.")

        return dict(notes=all_notes, timestamps=timestamps,
                    action_items=all_actions, summary=summary,
                    key_topics=key_topics)

    # ── Ollama HTTP call ──────────────────────────────────────

    def _call(self, user_prompt: str, temperature: float = None) -> str:
        """
        POST to Ollama's /api/chat endpoint.
        Ollama runs locally at http://localhost:11434 — no internet needed.
        """
        import urllib.request, urllib.error

        temp = temperature if temperature is not None else self.temperature

        payload = json.dumps({
            "model":  self.model,
            "stream": False,
            "options": {"temperature": temp, "num_predict": 800,"stop": ["```"]},
            "messages": [
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": user_prompt},
            ],
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                data = json.loads(resp.read())
                return data["message"]["content"]
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.base_url}.\n"
                "Make sure Ollama is running:  ollama serve\n"
                f"Original error: {e}"
            )

    def _check_ollama(self):
        """Quick check that Ollama is running before we start processing."""
        import urllib.request, urllib.error
        try:
            urllib.request.urlopen(f"{self.base_url}/api/tags", timeout=3)
            logger.info("  Ollama is running at %s", self.base_url)
        except Exception:
            logger.warning(
                "Ollama not reachable at %s — make sure you ran: ollama serve",
                self.base_url
            )


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _parse(raw: str) -> dict:
    """Strip markdown fences and parse JSON from LLM response."""
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$",          "", raw.strip())
    # Extract first {...} block (handles models that add prose before/after)
    match = re.search(r"\{[\s\S]+\}", raw)
    if match:
        raw = match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("JSON parse failed (%s). Raw: %.200s", e, raw)
        return {}


def _chunk(text: str, size: int, overlap: int) -> list:
    if len(text) <= size:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        if end < len(text):
            # Break at sentence boundary
            bp = text.rfind(". ", start, end)
            if bp != -1:
                end = bp + 2
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def _embed_timestamps(segments: list) -> str:
    parts = []
    for seg in segments:
        label = _fmt_time(seg["start"])
        parts.append(f"[{label}] {seg['text']}")
    return " ".join(parts)


def _fmt_time(secs: float) -> str:
    secs = int(secs or 0)
    h, r = divmod(secs, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"