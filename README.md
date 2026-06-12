# 🎬 Video Note Extractor — Free Local Version
### Zero cost · No API keys · Runs entirely on your machine

---

## The core idea

| What you had before | What this version uses |
|---|---|
| OpenAI Whisper API (paid) | Whisper running **locally on your PC** (free) |
| GPT-4o / Claude API (paid) | **Ollama** — local LLM server (free) |
| `.env` with secret keys | `.env` with just settings — no secrets |
| Internet for every request | Internet only for YouTube download |

---

## Step 0 — Check your system first

```bash
python setup_check.py
```

This script checks everything and tells you exactly what to install.
Run it before doing anything else.

---

## Step 1 — Install Ollama (the local LLM engine)

**Ollama** runs open-source AI models (Llama 3, Mistral, Phi-3) on your own machine.
Think of it as "ChatGPT but it runs inside your laptop."

**Download:** https://ollama.com/download

After installing, open a terminal and run:
```bash
# Start the Ollama server (keep this terminal open while using the tool)
ollama serve
```

Then in a NEW terminal, pull a model:
```bash
# Pick ONE based on your RAM:

ollama pull phi3      # 4 GB RAM minimum  — fast, decent quality
ollama pull mistral   # 4 GB RAM minimum  — fast, good quality
ollama pull llama3    # 8 GB RAM minimum  — best quality (recommended)
```

**How to know which model to pick:**

| Your RAM | Recommended Whisper | Recommended Ollama model |
|---|---|---|
| Under 4 GB | tiny | phi3 |
| 4–8 GB | base | phi3 or mistral |
| 8–16 GB | base or small | llama3 (default) |
| 16 GB+ | small or medium | llama3 or llama3:13b |

---

## Step 2 — Install ffmpeg

ffmpeg is a free audio processing tool. Install it at the system level:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian / WSL
sudo apt install ffmpeg

# Windows
# Download from: https://www.gyan.dev/ffmpeg/builds/
# Extract to C:\ffmpeg, then add C:\ffmpeg\bin to your PATH
# Restart VS Code after adding to PATH
```

---

## Step 3 — Set up the project

```bash
# Clone or download the project, then enter it
cd video-note-extractor

# Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt
```

---

## Step 4 — Configure .env (no keys needed!)

Your `.env` file only has settings — no secrets:

```env
WHISPER_MODEL=base
WHISPER_LANGUAGE=auto
OLLAMA_MODEL=llama3
OLLAMA_BASE_URL=http://localhost:11434
OUTPUT_DIR=./outputs
```

Change `WHISPER_MODEL` and `OLLAMA_MODEL` based on the RAM table above.

---

## Step 5 — Run it!

**Make sure Ollama is running first** (`ollama serve` in a separate terminal).

```bash
# Process a YouTube video
python cli.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Process a local video file
python cli.py ./my_lecture.mp4

# Use a faster model for quick testing
python cli.py "https://youtu.be/..." --model phi3 --whisper tiny

# Print results in terminal
python cli.py ./meeting.mp3 --print

# See all options
python cli.py --help
```

**Output files appear in `./outputs/` folder:**
- `YYYYMMDD_HHMMSS_VideoTitle.md`   ← human-readable notes
- `YYYYMMDD_HHMMSS_VideoTitle.json` ← structured data
- `YYYYMMDD_HHMMSS_VideoTitle_transcript.txt` ← raw transcript

---

## Common errors and fixes

### "Ollama not reachable"
```bash
# Open a NEW terminal and run:
ollama serve
# Keep it running, then try again
```

### "model not found"
```bash
# Pull the model first:
ollama pull llama3
# or for smaller RAM:
ollama pull phi3
```

### "ffmpeg not found"
Reinstall ffmpeg and restart VS Code. Check with: `ffmpeg -version`

### Whisper is very slow
Switch to a smaller model in `.env`:
```env
WHISPER_MODEL=tiny
```

### Output quality is poor / incoherent
Switch to a better Ollama model:
```bash
ollama pull llama3      # if you have 8 GB RAM
ollama pull mistral     # good alternative
```
Then update `.env`: `OLLAMA_MODEL=llama3`

### Out of memory / system crashes
You're running a model too large for your RAM.
```bash
# Switch to smaller model:
ollama pull phi3
```
Update `.env`: `OLLAMA_MODEL=phi3`, `WHISPER_MODEL=tiny`

---

## Project structure

```
video-note-extractor/
├── .env                          ← your settings (no secrets!)
├── setup_check.py                ← run this first
├── cli.py                        ← main entry point
├── requirements.txt
├── backend/
│   ├── core/
│   │   ├── pipeline.py           ← orchestrates everything
│   │   ├── audio_extractor.py    ← ffmpeg + yt-dlp
│   │   ├── transcriber.py        ← Whisper (runs locally)
│   │   ├── llm_processor.py      ← Ollama (runs locally)
│   │   └── output_formatter.py   ← Markdown + JSON
│   └── api/
│       └── server.py             ← FastAPI server
├── frontend/
│   └── index.html                ← web UI demo
├── outputs/                      ← results saved here
└── tmp/                          ← temp audio files
```

---

## How it compares to the paid version

| Feature | Paid version | This version |
|---|---|---|
| Cost | $0.006/min (Whisper) + LLM tokens | **$0** |
| Privacy | Audio sent to OpenAI/Anthropic servers | **Stays on your machine** |
| Internet needed | Yes, for every request | Only for YouTube download |
| Quality | GPT-4o (best available) | Llama 3 (very good, 80–90% of GPT-4 quality) |
| Speed | Fast (server-side GPU) | Slower (your CPU/GPU) |
| Setup | Just add API key | Install Ollama + pull model |
