#!/usr/bin/env python3
"""
setup_check.py — Run this FIRST before anything else.

Checks your system and tells you exactly what to install.
Usage:  python setup_check.py
"""

import subprocess
import sys
import shutil
import platform
import urllib.request


def check(label, ok, fix_msg=""):
    status = "✅" if ok else "❌"
    print(f"  {status}  {label}")
    if not ok and fix_msg:
        for line in fix_msg.split("\n"):
            print(f"       {line}")
    return ok


def run(cmd):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        return r.returncode == 0, r.stdout.strip()
    except Exception:
        return False, ""


print("\n" + "═" * 58)
print("  Video Note Extractor — System Check")
print("═" * 58)

all_ok = True

# Python version
ok, out = run([sys.executable, "--version"])
ver = sys.version_info
ver_ok = ver.major == 3 and ver.minor >= 10
all_ok &= check(
    f"Python 3.10+ ({sys.version.split()[0]})",
    ver_ok,
    "Download from: https://python.org/downloads"
)

# ffmpeg
ffmpeg_ok = shutil.which("ffmpeg") is not None
all_ok &= check(
    "ffmpeg installed",
    ffmpeg_ok,
    {
        "Darwin":  "brew install ffmpeg",
        "Linux":   "sudo apt install ffmpeg",
        "Windows": "Download: https://www.gyan.dev/ffmpeg/builds/\n"
                   "         Then add ffmpeg/bin to your PATH",
    }.get(platform.system(), "See: https://ffmpeg.org/download.html")
)

# yt-dlp
ytdlp_ok = shutil.which("yt-dlp") is not None
all_ok &= check(
    "yt-dlp installed",
    ytdlp_ok,
    "pip install yt-dlp"
)

# Ollama running
try:
    urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2)
    ollama_running = True
except Exception:
    ollama_running = False

all_ok &= check(
    "Ollama server running (localhost:11434)",
    ollama_running,
    "1. Install Ollama: https://ollama.com/download\n"
    "2. Open a NEW terminal and run:  ollama serve\n"
    "3. Keep that terminal open, then re-run this script"
)

# Ollama models
if ollama_running:
    try:
        import json
        with urllib.request.urlopen("http://localhost:11434/api/tags") as r:
            data = json.loads(r.read())
        models = [m["name"] for m in data.get("models", [])]
        has_model = len(models) > 0
        model_list = ", ".join(models) if models else "none"
        all_ok &= check(
            f"Ollama models available ({model_list})",
            has_model,
            "Pull a model with:  ollama pull llama3\n"
            "Or a smaller one:   ollama pull phi3   (needs only 4GB RAM)"
        )
    except Exception:
        check("Ollama models", False, "ollama pull llama3")

# Python packages
pkgs = ["whisper", "dotenv", "fastapi", "uvicorn", "yt_dlp"]
missing = []
for pkg in pkgs:
    try:
        __import__(pkg)
    except ImportError:
        missing.append(pkg)

all_ok &= check(
    "Python packages installed",
    len(missing) == 0,
    f"Missing: {', '.join(missing)}\nRun: pip install -r requirements.txt"
)

print()

# RAM check
try:
    import os
    if platform.system() == "Darwin":
        ok2, mem = run(["sysctl", "-n", "hw.memsize"])
        ram_gb = int(mem) / 1e9 if ok2 else 0
    elif platform.system() == "Linux":
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal"):
                    ram_gb = int(line.split()[1]) / 1e6
                    break
    else:
        ram_gb = 8  # Can't easily detect on Windows without psutil

    print("  💾  RAM detected: ~{:.0f} GB".format(ram_gb))
    if ram_gb < 4:
        print("     ⚠  4GB or less — use Whisper 'tiny' and model 'phi3'")
    elif ram_gb < 8:
        print("     ℹ  4–8 GB — recommended: Whisper 'base', model 'phi3' or 'mistral'")
    elif ram_gb < 16:
        print("     ✅  8–16 GB — good: Whisper 'base'/'small', model 'llama3'")
    else:
        print("     ✅  16 GB+ — excellent: Whisper 'small'/'medium', model 'llama3:13b'")
except Exception:
    pass

# Recommended .env
print()
print("  📋  Recommended .env settings for your machine:")

try:
    if ram_gb < 8:
        w, m = "tiny", "phi3"
    elif ram_gb < 16:
        w, m = "base", "llama3"
    else:
        w, m = "small", "llama3"
except:
    w, m = "base", "llama3"

print(f"      WHISPER_MODEL={w}")
print(f"      OLLAMA_MODEL={m}")

print()
if all_ok:
    print("  🎉  All checks passed! You're ready to run:")
    print("      python cli.py \"https://youtu.be/VIDEO_ID\"")
else:
    print("  ⚠  Fix the issues above, then run this script again.")
print("═" * 58 + "\n")