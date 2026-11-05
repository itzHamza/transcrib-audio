"""
Local test harness for audio2pdf — runs the EXACT same code path as the
dockerized service, directly on this machine (Windows-friendly).

Usage:
  python test.py                          # default URL (auto: subtitles first)
  python test.py <youtube-or-audio-url>
  python test.py <url> --no-summary       # skip the OpenRouter summary
  python test.py <url> --force-audio      # skip subtitles, force the audio path
  python test.py <url> --lang ar          # force summary language
  python test.py <url> --title "Cours ..."  # override PDF/file title

Env: reads .env next to this file (falls back to the main app's .env for the
OpenRouter/DetectLanguage keys). Output: ./out/<title>.pdf + auto-open.
"""

import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

HERE = Path(__file__).parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)

DEFAULT_URL = "https://www.youtube.com/watch?v=Larj8qcbfyw"

# ── 1. Env loading (service .env first, main-app .env as fallback, no override)
from dotenv import load_dotenv  # noqa: E402

load_dotenv(HERE / ".env")
load_dotenv(Path(r"C:\xampp\htdocs\next\my-app\.env"))  # won't override existing
if not os.getenv("OPENROUTER_API_KEY") and os.getenv("NEXT_OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = os.getenv("NEXT_OPENROUTER_API_KEY")

# ── 2. Local yt-dlp shim: the package is installed but no yt-dlp.exe script
#    exists on this machine (pip 3.14 quirk), so route the service's
#    subprocess.run(["yt-dlp", ...]) through `python -m yt_dlp` transparently.
_orig_run = subprocess.run


def _run_shim(cmd, *a, **k):
    if isinstance(cmd, (list, tuple)) and cmd and str(cmd[0]).lower() in ("yt-dlp",):
        cmd = [sys.executable, "-m", "yt_dlp", *list(cmd)[1:]]
    return _orig_run(cmd, *a, **k)


subprocess.run = _run_shim

import app  # noqa: E402  (registers fonts + logging config)

# Same cookie injection the production service performs at FastAPI startup.
app._YT_GLOBAL_ARGS.extend(app._yt_cookie_args())
if app._YT_GLOBAL_ARGS:
    print("[env] yt-dlp cookies enabled")

# ── 3. CLI args
args = [a for a in sys.argv[1:]]
url = next((a for a in args if not a.startswith("--")), DEFAULT_URL)
force_audio = "--force-audio" in args
no_summary = "--no-summary" in args


def _after(flag: str) -> str | None:
    if flag in args:
        i = args.index(flag)
        if i + 1 < len(args):
            return args[i + 1]
    return None


lang = _after("--lang")
title_arg = _after("--title")
summarize = not no_summary


async def main() -> None:
    t0 = time.time()
    progress = app.ProgressReporter(None, None)  # console-only (logging)

    source = "?"
    full_text = ""

    # ── Step 1: subtitles (fast path)
    if not force_audio:
        subs_dir = tempfile.mkdtemp(prefix="yt_subs_")
        t = time.time()
        candidates = app._yt_fetch_subtitles(url, subs_dir)
        print(f"[1] caption files: {[p.name for p in candidates]}  ({time.time() - t:.1f}s)")
        best_text, best_lang_tag = "", ""
        for f in sorted(candidates, key=lambda p: app._yt_sub_lang_rank(p.name)):
            text = app._vtt_or_srt_to_text(f)
            if len(text) >= 200:
                best_text = text
                parts = f.stem.split(".")
                best_lang_tag = parts[1] if len(parts) > 1 else "auto"
                break
        if best_text:
            source = f"subtitles ({best_lang_tag})"
            full_text = best_text
            print(f"[1] USING SUBTITLES — {len(full_text)} chars — preview:")
            print("    " + full_text[:260].replace("\n", " ") + " …")
    else:
        print("[1] --force-audio → subtitles check skipped")

    # ── Step 2 (fallback): audio download → normalize check → transcribe
    if not full_text:
        if shutil.which("ffmpeg") is None:
            print("[2] ffmpeg NOT on PATH — audio fallback unavailable locally. Install it or use a video with captions.")
            return
        if not os.getenv("GROQ_API_KEY"):
            print("[2] GROQ_API_KEY missing in .env — audio fallback needs it for transcription.")
            return
        t = time.time()
        tmp_path, meta = app._yt_download_audio(url)
        print(f"[2] audio downloaded: {Path(tmp_path).stat().st_size / (1024*1024):.1f} MB  ({time.time() - t:.1f}s)")
        info = app.probe_audio(tmp_path)
        print(f"[2] probe: {info['duration']/60:.1f} min")
        chunks = await app.chunk_audio(tmp_path, info, progress)
        print(f"[2] chunks: {len(chunks)}")
        texts = await app.transcribe_chunks(chunks, progress)
        full_text = "\n\n".join(t2 for t2 in texts if t2).strip()
        source = "audio (Groq whisper)"
        print(f"[2] transcription: {len(full_text)} chars  ({time.time() - t:.1f}s)")

    # ── Step 3: language + summary (summary is SKIPPED for subtitles source)
    detected = app.detect_language(full_text[:1500], "en")
    print(f"[3] detected language: {detected}")
    summary = ""
    if summarize and not best_text:
        t = time.time()
        summary = await app.summarize_transcript(full_text, detected, progress)
        print(f"[3] summary: {len(summary)} chars ({time.time() - t:.1f}s)")
    elif best_text:
        print("[3] summary skipped (subtitles are already clean text)")

    # ── Step 4: PDF
    fetched_title = None if title_arg else (app._yt_fetch_title(url) or "Transcription YouTube")
    safe_title = app._sanitize_title(title_arg or fetched_title or "Test", "")
    pdf_path, _pdf_name = app.generate_pdf(full_text, summary, safe_title, detected)

    clean = re.sub(r'[<>:"/\\|?*]', "", safe_title)[:60] or "yt-test"
    dest = OUT / f"{clean}-{time.strftime('%H%M%S')}.pdf"
    shutil.copyfile(pdf_path, dest)

    minutes_total = (time.time() - t0) / 60
    print(f"[4] PDF written: {dest}")
    print(f"    source={source} | lang={detected} | summary={'yes' if summary else 'no'} | total={minutes_total:.1f} min")
    try:
        os.startfile(dest)  # open for visual check
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(main())
