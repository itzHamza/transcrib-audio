"""
Audio → PDF transcription service (TBiB Space).

Pipeline per job:
  1. Download the audio (streamed to /tmp)
  2. ffprobe validation (catches truncated recordings, e.g. "moov atom not found")
  3. ONE ffmpeg pass → 16 kHz mono MP3 (fixes every codec/container mismatch,
     including the old AAC-into-OGG segfault, and shrinks files below Groq's
     25 MB limit for typical lectures)
  4. Adaptive chunking only when the normalized file still exceeds the safe
     limit (Groq accepts up to 80 MB per request; we target 60 MB chunks)
  5. Parallel Groq whisper-large-v3 transcription (bounded concurrency)
  6. One whole-document structured summary (map-reduce only for very long docs)
     — deliberately detailed: language-mixed transcripts are summarized in the
     requested language WITHOUT dropping technical/clinical details
  7. Unicode PDF rendering with proper Arabic shaping (Noto Naskh / fallback)

Optional live progress: when a `job` + `callback` are supplied, every stage
POSTs an event to the main app so the UI can show live logs.

Blocking work (requests/ffmpeg/yt-dlp/sync SDK calls) runs inside
asyncio.to_thread so the single uvicorn worker stays responsive (health checks
and unrelated requests never queue behind a running job).
"""

import asyncio
import json
import logging
import math
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
import requests
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from groq import Groq
from openai import OpenAI
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

load_dotenv()

# ── Logging (stdout, uvicorn streams it; secrets never logged) ────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("audio2pdf")

has_openrouter_key = bool(os.getenv("OPENROUTER_API_KEY"))
if not has_openrouter_key:
    log.warning("OPENROUTER_API_KEY not configured — summarization disabled")
has_groq_key = bool(os.getenv("GROQ_API_KEY"))
if not has_groq_key:
    log.warning("GROQ_API_KEY not configured — transcription will fail")
log.info("audio2pdf service starting")

openrouter_client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY") or "",
    base_url="https://openrouter.ai/api/v1",
)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY") or "")

# ── Config ────────────────────────────────────────────────────────────────────
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500 MB raw download cap
SAFE_TRANSCRIBE_BYTES = 24 * 1024 * 1024
# NOTE: Groq returns "413 Payload Too Large" well below 80MB on this plan
# (observed live on a normalized ~30MB file) — budget stays conservative at
# 24MB so chunk_audio() engages instead of shipping an over-limit single chunk.
MIN_CHUNK_SECONDS = 90
SUMMARY_SINGLE_PASS_CHARS = 60_000
SUMMARY_MODEL = "google/gemini-2.5-flash-lite"
GROQ_MODEL = "whisper-large-v3"
TRANSCRIBE_CONCURRENCY = 5

ALLOWED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".m4b", ".aac", ".ogg", ".oga",
    ".flac", ".mp4", ".webm", ".wma", ".opus",
}
ALLOWED_URL_SCHEMES = ("http://", "https://")

ALLOWED_ORIGINS = [
    o.strip()
    for o in (os.getenv("ALLOWED_ORIGINS") or "https://tbib.space,http://localhost:3000").split(",")
    if o.strip()
]
ALLOWED_CALLBACK_HOSTS = [
    o.strip().lower().lstrip("*")
    for o in (os.getenv("ALLOWED_CALLBACK_HOSTS") or "tbib.space,localhost,127.0.0.1").split(",")
    if o.strip()
]

app = FastAPI(title="Audio to PDF API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")

LANGUAGE_NAMES = {"fr": "français", "ar": "arabe (العربية)", "en": "English", "es": "español", "de": "Deutsch"}

# ── Progress reporting (live logs) ───────────────────────────────────────────


class ProgressReporter:
    """Posts stage events to the main app so the UI can show live logs."""

    def __init__(self, job: Optional[str], callback: Optional[str]):
        self.job = (job or "").strip()
        self.callback = (callback or "").strip()
        self.enabled = bool(self.job and self.callback)
        self.sequence = 0

    def validate_callback(self) -> bool:
        if not self.enabled:
            return False
        try:
            host = httpx.URL(self.callback).host.lower()
        except Exception:
            return False
        return any(host == h or host.endswith("." + h) for h in ALLOWED_CALLBACK_HOSTS)

    def ping(self) -> bool:
        """One probe POST — any HTTP response means the callback is reachable."""
        headers = {}
        secret = os.getenv("AUDIO_PROGRESS_SECRET")
        if secret:
            headers["x-audio-progress-secret"] = secret
        try:
            requests.post(
                self.callback,
                json={
                    "job": self.job,
                    "event": {"stage": "ping", "message": "service actif", "pct": 0},
                },
                headers=headers,
                timeout=4,
            )
            return True
        except Exception:
            return False

    def engage(self) -> None:
        """Validate callback host + one-shot connectivity probe; disables noisy retries."""
        if not self.job or not self.callback:
            self.enabled = False
            return
        if not self.validate_callback():
            log.warning("[job %s] callback host not allowed — live logs disabled", self.job[:8])
            self.enabled = False
            return
        if not self.ping():
            log.warning(
                "[job %s] progress callback unreachable at startup — live logs disabled for this job",
                self.job[:8],
            )
            self.enabled = False

    def emit(
        self,
        stage: str,
        message: str,
        pct: Optional[int] = None,
        level: str = "info",
        status: Optional[str] = None,
    ) -> None:
        self.sequence += 1
        log.info("[job %s] %s | %s (pct=%s)", (self.job or "-")[:8], stage, message, pct)
        if not self.enabled:
            return
        payload = {
            "job": self.job,
            "event": {
                "seq": self.sequence,
                "ts": int(time.time() * 1000),
                "stage": stage,
                "message": str(message)[:300],
                "pct": pct,
                "level": level,
                "status": status,
            },
        }
        headers = {}
        secret = os.getenv("AUDIO_PROGRESS_SECRET")
        if secret:
            headers["x-audio-progress-secret"] = secret
        try:
            requests.post(self.callback, json=payload, headers=headers, timeout=5)
        except Exception as exc:  # never fail the job because of logging
            log.warning("[job %s] progress post failed: %s", self.job[:8], exc)

    def done(self, message: str = "Terminé") -> None:
        self.emit("done", message, pct=100, status="done")

    def failed(self, message: str) -> None:
        self.emit("failed", message, level="error", status="failed")


# ── ff helpers ───────────────────────────────────────────────────────────────


def run_ffmpeg_sync(cmd: List[str], timeout: int = 600) -> Tuple[bool, str]:
    """Run an ffmpeg/ffprobe command; returns (ok, stderr_tail)."""
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    stderr_tail = (proc.stderr or "")[-4000:]
    return proc.returncode == 0, stderr_tail


def probe_audio(path: str) -> Dict[str, Any]:
    """Return duration (s), size (bytes) and a validation verdict."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration,size,bit_rate",
        "-of", "json", path,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Analyse du fichier audio trop lente (timeout).") from exc

    stderr = (proc.stderr or "").lower()
    if "moov atom not found" in stderr or "does not contain any stream" in stderr or "invalid data" in stderr:
        raise HTTPException(
            status_code=422,
            detail=(
                "Fichier audio invalide ou coupé (enregistrement interrompu). "
                "Relancez l'enregistrement puis réessayez l'upload."
            ),
        )
    if proc.returncode != 0:
        raise HTTPException(status_code=422, detail="Fichier audio illisible : ce format n'est pas reconnu.")

    try:
        info = json.loads(proc.stdout or "{}")
        fmt = info.get("format", {})
        duration = float(fmt.get("duration") or 0)
        size = int(fmt.get("size") or 0)
        bit_rate = int(fmt.get("bit_rate") or 0)
    except (ValueError, TypeError):
        duration, size, bit_rate = 0, 0, 0

    if duration <= 0 or size <= 0:
        raise HTTPException(status_code=422, detail="Fichier audio vide ou non lisible.")

    return {"duration": duration, "size": size, "bit_rate": bit_rate}


async def download_audio(url: str, expected_ext: str) -> Tuple[str, Dict[str, Any]]:
    """Streamed download with a hard size cap. Runs in a worker thread."""
    def _download() -> Tuple[str, Dict[str, Any]]:
        tmp_path = None
        try:
            with requests.get(url, stream=True, timeout=30, allow_redirects=True) as resp:
                content_length = int(resp.headers.get("content-length") or 0)
                if content_length > MAX_FILE_SIZE:
                    raise HTTPException(status_code=413, detail="Fichier audio trop volumineux (max 500 Mo).")
                resp.raise_for_status()
                suffix = expected_ext or Path(url.split("?", 1)[0]).suffix or ".mp3"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp_path = tmp.name
                    downloaded = 0
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        downloaded += len(chunk)
                        if downloaded > MAX_FILE_SIZE:
                            raise HTTPException(status_code=413, detail="Fichier audio trop volumineux (max 500 Mo).")
                        tmp.write(chunk)
            return tmp_path, {"bytes": downloaded}
        except HTTPException:
            _silent_unlink(tmp_path)
            raise
        except requests.RequestException as exc:
            _silent_unlink(tmp_path)
            raise HTTPException(
                status_code=502,
                detail="Impossible de télécharger le fichier audio (serveur source injoignable).",
            ) from exc
        except Exception as exc:
            _silent_unlink(tmp_path)
            raise HTTPException(status_code=500, detail="Erreur inattendue pendant le téléchargement.") from exc

    return await asyncio.to_thread(_download)


def _silent_unlink(path: Optional[str]) -> None:
    if not path:
        return
    try:
        os.unlink(path)
    except OSError:
        pass


async def normalize_audio(input_path: str, progress: ProgressReporter) -> Tuple[str, Dict[str, Any]]:
    """
    Single ffmpeg pass to 16 kHz mono MP3.

    Why: whisper only needs 16 kHz mono → tiny files (≈10 MB per full hour),
    every codec/container mismatch disappears (AAC, opus, whatever), the 25 MB
    Groq limit stops being a hazard, and downstream `segment copy` slicing is
    always container-safe (MP3).
    """
    out_dir = tempfile.mkdtemp(prefix="norm_")
    out_path = str(Path(out_dir) / "norm.mp3")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", input_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "libmp3lame", "-q:a", "5",
        out_path,
    ]
    progress.emit("normalize", "Conversion audio (16 kHz mono)…", pct=22)

    def _convert() -> Tuple[bool, str]:
        return run_ffmpeg_sync(cmd, timeout=900)

    ok, stderr = await asyncio.to_thread(_convert)
    if not ok or not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        # Fallback: keep original bytes (works for already-clean inputs) and
        # slice it as-is; validation errors are honored from probe_audio().
        log.warning("normalize failed, falling back to raw copy: %s", stderr[:300])
        return input_path, probe_audio(input_path)

    info = probe_audio(out_path)  # also validates the result
    progress.emit("normalize", "Conversion terminée", pct=30)
    return out_path, info


async def chunk_audio(
    normalized_path: str,
    info: Dict[str, Any],
    progress: ProgressReporter,
) -> List[str]:
    """Slice only when needed; MP3 segmentation is always container-safe."""
    size = info.get("size") or 0
    duration = info.get("duration") or 0
    if size <= SAFE_TRANSCRIBE_BYTES or duration <= 0 or not Path(normalized_path).suffix == ".mp3":
        return [normalized_path]

    out_dir = tempfile.mkdtemp(prefix="chunks_")
    # Sizing derived purely from the size ratio (with a 10% safety margin so
    # the floor() never lands a segment over the per-request budget).
    chunk_seconds = int(
        max(
            MIN_CHUNK_SECONDS,
            min(1800, math.floor(duration * (SAFE_TRANSCRIBE_BYTES / size) * 0.9)),
        )
    )
    output_pattern = str(Path(out_dir) / "part_%03d.mp3")
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", normalized_path,
        "-f", "segment", "-segment_time", str(chunk_seconds),
        "-reset_timestamps", "1", "-c", "copy",
        output_pattern,
    ]
    progress.emit(
        "chunking",
        f"Découpage en tranches (~{chunk_seconds // 60} min) pour l'API…",
        pct=34,
    )

    def _slice() -> Tuple[bool, str]:
        return run_ffmpeg_sync(cmd, timeout=600)

    ok, _stderr = await asyncio.to_thread(_slice)
    if not ok:
        raise HTTPException(status_code=500, detail="Erreur interne pendant le découpage audio.")
    parts = sorted(str(p) for p in Path(out_dir).glob("part_*.mp3"))
    if not parts:
        raise HTTPException(status_code=500, detail="Erreur interne pendant le découpage audio.")
    return parts


async def transcribe_chunks(
    chunk_paths: List[str],
    progress: ProgressReporter,
) -> List[str]:
    """Groq whisper-large-v3, chunk-level auto language detection,并行 bounded."""
    if len(chunk_paths) == 1:
        progress.emit("transcribe", "Transcription du document (IA)…", pct=45)
    else:
        progress.emit(
            "transcribe", f"Transcription en {len(chunk_paths)} tranches (IA)…", pct=45
        )

    sem = asyncio.Semaphore(TRANSCRIBE_CONCURRENCY)
    results: List[Optional[str]] = [None] * len(chunk_paths)
    fatal: list = []

    async def _one(idx: int, path: str) -> None:
        async with sem:
            def _call() -> str:
                with open(path, "rb") as fh:
                    transcription = groq_client.audio.transcriptions.create(
                        file=(Path(path).name, fh.read()),
                        model=GROQ_MODEL,
                        # No `language=` override: whisper auto-detects per chunk,
                        # which is what mixed Arabic/French/English lectures need.
                        response_format="text",
                        temperature=0.0,
                    )
                return (transcription if isinstance(transcription, str) else getattr(transcription, "text", "")) or ""

            # One retry per chunk: covers single-shot API/network blips without
            # losing transcript content; a chunk that fails twice fails the job.
            last_err: Optional[Exception] = None
            text: Optional[str] = None
            for attempt in (0, 1):
                if attempt:
                    await asyncio.sleep(2)
                try:
                    text = await asyncio.to_thread(_call)
                    last_err = None
                    break
                except Exception as exc:
                    message = str(exc)
                    if "too large" in message.lower() or "payload too large" in message.lower():
                        raise HTTPException(
                            status_code=422,
                            detail="Un extrait dépasse la taille max de l'API de transcription. Réessayez.",
                        ) from exc
                    last_err = exc

            if text is None:
                fatal.append((idx, str(last_err)))
            results[idx] = text or ""
            done = sum(1 for r in results[: idx + 1] if r is not None)
            pct = 45 + int(40 * done / len(chunk_paths))
            progress.emit("transcribe", f"Tranche {done}/{len(chunk_paths)} transcrite", pct=pct)

    await asyncio.gather(*[_one(i, p) for i, p in enumerate(chunk_paths)])
    if fatal:
        raise HTTPException(
            status_code=502,
            detail="La transcription IA a échoué sur une ou plusieurs tranches. Réessayez dans un instant.",
        )
    return [r or "" for r in results]


def detect_language(text: str, fallback: str = "en") -> str:
    """Best-effort text language detection for the summary prompt."""
    api_key = os.getenv("DETECTLANGUAGE_API_KEY")
    if not api_key or not text:
        return fallback
    sample = text[:1500]
    try:
        resp = requests.post(
            "https://ws.detectlanguage.com/0.2/detect",
            headers={"Authorization": f"Bearer {api_key}"},
            data={"q": sample},
            timeout=10,
        )
        resp.raise_for_status()
        detections = resp.json().get("data", {}).get("detections") or []
        if detections:
            return detections[0].get("language") or fallback
        return fallback
    except Exception as exc:
        log.warning("language detection failed: %s", exc)
        return fallback


def summary_prompt(name: str, text: str, section_hint: str = "") -> str:
    """
    Deliberately COMPLETE summary: the old prompt ("clearly and concisely")
    discarded medical detail. This one demands structure + all technical info.
    """
    return (
        f"Rédige une synthèse DÉTAILLÉE et STRUCTURÉE en français du transcript suivant{section_hint}. "
        f"IMPORTANT: ta réponse doit être ENTIÈREMENT en {name}.\n\n"
        "CONSIGNES:\n"
        "1. Marque des titres (##), sous-titres et listes à puces (markdown).\n"
        "2. NE RÉSUME PAS TROP COURT: conserve TOUTES les informations techniques et "
        "cliniques (définitions, classifications, signes/symptômes, examens, bilans, "
        "dosages, mécanismes, complications, traitements, contre-indications, cas cliniques).\n"
        "3. Le transcript peut mélanger plusieurs langues (arabe/français/anglais): traduis les "
        "concepts dans la langue demandée, en conservant la nomenclature latine/médicale d'origine "
        "entre parenthèses quand utile.\n"
        "4. Termine par un tableau récapitulatif des termes clés (terme | définition).\n"
        "5. N'invente rien: si une information est absente du transcript, ne l'ajoute pas.\n"
        "6. Longueur cible: environ 800 à 1200 mots — dense et enrichi, mais sans "
        "expansion inutile: chaque phrase doit porter une information.\n\n"
        f"TRANSCRIPT:\n\n{text}"
    )


async def summarize_transcript(full_text: str, language: str, progress: ProgressReporter) -> str:
    """
    Whole-document summary (parallel map-reduce for long transcripts) instead
    of the old per-10-minute-chunk summaries that produced disconnected minis.

    Resilience: one failed slice never aborts the pipeline — failed map parts
    are skipped (reduce works on whatever succeeded) and an empty result simply
    ships the PDF with the full transcript, no summary section.
    """
    if not has_openrouter_key or not full_text:
        return ""

    name = LANGUAGE_NAMES.get(language) or language
    progress.emit("summary", "Rédaction de la synthèse détaillée…", pct=88)

    def _call_llm_sync(prompt: str) -> str:
        response = openrouter_client.chat.completions.create(
            model=SUMMARY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=8192,
        )
        return (response.choices[0].message.content or "").strip()

    async def _call_llm(prompt: str) -> Optional[str]:
        try:
            text = await asyncio.to_thread(_call_llm_sync, prompt)
            return text or None  # guard against empty 200 bodies
        except Exception as exc:
            log.warning("summary LLM call failed: %s", exc)
            return None

    try:
        if len(full_text) <= SUMMARY_SINGLE_PASS_CHARS:
            summary = await _call_llm(summary_prompt(name, full_text))
            if summary:
                return summary
            log.warning("single-pass summary empty/failed — shipping transcript-only PDF")
            return ""

        # map-reduce: dense per-slice extraction in parallel, then one merge.
        step = 40_000
        slices = [full_text[i : i + step] for i in range(0, len(full_text), step)]
        sem = asyncio.Semaphore(3)

        async def _map(idx: int, sl: str) -> Optional[str]:
            async with sem:
                return await _call_llm(
                    summary_prompt(
                        name,
                        sl,
                        section_hint=f" (PARTIE {idx + 1}/{len(slices)} du document)",
                    )
                )

        partials = await asyncio.gather(*[_map(i, s) for i, s in enumerate(slices)])
        # preserve document order; drop failed slices
        usable = [p for p in (p for p in partials) if p]
        if not usable:
            progress.emit("summary", "Synthèse indisponible — transcription complète jointe", level="warning")
            return ""

        progress.emit(
            "summary", f"{len(usable)}/{len(slices)} parties traitées — fusion finale…", pct=92
        )
        merged = await _call_llm(
            f"Fusionne ces synthèses partielles en UNE synthèse détaillée UNIQUE, "
            f"structure en titres markdown, SANS répétitions et SANS perdre d'informations, "
            f"en {name} uniquement:\n\n" + "\n\n---\n\n".join(usable)
        )
        if not merged:
            # Merge failed: fall back to joining the partials verbatim —
            # every detail survives, just less polished.
            progress.emit("summary", "Fusion indisponible — assemblage direct des parties", level="warning")
            return "\n\n---\n\n".join(usable)
        return merged
    except Exception as exc:
        log.warning("summary generation failed: %s", exc)

    return ""


# ── PDF rendering (Unicode + Arabic shaping) ─────────────────────────────────

_FONT_PATTERNS = [
    "NotoNaskhArabic-Regular.ttf",
    "*NotoNaskhArabic-Regular.ttf",
    "*Amiri-Regular.ttf",
]

FONT_CONF: Dict[str, Dict[str, Any]] = {}


def _register_unicode_fonts() -> None:
    """Find a system Arabic-capable font (fonts-noto-core docker package)."""
    search_roots = ["/usr/share/fonts", "/System/Library/Fonts", "C:/Windows/Fonts", "/usr/local/share/fonts"]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for pattern in _FONT_PATTERNS:
            matches = []
            try:
                matches = list(Path(root).rglob(pattern))
            except OSError:
                continue
            if matches:
                font_path = str(matches[0])
                try:
                    pdfmetrics.registerFont(TTFont("Unicode", font_path))
                    log.info("registered Unicode PDF font: %s", font_path)
                    FONT_CONF["regular"] = {"name": "Unicode", "arabic": True}
                    return
                except Exception as exc:
                    log.warning("font load failed: %s | %s", font_path, exc)
    FONT_CONF["regular"] = {"name": "Helvetica", "arabic": False}
    log.info("no Arabic font found; falling back to Helvetica (latin-1 only)")


_register_unicode_fonts()


def _shape_text(text: str) -> str:
    """Shape+Bidi any text containing Arabic; latin text passes through."""
    if not ARABIC_RANGE.search(text):
        return text
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except Exception as exc:  # deps missing / shaping hiccup
        log.warning("arabic shaping unavailable: %s", exc)
        return text


def _wrap_lines(
    text: str,
    font_name: str,
    font_size: int,
    max_width: float,
) -> List[str]:
    """Greedy word wrap that splits at the measured character level at most."""
    out: List[str] = []
    for paragraph in text.split("\n"):
        words = _shape_text(paragraph).split(" ")
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip() if current else word
            if pdfmetrics.stringWidth(candidate, font_name, font_size) <= max_width:
                current = candidate
                continue
            if current:
                out.append(current)
            # Very long words (URLs / shaped runs) get hard-split
            while pdfmetrics.stringWidth(word, font_name, font_size) > max_width and word:
                # binary-ish search for max prefix that fits
                lo, hi = 1, len(word)
                best = 1
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if pdfmetrics.stringWidth(word[:mid], font_name, font_size) <= max_width:
                        best, lo = mid, mid + 1
                    else:
                        hi = mid - 1
                out.append(word[:best])
                word = word[best:]
            current = word
        if current:
            out.append(current)
    return out


def generate_pdf(
    full_text: str,
    summary: str,
    source_title: str = "Transcription Audio",
    detected_language: str = "en",
) -> Tuple[str, str]:
    """Render the transcript PDF; returns (path, filename)."""
    temp_pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    font_conf = FONT_CONF.get("regular") or {"name": "Helvetica", "arabic": False}
    font_name = font_conf["name"]

    c = canvas.Canvas(temp_pdf_path, pagesize=A4)
    width, height = A4
    left, right = 18 * mm, width - 18 * mm
    max_text_width = right - left
    top = height - 20 * mm

    def new_page():
        c.showPage()

    y = top

    def draw_wrapped(text: str, size: int = 11, gap_after: int = 6, font: Optional[str] = None) -> None:
        nonlocal y
        f = font or font_name
        for line in _wrap_lines(text, f, size, max_text_width):
            if y < 20 * mm:
                new_page()
                y = top
            c.setFont(f, size)
            c.drawString(left, y, line)
            y -= (size * 1.45)
        y -= gap_after

    def heading(text: str, size: int = 16):
        nonlocal y
        if y < 40 * mm:
            new_page()
            y = top
        display = _shape_text(text)
        f = font_name if (ARABIC_RANGE.search(text) or font_conf.get("arabic")) else "Helvetica"
        c.setFont(f, size)
        c.drawString(left, y, display)
        y -= size + 8

    # Title block
    heading(_shape_text(source_title or "Transcription Audio"), size=16)
    c.setFont("Helvetica", 9)
    meta = f"Langue détectée: {detected_language} • Généré par audio2pdf • {time.strftime('%Y-%m-%d %H:%M')}"
    c.drawString(left, y, meta)
    y -= 18

    # Summary section
    if summary:
        heading("Synthèse détaillée", size=13)
        draw_wrapped(summary, size=11)
        if y < 60 * mm:
            new_page()

    heading("Transcription complète", size=13)
    draw_wrapped(full_text, size=10, gap_after=4)

    c.save()
    return temp_pdf_path, "transcript.pdf"


# ── Shared pipeline ──────────────────────────────────────────────────────────


def validate_url(url: str) -> Optional[str]:
    u = (url or "").strip()
    if not u:
        raise HTTPException(status_code=400, detail="URL manquante.")
    if not u.lower().startswith(ALLOWED_URL_SCHEMES):
        raise HTTPException(status_code=400, detail="URL invalide (http/https requis).")
    suffix = Path(u.split("?", 1)[0]).suffix.lower()
    if suffix and suffix not in ALLOWED_EXTENSIONS:
        # Unknown suffix: reject BUT with clear guidance (old behavior was a
        # bare 400 without detail)
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise HTTPException(status_code=400, detail=f"Extension non supportée ({supported}).")
    return suffix


def cleanup_files(paths):
    for p in paths or []:
        _silent_unlink(p)


async def process_audio_pipeline(
    url: str,
    summarize: bool,
    lang_hint: Optional[str],
    job: Optional[str],
    callback: Optional[str],
) -> Tuple[str, str, str, List[str]]:
    """Full pipeline; returns (pdf_path, filename, detected_language, cleanup_paths)."""
    started = time.time()
    progress = ProgressReporter(job, callback)
    progress.engage()

    suffix = validate_url(url)
    progress.emit("download", "Téléchargement de l'audio…", pct=5)
    tmp_path, dl_info = await download_audio(url, suffix)
    cleanup: List[str] = [tmp_path]

    try:
        if dl_info.get("bytes", 0) > MAX_FILE_SIZE:
            raise HTTPException(status_code=413, detail="Fichier audio trop volumineux (max 500 Mo).")

        progress.emit("probe", "Analyse du fichier…", pct=15)
        probe_audio(tmp_path)

        normalized_path, info = await normalize_audio(tmp_path, progress)
        if normalized_path != tmp_path:
            cleanup.append(normalized_path)

        chunk_paths = await chunk_audio(normalized_path, info, progress)
        cleanup.extend([p for p in chunk_paths if p not in cleanup])

        texts = await transcribe_chunks(chunk_paths, progress)
        full_text = "\n\n".join(t for t in texts if t).strip()
        if not full_text:
            raise HTTPException(status_code=422, detail="Aucun contenu vocal détecté dans l'audio.")

        # Language for the summary prompt: client param > API detection (sampled
        # across the first TWO chunks) > last-resort "en".
        detected_language = (lang_hint or "").strip().lower()
        if not detected_language:
            detected_language = detect_language(
                "\n".join(texts[:2])[:1500], fallback="en"
            )

        summary = ""
        if summarize:
            summary = await summarize_transcript(full_text, detected_language, progress)

        progress.emit("pdf", "Génération du PDF…", pct=95)
        cleaned_title = Path(url.split("?", 1)[0]).stem[:80] or "Transcription Audio"
        pdf_path, pdf_name = generate_pdf(full_text, summary, cleaned_title, detected_language)
        cleanup.append(pdf_path)

        progress.done("PDF prêt")
        log.info(
            "job done in %.1fs | lang=%s | summary=%s | %d chars",
            time.time() - started, detected_language, bool(summary), len(full_text),
        )
        return pdf_path, pdf_name, detected_language, cleanup
    except HTTPException as he:
        progress.failed(str(he.detail))
        raise
    except Exception as exc:
        progress.failed("Erreur interne du service audio.")
        log.exception("pipeline failure")
        raise HTTPException(status_code=500, detail="Erreur interne du service audio.") from exc


# ── Endpoints (same public contract as v1) ───────────────────────────────────


@app.get("/health")
async def health() -> Dict[str, Any]:
    ffmpeg_ok = False
    try:
        ffmpeg_ok, _ = run_ffmpeg_sync(["ffmpeg", "-version"], timeout=10)
    except Exception:
        ffmpeg_ok = False
    return {
        "status": "healthy",
        "ffmpeg_available": ffmpeg_ok,
        "has_groq_key": has_groq_key,
        "has_openrouter_key": has_openrouter_key,
        "unicode_font": FONT_CONF.get("regular", {}).get("name"),
    }


@app.get("/audio-to-pdf")
async def audio_to_pdf(
    background_tasks: BackgroundTasks,
    url: str = "",
    summarize: bool = True,
    lang: Optional[str] = None,
    job: Optional[str] = None,
    callback: Optional[str] = None,
):
    pdf_path, pdf_name, _language, cleanup = await process_audio_pipeline(
        url=url,
        summarize=summarize,
        lang_hint=lang,
        job=job,
        callback=callback,
    )
    background_tasks.add_task(cleanup_files, cleanup)
    return FileResponse(path=pdf_path, filename=pdf_name, media_type="application/pdf")


@app.get("/youtube-to-pdf")
async def youtube_to_pdf(
    background_tasks: BackgroundTasks,
    url: str = "",
    summarize: bool = True,
    lang: Optional[str] = None,
    job: Optional[str] = None,
):
    progress = ProgressReporter(job, None)
    progress.engage()
    def _yt_download() -> Tuple[str, Dict[str, Any]]:
        out_dir = tempfile.mkdtemp(prefix="yt_")
        template = str(Path(out_dir) / "yt.%(ext)s")
        cmd = [
            "yt-dlp", "-x", "--audio-format", "mp3", "--audio-quality", "5",
            "--no-playlist", "--no-progress",
            "-o", template, url,
            "--max-filesize", str(MAX_FILE_SIZE),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise HTTPException(status_code=502, detail="Impossible de télécharger la piste YouTube.")
        files = sorted(Path(out_dir).glob("yt.*"))
        if not files:
            raise HTTPException(status_code=502, detail="Aucune piste audio récupérée depuis YouTube.")
        return str(files[0]), {"out_dir": out_dir}

    try:
        tmp_path, meta = await asyncio.to_thread(_yt_download)
    except HTTPException:
        raise
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Téléchargement YouTube trop lent (timeout).") from exc

    try:
        info = probe_audio(tmp_path)
        chunk_paths = await chunk_audio(tmp_path, info, progress)
        texts = await transcribe_chunks(chunk_paths, progress)
        full_text = "\n\n".join(t for t in texts if t).strip()
        if not full_text:
            raise HTTPException(status_code=422, detail="Aucun contenu vocal détecté dans la piste.")

        detected_language = (lang or "").strip().lower() or detect_language(full_text[:1500], fallback="en")
        summary = await summarize_transcript(full_text, detected_language, progress) if summarize else ""
        title = "Transcription YouTube"
        pdf_path, pdf_name = generate_pdf(full_text, summary, title, detected_language)
        progress.done("PDF prêt")

        cleanup = [tmp_path, pdf_path]
        out_dir = meta.get("out_dir")
        if out_dir:
            cleanup.extend(str(p) for p in Path(out_dir).glob("*"))
    except HTTPException as he:
        progress.failed(str(he.detail))
        raise
    except Exception as exc:
        progress.failed("Erreur interne du service audio.")
        log.exception("youtube pipeline failure")
        raise HTTPException(status_code=500, detail="Erreur interne du service audio.") from exc

    background_tasks.add_task(cleanup_files, cleanup)
    return FileResponse(path=pdf_path, filename=pdf_name, media_type="application/pdf")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
