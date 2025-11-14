import os
import subprocess
import tempfile
import requests
import shutil
from pathlib import Path
from urllib.parse import urlparse
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from openai import OpenAI
from groq import Groq

# ------------------------------
# Configuration
# ------------------------------
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# Limits
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
CHUNK_LENGTH = 600  # 10 minutes (Groq has 25MB file size limit per request)
MAX_TEXT_BEFORE_SUMMARY = 4000
ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.ogg', '.flac'}
ALLOWED_DOMAINS = None  # Set to list of domains to restrict, e.g., ['example.com']

app = FastAPI(title="Audio to PDF API")

# ------------------------------
# CORS Configuration
# ------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your actual frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Helper Functions
# ------------------------------

class AudioProcessingError(Exception):
    """Custom exception for audio processing errors"""
    pass


def validate_url(url: str) -> bool:
    """Validate URL safety"""
    try:
        parsed = urlparse(url)
        if parsed.scheme not in ['http', 'https']:
            return False
        if ALLOWED_DOMAINS and parsed.netloc not in ALLOWED_DOMAINS:
            return False
        # Check file extension
        ext = Path(parsed.path).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            return False
        return True
    except Exception:
        return False


def download_audio(url: str) -> str:
    """Download audio file with size limits and validation"""
    if not validate_url(url):
        raise AudioProcessingError("Invalid or unsupported URL")
    
    ext = Path(urlparse(url).path).suffix or ".mp3"
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
    
    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            
            # Check content length
            content_length = r.headers.get('content-length')
            if content_length and int(content_length) > MAX_FILE_SIZE:
                raise AudioProcessingError(f"File too large (max {MAX_FILE_SIZE/1024/1024}MB)")
            
            downloaded = 0
            for chunk in r.iter_content(chunk_size=1024*1024):
                downloaded += len(chunk)
                if downloaded > MAX_FILE_SIZE:
                    raise AudioProcessingError("File size exceeded during download")
                tmp_file.write(chunk)
        
        tmp_file.close()
        return tmp_file.name
    
    except requests.RequestException as e:
        tmp_file.close()
        os.unlink(tmp_file.name)
        raise AudioProcessingError(f"Download failed: {str(e)}")


def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.DEVNULL, 
                      stderr=subprocess.DEVNULL, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def split_audio(input_file: str, chunk_length=CHUNK_LENGTH) -> list:
    """Split audio into chunks with error handling"""
    if not check_ffmpeg():
        raise AudioProcessingError("FFmpeg not installed or not in PATH")
    
    ext = Path(input_file).suffix
    output_dir = tempfile.mkdtemp()
    output_pattern = str(Path(output_dir) / f"part_%03d{ext}")
    
    try:
        result = subprocess.run([
            "ffmpeg", "-i", input_file,
            "-f", "segment", 
            "-segment_time", str(chunk_length),
            "-c", "copy", 
            output_pattern
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode != 0:
            raise AudioProcessingError(f"FFmpeg error: {result.stderr}")
        
        chunks = sorted([
            str(Path(output_dir) / f) 
            for f in os.listdir(output_dir)
        ])
        
        if not chunks:
            raise AudioProcessingError("No chunks created")
        
        return chunks
    
    except subprocess.TimeoutExpired:
        raise AudioProcessingError("Audio splitting timed out")


def transcribe_with_groq(audio_file: str) -> str:
    """Transcribe audio using Groq's Whisper API (ultra-fast)"""
    try:
        # Check file size (Groq has 25MB limit per file)
        file_size = os.path.getsize(audio_file)
        if file_size > 25 * 1024 * 1024:
            raise AudioProcessingError(f"Chunk too large for Groq API (max 25MB, got {file_size/1024/1024:.1f}MB)")
        
        with open(audio_file, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(Path(audio_file).name, file.read()),
                model="whisper-large-v3",
                response_format="text",
                temperature=0.0  # Deterministic output
            )
        
        return transcription.strip()
    
    except Exception as e:
        raise AudioProcessingError(f"Groq transcription failed: {str(e)}")


def summarize_text(text: str) -> str:
    """Summarize text using GPT-4o-mini with error handling"""
    if not text.strip():
        return ""
    
    prompt = (
        "Summarize this transcript clearly and concisely, Keep it professional, keep same language of audio, 60% of times it frensh, 40% english,  "
        "preserving key technical/medical terms and main points:\n\n"
        f"{text}"
    )
    
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        print(f"⚠️ Summarization error: {e}")
        # Return truncated text as fallback
        return text[:MAX_TEXT_BEFORE_SUMMARY] + "...[truncated]"


def generate_pdf(text: str, output_path: str = None) -> str:
    """Generate PDF with better text handling"""
    if output_path is None:
        output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name
    
    c = canvas.Canvas(output_path, pagesize=A4)
    width, height = A4
    margin = 50
    y_position = height - margin
    line_height = 14
    
    c.setFont("Helvetica", 11)
    
    for line in text.split("\n"):
        # Simple word wrapping
        words = line.split()
        current_line = ""
        
        for word in words:
            test_line = f"{current_line} {word}".strip()
            if c.stringWidth(test_line, "Helvetica", 11) < (width - 2 * margin):
                current_line = test_line
            else:
                # Draw current line and start new one
                if y_position < margin + line_height:
                    c.showPage()
                    c.setFont("Helvetica", 11)
                    y_position = height - margin
                c.drawString(margin, y_position, current_line)
                y_position -= line_height
                current_line = word
        
        # Draw remaining text
        if current_line:
            if y_position < margin + line_height:
                c.showPage()
                c.setFont("Helvetica", 11)
                y_position = height - margin
            c.drawString(margin, y_position, current_line)
            y_position -= line_height
        
        # Add spacing between paragraphs
        y_position -= line_height / 2
    
    c.save()
    return output_path


def cleanup_files(*file_paths):
    """Safely cleanup temporary files and directories"""
    for path in file_paths:
        if not path:
            continue
        try:
            path_obj = Path(path)
            if path_obj.is_file():
                path_obj.unlink(missing_ok=True)
            elif path_obj.is_dir():
                shutil.rmtree(path_obj, ignore_errors=True)
        except Exception as e:
            print(f"⚠️ Cleanup warning for {path}: {e}")


# ------------------------------
# API Endpoints
# ------------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    groq_available = bool(os.environ.get("GROQ_API_KEY"))
    return {
        "status": "healthy",
        "ffmpeg_available": check_ffmpeg(),
        "transcription_engine": "Groq Whisper-Large-v3",
        "groq_configured": groq_available
    }


@app.get("/audio-to-pdf")
async def audio_to_pdf(
    background_tasks: BackgroundTasks,
    url: str = Query(..., description="URL to audio file"),
    summarize: bool = Query(True, description="Enable automatic summarization")
):
    """
    Convert audio file to PDF transcript using Groq's ultra-fast Whisper API
    
    - **url**: Direct URL to audio file (mp3, wav, m4a, ogg, flac)
    - **summarize**: Enable automatic text summarization for long content
    
    Note: Groq has a 25MB per-file limit, so long audio is automatically split into chunks
    """
    audio_file = None
    chunk_files = []
    pdf_path = None
    chunk_dir = None
    
    try:
        # Validate Groq API key
        if not os.environ.get("GROQ_API_KEY"):
            raise AudioProcessingError("GROQ_API_KEY not configured")
        
        # Download audio
        print(f"📥 Downloading audio from {url[:50]}...")
        audio_file = download_audio(url)
        
        # Split into chunks (Groq has 25MB limit)
        print(f"✂️ Splitting audio...")
        chunk_files = split_audio(audio_file)
        chunk_dir = Path(chunk_files[0]).parent if chunk_files else None
        
        # Process chunks with Groq
        full_text = ""
        
        for i, chunk in enumerate(chunk_files):
            print(f"🎧 Transcribing chunk {i+1}/{len(chunk_files)} with Groq...")
            
            # Transcribe with Groq (ultra-fast)
            chunk_text = transcribe_with_groq(chunk)
            print(f"📝 chunk {i+1 } : {chunk_text}  ")
            
            # Summarize individual chunk if too long
            if summarize and len(chunk_text) > MAX_TEXT_BEFORE_SUMMARY:
                print(f"📝 Summarizing chunk {i+1}...")
                chunk_text = summarize_text(chunk_text)
            
            full_text += chunk_text + "\n\n"
        
        if not full_text.strip():
            raise AudioProcessingError("No transcription generated")
        
        # Generate PDF
        print(f"📄 Generating PDF...")
        pdf_path = generate_pdf(full_text)
        
        # Schedule cleanup after response
        background_tasks.add_task(
            cleanup_files, 
            audio_file, 
            pdf_path,
            chunk_dir
        )
        
        return FileResponse(
            pdf_path,
            filename="transcript.pdf",
            media_type="application/pdf",
            headers={
                "Content-Disposition": "attachment; filename=transcript.pdf"
            }
        )
    
    except AudioProcessingError as e:
        cleanup_files(audio_file, pdf_path, chunk_dir)
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        cleanup_files(audio_file, pdf_path, chunk_dir)
        print(f"❌ Unexpected error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Processing failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)