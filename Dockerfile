FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ffmpeg: audio decoding/normalization/segmentation
# fonts-noto-core: Noto Naskh Arabic (Arabic-shaped fallback)
# fonts-dejavu-core: DejaVu Sans — full Latin + Arabic in ONE font (mixed
#   French/Arabic summaries render without blank-dots)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg fonts-noto-core fonts-dejavu-core ca-certificates curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Deno JS runtime — yt-dlp's YouTube n-challenge solver probes deno by
# default; without it only storyboard images are extractable on logged-in
# @SBI-protected pages.
RUN curl -fsSL https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip \
        -o /tmp/deno.zip && \
    python -c "import zipfile; zipfile.ZipFile('/tmp/deno.zip').extract('deno', '/usr/local/bin/')" && \
    chmod +x /usr/local/bin/deno && \
    rm -f /tmp/deno.zip && \
    /usr/local/bin/deno --version

COPY requirements.txt .

# NOTE: no torch / openai-whisper / pytube — local whisper was never used
# (transcription runs on Groq), this trims the image by ~1.5 GB.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
