FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# ffmpeg: audio decoding/normalization/segmentation
# fonts-noto-core: Noto Naskh Arabic for Unicode PDF rendering
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg fonts-noto-core ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# NOTE: no torch / openai-whisper / pytube — local whisper was never used
# (transcription runs on Groq), this trims the image by ~1.5 GB.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
