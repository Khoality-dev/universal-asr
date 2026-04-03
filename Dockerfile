FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# System deps for soundfile / audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY universal_asr/ universal_asr/

RUN mkdir -p /app/data/models

EXPOSE 9000

CMD ["python", "-m", "universal_asr.main"]
