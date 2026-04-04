FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

WORKDIR /app

# System deps for soundfile / audio processing + espeak-ng for VieNeu TTS
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libsndfile1 espeak-ng libespeak-ng1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY universal_voice/ universal_voice/

RUN mkdir -p /app/data/models

EXPOSE 14213

CMD ["python", "-m", "universal_voice.main"]
