FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ffprobe validates uploaded media; ffmpeg is the same Debian package.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The AASIST audio anti-spoofing checkpoint (P6-T2, D028). Fetched once at build time from a
# pinned Hugging Face *revision* rather than `main`, and the build fails if the bytes do not
# hash to the digest recorded in app/audio_detector.py — so the image carries one exact
# checkpoint and the worker never downloads a model while serving a job. 1.6 MB.
ARG AASIST_MODEL_URL=https://huggingface.co/SpeechAntiSpoofingBenchmarks/AASIST/resolve/16774d458d86d2a021ae31646c1bf66a5331b53e/aasist.onnx
ARG AASIST_MODEL_SHA256=130e536266b7c537f9a13029e1612a9f392fd1cc827783683b6d1c062a3db5e1
RUN python -c "\
import hashlib, sys, urllib.request; \
url, expected, target = sys.argv[1], sys.argv[2], '/models/aasist.onnx'; \
import os; os.makedirs('/models', exist_ok=True); \
payload = urllib.request.urlopen(url, timeout=120).read(); \
digest = hashlib.sha256(payload).hexdigest(); \
sys.exit('AASIST checksum mismatch: expected %s, got %s' % (expected, digest)) if digest != expected else None; \
open(target, 'wb').write(payload); \
print('AASIST checkpoint verified: %s (%d bytes)' % (digest, len(payload)))\
" "$AASIST_MODEL_URL" "$AASIST_MODEL_SHA256"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
