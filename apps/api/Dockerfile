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

# The EfficientNet-B7 face manipulation detector (R3-T2), selected and benchmarked in R3-T1.
# Two artifacts, each pinned to an immutable revision and each refused unless the bytes hash to
# the digest recorded in app/face_detector.py — so the image carries exactly the weights that
# benchmark measured, and the worker never downloads a model while serving a job.
#
# The classifier is Selim Seferbekov's DFDC-winning entry as exported by Facetorch (273 MiB).
# The locator is YuNet, which is what turns a frame into the face crop the classifier was
# trained on (227 KiB); it is preprocessing for this detector rather than a capability of its
# own, so it is pinned and verified exactly as the classifier is.
#
# Fetched by two near-identical steps rather than through a shared helper. They are eight
# lines that differ in two arguments, and the duplication keeps each one readable as the thing
# it is — a URL, a digest and a destination — which is worth more here than removing it.
ARG FACE_CLASSIFIER_URL=https://huggingface.co/tomas-gajarsky/facetorch-deepfake-efficientnet-b7/resolve/4acc494f37eb63d7457166eff2acb45c5b04b9a6/model-torch2.11.pt2
ARG FACE_CLASSIFIER_SHA256=97b49a70174c0d4f72d9d510d817bdc49a907af9af0242a6a1ba934a7cc9e4b7
RUN python -c "\
import hashlib, sys, urllib.request; \
url, expected, target = sys.argv[1], sys.argv[2], '/models/face_manipulation/model-torch2.11.pt2'; \
import os; os.makedirs('/models/face_manipulation', exist_ok=True); \
payload = urllib.request.urlopen(url, timeout=600).read(); \
digest = hashlib.sha256(payload).hexdigest(); \
sys.exit('face classifier checksum mismatch: expected %s, got %s' % (expected, digest)) if digest != expected else None; \
open(target, 'wb').write(payload); \
print('face classifier verified: %s (%d bytes)' % (digest, len(payload)))\
" "$FACE_CLASSIFIER_URL" "$FACE_CLASSIFIER_SHA256"

ARG FACE_LOCATOR_URL=https://media.githubusercontent.com/media/opencv/opencv_zoo/47534e27c9851bb1128ccc0102f1145e27f23f98/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
ARG FACE_LOCATOR_SHA256=8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
RUN python -c "\
import hashlib, sys, urllib.request; \
url, expected, target = sys.argv[1], sys.argv[2], '/models/face_manipulation/face_detection_yunet_2023mar.onnx'; \
payload = urllib.request.urlopen(url, timeout=120).read(); \
digest = hashlib.sha256(payload).hexdigest(); \
sys.exit('face locator checksum mismatch: expected %s, got %s' % (expected, digest)) if digest != expected else None; \
open(target, 'wb').write(payload); \
print('face locator verified: %s (%d bytes)' % (digest, len(payload)))\
" "$FACE_LOCATOR_URL" "$FACE_LOCATOR_SHA256"

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
