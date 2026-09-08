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

# The landmark stack LipForensics aligns its mouth crops with (R5-T2), installed on its own
# line because it is the one requirement here whose declared dependencies must not be
# resolved. `face-alignment` asks for `opencv-python` and `dlib`; the first would install a
# second, GUI-linked OpenCV beside the headless wheel this image already carries, and the
# second is for a detector this codebase never selects. What it genuinely imports is pinned
# in requirements.txt above, so `--no-deps` leaves nothing missing — the comment there lists
# both halves.
RUN pip install --no-cache-dir --no-deps face-alignment==1.5.0

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

# The LipForensics lip-sync detector (R5-T2), selected and benchmarked in R5-T1. Eight
# artifacts in four roles, every one pinned to an immutable origin and refused unless its bytes
# hash to the digest recorded in app/lip_forensics.py — so the image carries exactly what that
# benchmark measured, and the worker never downloads a model while serving a job.
#
# One step with a table rather than eight near-identical steps, unlike the two detectors above.
# The difference is that these artifacts are not eight independent fetches: they are one
# detector's pinned checkout, and the module that loads them holds the same table for the same
# reason. Writing it out eight times would obscure that they stand or fall together.
#
# The four roles:
#
# - the forgery classifier Haliassos et al. trained on FaceForensics++ (137 MiB), hosted on
#   Google Drive because that is where the upstream README links it. Drive serves an HTML
#   interstitial before a file this size, so the confirmation token is read out of that page
#   and replayed — which is what the cookie jar and the `uuid` below are for, and the only
#   reason this fetch is not a plain URL like the others;
# - the four upstream source files the wrapper *executes* — the network definition and its
#   config. They are fetched at the pinned commit rather than cloned, because a commit sha on
#   raw.githubusercontent is as immutable as a clone and does not put git in this image;
# - the LRW mean face every frame is aligned onto, from the same commit;
# - the S3FD detector and 2D-FAN landmark model `face-alignment` reads, laid out as a torch hub
#   directory because that is how that library finds them.
#
# The layout under /models/lip_forensics is R5-T1's cache layout exactly, so
# DEEPGUARD_LIPFORENSICS_MODEL_DIR can be pointed at a developer's benchmark cache unchanged.
RUN python - <<'FETCH'
import hashlib, http.cookiejar, os, re, urllib.request

REVISION = "d0bf5553bfb9676f1771d590472b26a3a76de894"
ROOT = "/models/lip_forensics"
SOURCE = f"{ROOT}/lipforensics-{REVISION}"
UPSTREAM = f"https://raw.githubusercontent.com/ahaliassos/LipForensics/{REVISION}"
DRIVE_ID = "1wfZnxZpyNd5ouJs0LjVls7zU0N_W73L7"

artifacts = [
    (f"{UPSTREAM}/models/resnet.py",
     f"{SOURCE}/models/resnet.py",
     "1d159120f159636b88291702e3e62ea5b43a8716687fb261308bc5092e3c122c"),
    (f"{UPSTREAM}/models/tcn.py",
     f"{SOURCE}/models/tcn.py",
     "35f57b7b65c976ed77857f9784a94baaa15bd0bbdc94033ec2b3ef48e00c1712"),
    (f"{UPSTREAM}/models/spatiotemporal_net.py",
     f"{SOURCE}/models/spatiotemporal_net.py",
     "eabfabbb06aa411796b10f9b1de0231bfee719fbc45ff0ab0c40f20820021d29"),
    (f"{UPSTREAM}/models/configs/lrw_resnet18_mstcn.json",
     f"{SOURCE}/models/configs/lrw_resnet18_mstcn.json",
     "dc9135351ee1b8458285b3bc5b8c42877f10a6403502437d312b0db81af678c9"),
    (f"{UPSTREAM}/preprocessing/20words_mean_face.npy",
     f"{SOURCE}/preprocessing/20words_mean_face.npy",
     "dbf68b2044171e1160716df7c53e8bbfaa0ee8c61fb41171d04cb6092bb81422"),
    ("https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
     f"{ROOT}/face-alignment/checkpoints/s3fd-619a316812.pth",
     "619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543"),
    ("https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar",
     f"{ROOT}/face-alignment/checkpoints/2DFAN4-11f355bf06.pth.tar",
     "11f355bf0693120222f5955ce3f9dc8fb5763ebb30a47d7906e509490d32e4aa"),
]

opener = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
)


def store(payload, target, expected):
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected:
        raise SystemExit(
            f"LipForensics checksum mismatch for {target}: "
            f"expected {expected}, got {digest}"
        )
    os.makedirs(os.path.dirname(target), exist_ok=True)
    with open(target, "wb") as handle:
        handle.write(payload)
    print(f"LipForensics artifact verified: {target} ({len(payload)} bytes)")


for url, target, expected in artifacts:
    store(opener.open(url, timeout=600).read(), target, expected)

# Drive's interstitial, replayed. The token is single-use and bound to the cookie the page
# above set, so the two requests have to share this opener.
page = opener.open(
    f"https://drive.usercontent.google.com/download?id={DRIVE_ID}&export=download",
    timeout=120,
).read().decode("utf-8", "replace")
token = re.search(r'name="uuid" value="([^"]+)"', page)
if token is None:
    raise SystemExit("Google Drive did not offer a download confirmation for the weights")
store(
    opener.open(
        f"https://drive.usercontent.google.com/download?id={DRIVE_ID}"
        f"&export=download&confirm=t&uuid={token.group(1)}",
        timeout=600,
    ).read(),
    f"{ROOT}/weights/lipforensics_ff.pth",
    "4b7790bc8e02d0c25ecfa0d8d6a2907123c2206cc32e2bad6044e50f013c253d",
)
FETCH

# The Effort face-forgery detector (R7-T11), measured in R7-T7, R7-T8 and R7-T9 and integrated as
# evidence-only by R7-T10's conclusion. Four artifacts in three roles, every one pinned to an
# immutable origin and refused unless its bytes hash to the digest recorded in app/effort.py — so
# the image carries exactly what those three studies measured, and the worker never downloads a
# model while serving a job.
#
# The three roles:
#
# - the forgery checkpoint Yan et al. trained on FaceForensics++ (1.13 GiB), hosted on Google
#   Drive because that is where the upstream README links it. Drive serves an HTML interstitial
#   before a file this size, so the confirmation token is read out of that page and replayed —
#   the same dance the LipForensics weights above need, and the only reason this fetch is not a
#   plain URL;
# - the dlib 81-landmark model every face alignment is built on, fetched from the upstream
#   repository at the pinned commit. Upstream's README requires this exact file for this
#   checkpoint;
# - the CLIP-L14 vision tower the checkpoint's weights are shaped against (1.59 GiB), from the
#   pinned HuggingFace revision. Vendored as the two files `CLIPModel.from_pretrained` reads so
#   that nothing is fetched at analysis time — app/effort.py loads this directory with
#   `local_files_only=True`.
#
# This is by far the largest addition to this image, and it is weights rather than code: about
# 2.7 GiB across the checkpoint and the backbone. That is the cost of running the detector the
# studies measured instead of a smaller one nobody measured.
RUN python - <<'FETCH'
import hashlib, http.cookiejar, os, re, urllib.request

ROOT = "/models/effort"
EFFORT_REVISION = "96f5dea2b534d400cfd7003f053c7e93c8e16461"
CLIP_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
DRIVE_ID = "1m4fyJecABU-Yk3bJ4b1WhUwQa0xCkMLI"
UPSTREAM = f"https://raw.githubusercontent.com/YZY-stack/Effort-AIGI-Detection/{EFFORT_REVISION}"
CLIP = f"https://huggingface.co/openai/clip-vit-large-patch14/resolve/{CLIP_REVISION}"

artifacts = [
    (f"{UPSTREAM}/DeepfakeBench/preprocessing/shape_predictor_81_face_landmarks.dat",
     f"{ROOT}/shape_predictor_81_face_landmarks.dat",
     "8cae4375589dd915d9a0a881101bed1bbb4e9887e35e63b024388f1ca25ff869"),
    (f"{CLIP}/config.json",
     f"{ROOT}/clip-vit-large-patch14/config.json",
     "8a09b467700c58138c29d53c605b34ebc69beaadd13274a8a2af8ad2c2f4032a"),
    (f"{CLIP}/model.safetensors",
     f"{ROOT}/clip-vit-large-patch14/model.safetensors",
     "a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb"),
]

opener = urllib.request.build_opener(
    urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
)


def store(response, target, expected):
    """Stream to disk while hashing, and keep the file only if it is the pinned one.

    Streamed rather than read whole, unlike the smaller artifacts fetched above: the checkpoint
    is 1.13 GiB and buffering it in the build container to hash it is a gigabyte of resident
    memory spent to avoid a temporary file.
    """
    os.makedirs(os.path.dirname(target), exist_ok=True)
    digest = hashlib.sha256()
    written = 0
    with open(target, "wb") as handle:
        for block in iter(lambda: response.read(1024 * 1024), b""):
            digest.update(block)
            handle.write(block)
            written += len(block)
    if digest.hexdigest() != expected:
        os.unlink(target)
        raise SystemExit(
            f"Effort checksum mismatch for {target}: "
            f"expected {expected}, got {digest.hexdigest()}"
        )
    print(f"Effort artifact verified: {target} ({written} bytes)")


for url, target, expected in artifacts:
    store(opener.open(url, timeout=900), target, expected)

# Drive's interstitial, replayed. The token is single-use and bound to the cookie the page above
# set, so the two requests have to share this opener.
page = opener.open(
    f"https://drive.usercontent.google.com/download?id={DRIVE_ID}&export=download",
    timeout=120,
).read().decode("utf-8", "replace")
token = re.search(r'name="uuid" value="([^"]+)"', page)
if token is None:
    raise SystemExit("Google Drive did not offer a download confirmation for the checkpoint")
store(
    opener.open(
        f"https://drive.usercontent.google.com/download?id={DRIVE_ID}"
        f"&export=download&confirm=t&uuid={token.group(1)}",
        timeout=1800,
    ),
    f"{ROOT}/effort_clip_L14_trainOn_FaceForensic.pth",
    "8d86711f098d16b49c048962fc3e16a906380f7bb17b8a0e89bd545b926943ee",
)
FETCH

COPY . .

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
