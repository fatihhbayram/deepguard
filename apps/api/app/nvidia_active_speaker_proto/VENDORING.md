# Vendored NVIDIA Active Speaker Detection protos

Source: <https://github.com/NVIDIA-Maxine/nim-clients>, `active-speaker-detection/protos/proto/`
(MIT, `SPDX-License-Identifier: MIT`). Only the `.proto` files are vendored — none of
NVIDIA's sample client code is copied.

| Vendored file                | Upstream path                                                  |
| ---------------------------- | -------------------------------------------------------------- |
| `common.proto`               | `nvidia/ai4m/common/v1/common.proto`                             |
| `audio.proto`                | `nvidia/ai4m/audio/v1/audio.proto`                               |
| `video.proto`                | `nvidia/ai4m/video/v1/video.proto`                               |
| `activespeakerdetection.proto` | `nvidia/ai4m/activespeakerdetection/v1/activespeakerdetection.proto` |

## The one modification

The files are otherwise byte-identical to upstream. Only the three cross-proto `import`
lines in `activespeakerdetection.proto` were repointed at this directory:

```text
import "nvidia/ai4m/audio/v1/audio.proto";   →  import "app/nvidia_active_speaker_proto/audio.proto";
import "nvidia/ai4m/common/v1/common.proto"; →  import "app/nvidia_active_speaker_proto/common.proto";
import "nvidia/ai4m/video/v1/video.proto";   →  import "app/nvidia_active_speaker_proto/video.proto";
```

Upstream's paths would make protoc emit `from nvidia.ai4m.audio.v1 import audio_pb2`,
which only resolves if `nvidia` is a top-level package on `sys.path` — NVIDIA's sample
client arranges that with `sys.path.insert`. Repointing the imports makes protoc emit
`from app.nvidia_active_speaker_proto import audio_pb2` instead, so the stubs import
cleanly from inside the `app` package with no path manipulation, matching D015's approach
for `app/nvidia_svd/`.

Every `package` declaration is untouched, so this changes nothing on the wire. The RPC the
client dials is still

```text
/nvidia.ai4m.activespeakerdetection.v1.ActiveSpeakerDetectionService/DetectActiveSpeaker
```

byte for byte identical to the official client's, and so are all message encodings. Only
the descriptors' own file names — a client-local detail — differ.

## Regenerating the stubs

Generated modules are committed (D015), so no protoc toolchain is needed at runtime or in
CI. To regenerate, from `apps/api` with `grpcio-tools` installed:

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. \
    app/nvidia_active_speaker_proto/activespeakerdetection.proto
python -m grpc_tools.protoc -I. --python_out=. \
    app/nvidia_active_speaker_proto/common.proto \
    app/nvidia_active_speaker_proto/audio.proto \
    app/nvidia_active_speaker_proto/video.proto
```

`common`, `audio` and `video` declare no services, so they get `--python_out` only; their
`_pb2_grpc.py` files would be empty stubs.
