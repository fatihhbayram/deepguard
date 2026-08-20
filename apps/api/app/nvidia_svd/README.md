# Vendored NVIDIA Synthetic Video Detector interface

This package contains third-party interface definitions, not DeepGuard logic.

## Source

`syntheticvideodetector.proto` is copied verbatim from NVIDIA's official Maxine NIM
sample client repository:

- Repository: <https://github.com/NVIDIA-Maxine/nim-clients>
- Path: `synthetic-video-detector/protos/proto/nvidia/maxine/syntheticvideodetector/v1/syntheticvideodetector.proto`
- License: MIT (SPDX-License-Identifier is retained in the file header)
- Copyright: NVIDIA CORPORATION & AFFILIATES

Only the `.proto` is vendored. None of NVIDIA's sample client code, scripts or assets
are copied into this repository.

## Generated modules

`syntheticvideodetector_pb2.py`, `syntheticvideodetector_pb2.pyi` and
`syntheticvideodetector_pb2_grpc.py` are generated from that `.proto` and are checked in
so that neither the runtime image nor CI needs a protoc toolchain.

Regenerate from `apps/api` after replacing the `.proto`:

```sh
pip install grpcio-tools==1.76.0
python -m grpc_tools.protoc -I . --python_out=. --pyi_out=. --grpc_python_out=. \
    app/nvidia_svd/syntheticvideodetector.proto
```

The proto is deliberately kept at this path inside the package: `protoc` derives the
generated cross-import from the file path, so this layout produces
`from app.nvidia_svd import syntheticvideodetector_pb2`, which resolves under the normal
`apps/api` import root without any post-generation patching.

The protobuf package name (`nvidia.maxine.syntheticvideodetector.v1`) is independent of
that path, so the gRPC method stays
`/nvidia.maxine.syntheticvideodetector.v1.SyntheticVideoDetectorService/DetectSyntheticVideo`
on the wire.
