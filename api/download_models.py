"""Bakes the two production models into the Docker image (run at build time)."""
import os

from huggingface_hub import snapshot_download

MODELS = ["google/siglip2-base-patch16-224", "sentence-transformers/all-MiniLM-L6-v2"]
# Only what transformers / sentence-transformers load; skips the duplicate
# PyTorch / ONNX / TF weight files some repos also carry.
SKIP = ["*.bin", "*.h5", "*.msgpack", "*.ot", "onnx/*", "openvino/*", "*.onnx"]

for repo in MODELS:
    path = snapshot_download(repo, ignore_patterns=SKIP)
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(path) for f in fs) / 1e6
    print(f"{repo}: {size:.0f} MB")
