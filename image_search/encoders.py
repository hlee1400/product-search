"""
Embedding models for image search.

Two families:
  shared  - one model embeds both images and text into the same space
            (CLIP, SigLIP, SigLIP 2), so text can search images directly.
  single  - a model that only does one modality: DINOv2 for images,
            all-MiniLM-L6-v2 for text (the same model search_app uses).

Every encoder returns L2-normalized float32 vectors, so a dot product is
cosine similarity.
"""
from __future__ import annotations

import os
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# key -> (hf model id, kind, display name)
MODELS = {
    "clip-b32":    ("openai/clip-vit-base-patch32",     "shared", "CLIP ViT-B/32"),
    "siglip-b16":  ("google/siglip-base-patch16-224",   "shared", "SigLIP B/16"),
    "siglip2-b16": ("google/siglip2-base-patch16-224",  "shared", "SigLIP 2 B/16"),
    "dinov2-b":    ("facebook/dinov2-base",             "image",  "DINOv2 B/14"),
    "minilm":      ("sentence-transformers/all-MiniLM-L6-v2", "text", "MiniLM-L6"),
}
SHARED_MODELS = [k for k, v in MODELS.items() if v[1] == "shared"]
IMAGE_MODELS = [k for k, v in MODELS.items() if v[1] == "image"]
TEXT_MODELS = [k for k, v in MODELS.items() if v[1] == "text"]


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def _features(out) -> torch.Tensor:
    # get_*_features returns a tensor in older transformers and a
    # ModelOutput with pooler_output in newer ones.
    return out if isinstance(out, torch.Tensor) else out.pooler_output


def _normalize(x: torch.Tensor) -> np.ndarray:
    x = torch.nn.functional.normalize(x.float(), dim=-1)
    return x.cpu().numpy().astype(np.float32)


def _quantize(model: torch.nn.Module) -> torch.nn.Module:
    """int8 weights for every Linear layer (CPU only). ~4x smaller and usually faster on
    CPU; the vectors in the index stay full precision. See api/README.md for the
    accuracy check."""
    from torch.ao.quantization import quantize_dynamic
    return quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)


class Encoder:
    def __init__(self, key: str, quantize: bool = False) -> None:
        self.key = key
        self.model_id, self.kind, self.label = MODELS[key]
        self.device = "cpu" if quantize else device()
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32

        if self.kind == "text":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_id, device=self.device)
            if quantize:
                self.model = _quantize(self.model)
            return

        from transformers import AutoModel, AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id, dtype=self.dtype).to(self.device).eval()
        if quantize:
            self.model = _quantize(self.model)
        # SigLIP was trained on text padded to a fixed 64 tokens and needs
        # the same padding at inference; CLIP uses ordinary padding.
        self.text_kwargs = (
            {"padding": "max_length", "max_length": 64}
            if "siglip" in self.model_id else {"padding": True}
        )
        # SigLIP 2 was trained on lowercased captions.
        self.lowercase = "siglip2" in self.model_id

    @property
    def does_text(self) -> bool:
        return self.kind in ("shared", "text")

    @property
    def does_images(self) -> bool:
        return self.kind in ("shared", "image")

    @torch.inference_mode()
    def encode_text(self, texts: list[str], batch_size: int = 256, progress: bool = False) -> np.ndarray:
        if self.kind == "text":
            return self.model.encode(
                texts, batch_size=batch_size, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=progress,
            ).astype(np.float32)
        if self.kind != "shared":
            raise ValueError(f"{self.label} does not embed text")
        if self.lowercase:
            texts = [t.lower() for t in texts]

        out = []
        for i in tqdm(range(0, len(texts), batch_size), disable=not progress, desc=f"{self.key} text"):
            batch = self.processor(
                text=texts[i:i + batch_size], truncation=True, return_tensors="pt", **self.text_kwargs
            ).to(self.device)
            out.append(_normalize(_features(self.model.get_text_features(**batch))))
        return np.concatenate(out)

    @torch.inference_mode()
    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        if not self.does_images:
            raise ValueError(f"{self.label} does not embed images")
        pixels = self.processor(images=images, return_tensors="pt")["pixel_values"]
        pixels = pixels.to(self.device, self.dtype)
        if self.kind == "shared":
            return _normalize(_features(self.model.get_image_features(pixel_values=pixels)))
        # DINOv2: the CLS token is the global image descriptor.
        return _normalize(self.model(pixel_values=pixels).last_hidden_state[:, 0])

    def encode_image_files(
        self, paths: list[str | Path], batch_size: int = 128, progress: bool = False,
        transform=None,
    ) -> np.ndarray:
        """Embed images from disk, decoding the next batch while the GPU runs."""
        def load(p):
            im = load_image(p)
            return transform(im, p) if transform else im

        batches = [paths[i:i + batch_size] for i in range(0, len(paths), batch_size)]
        out = []
        with ThreadPoolExecutor(max_workers=8) as pool:
            # Submit decoding for batch i+1 before running batch i on the GPU.
            pending = [pool.submit(load, p) for p in batches[0]] if batches else []
            for i in tqdm(range(len(batches)), disable=not progress, desc=f"{self.key} images"):
                imgs = [f.result() for f in pending]
                if i + 1 < len(batches):
                    pending = [pool.submit(load, p) for p in batches[i + 1]]
                out.append(self.encode_images(imgs))
        return np.concatenate(out)
