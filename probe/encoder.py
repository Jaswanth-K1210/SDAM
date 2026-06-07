"""DINOv2 ViT-S/14 encoder wrapper (Task 4).

Self-supervised (NOT post-ReLU-clamped like ImageNet-supervised ViT), so it
should not collapse CLEVR features into the cosine~0.68 cone. torch is imported
lazily so the rest of the probe stays importable/testable without torch.
"""
from __future__ import annotations

import numpy as np


def get_dinov2_encoder(device: str = "cuda"):
    """Frozen DINOv2 ViT-S/14 from torch.hub, eval mode, no grad. Output dim 384."""
    import torch

    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract_features(model, image_paths: list[str], device: str, batch_size: int = 64) -> np.ndarray:
    """Load images -> 224x224 (multiple of 14) -> ImageNet-normalize -> DINOv2 CLS.

    Returns (n, 384) float32. L2-normalization is NOT applied here — the probe
    decides per-metric whether to center/normalize.
    """
    import torch
    import torchvision.transforms as T
    from PIL import Image

    tfm = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    feats = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start:start + batch_size]
        imgs = torch.stack([tfm(Image.open(p).convert("RGB")) for p in batch_paths]).to(device)
        with torch.no_grad():
            out = model(imgs)               # DINOv2 returns the CLS token (B, 384)
        feats.append(out.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(feats, axis=0)
