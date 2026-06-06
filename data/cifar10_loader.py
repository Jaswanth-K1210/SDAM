"""CIFAR-10 real feature extractor.

Extracts 512-d ResNet-18 (ImageNet-pretrained) features for all 10 CIFAR-10
classes, L2-normalizes them, and caches the result to disk. This is the primary
real-data source for the S-DAM experiments (replacing the synthetic fallback).
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# CIFAR class names in canonical label order.
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def load_cifar10_features(
    cache_path: str = "data/cifar10_feats.pt",
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Return {class_name: Tensor(~5000, 512)} of L2-normalized ResNet-18 features.

    Extracts features on first run (downloading CIFAR-10 and the pretrained
    ResNet-18) and caches them to ``cache_path``; subsequent runs load the cache.
    """
    if os.path.exists(cache_path):
        print(f"Loading cached features from {cache_path}")
        return torch.load(cache_path, map_location="cpu")

    print("Extracting CIFAR-10 features...")

    import torchvision
    from torchvision import transforms
    from torchvision.models import ResNet18_Weights, resnet18

    # Auto-upgrade to GPU when available — extraction is the slow step.
    if device == "cpu" and torch.cuda.is_available():
        device = "cuda"

    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()  # -> 512-d penultimate features
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    dataset = torchvision.datasets.CIFAR10(
        root="data/cifar10", train=True, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=512, num_workers=2, shuffle=False
    )

    feats_by_class: dict[int, list[torch.Tensor]] = {i: [] for i in range(10)}
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            out = model(images)               # (B, 512)
            out = F.normalize(out, dim=1)      # L2-normalize
            out = out.cpu()
            for feat, lbl in zip(out, labels):
                feats_by_class[int(lbl)].append(feat)

    feats = {
        CIFAR10_CLASSES[i]: torch.stack(feats_by_class[i]) for i in range(10)
    }

    torch.save(feats, cache_path)
    print(f"Saved features to {cache_path}")
    return feats


if __name__ == "__main__":
    feats = load_cifar10_features()
    print("Categories:", list(feats.keys()))
    print("Feature dim:", next(iter(feats.values())).shape[1])
    for name, t in feats.items():
        print(f"  {name:12s}: {tuple(t.shape)}")
