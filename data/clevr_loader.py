"""CLEVR feature loading and the synthetic fallback.

Features are ViT-S/16 embeddings (frozen) projected to ``seed_dim`` and labeled
by Spelke category. If CLEVR is not present, experiments automatically fall
back to ``synthetic_features`` with a clear warning -- experiments must never
crash on missing data.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import torch

from sdam.seeds import SPELKE_SYSTEMS, canonical_spelke_directions

# Each lambda decides whether a CLEVR question dict belongs to a Spelke system.
CATEGORY_FILTERS = {
    "OBJECTNESS": lambda q: str(q.get("answer")) == "1",
    "NUMEROSITY": lambda q: str(q.get("answer")) in {"3", "4", "5", "6"},
    "GEOMETRY": lambda q: ("left of" in q.get("question", "").lower())
    or ("right of" in q.get("question", "").lower()),
    "AGENTNESS": lambda q: "cylinder" in q.get("question", "").lower(),
}


def get_vit_encoder(device):
    """Load a frozen ViT-S/16 from timm plus a fixed random 384->128 projection.

    Returns ``(vit, proj)``. Both are frozen and in eval mode.
    """
    import timm

    vit = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
    vit.eval().to(device)
    for p in vit.parameters():
        p.requires_grad_(False)

    gen = torch.Generator().manual_seed(42)
    proj = torch.nn.Linear(384, 128, bias=False)
    with torch.no_grad():
        proj.weight.copy_(torch.randn(128, 384, generator=gen))
    proj.eval().to(device)
    for p in proj.parameters():
        p.requires_grad_(False)
    return vit, proj


def _feature_dir(clevr_root: str) -> str:
    return os.path.join(clevr_root, "features")


def extract_clevr_features(clevr_root: str, n_per_category: int = 500, device="cpu"):
    """Extract and cache ViT features labeled by Spelke category.

    Caches to ``<clevr_root>/features/<CATEGORY>.pt``. Checks the cache first.
    Returns a dict {category_name: (n, 128) tensor}.
    """
    feat_dir = _feature_dir(clevr_root)
    os.makedirs(feat_dir, exist_ok=True)

    cached = {}
    missing = []
    for name in SPELKE_SYSTEMS:
        path = os.path.join(feat_dir, f"{name}.pt")
        if os.path.exists(path):
            cached[name] = torch.load(path, map_location=device)
        else:
            missing.append(name)
    if not missing:
        return cached

    # Real extraction requires the CLEVR images + questions on disk. We perform
    # a lazy import and a guarded walk; any failure raises FileNotFoundError so
    # callers can fall back to synthetic data.
    import glob
    import json

    from PIL import Image

    questions_path = os.path.join(clevr_root, "questions", "CLEVR_train_questions.json")
    images_dir = os.path.join(clevr_root, "images", "train")
    if not (os.path.exists(questions_path) and os.path.isdir(images_dir)):
        raise FileNotFoundError(
            f"CLEVR not found under {clevr_root!r} (expected {questions_path} and "
            f"{images_dir}). Use synthetic_features() as a fallback."
        )

    vit, proj = get_vit_encoder(device)
    import timm.data

    cfg = timm.data.resolve_data_config({}, model=vit)
    transform = timm.data.create_transform(**cfg)

    with open(questions_path) as f:
        questions = json.load(f)["questions"]

    out = {name: [] for name in missing}
    for q in questions:
        for name in missing:
            if len(out[name]) >= n_per_category:
                continue
            if CATEGORY_FILTERS[name](q):
                img_glob = glob.glob(os.path.join(images_dir, q["image_filename"]))
                if not img_glob:
                    continue
                img = Image.open(img_glob[0]).convert("RGB")
                with torch.no_grad():
                    t = transform(img).unsqueeze(0).to(device)
                    feat = proj(vit(t))
                    feat = torch.nn.functional.normalize(feat, dim=1)
                out[name].append(feat.squeeze(0).cpu())
        if all(len(out[n]) >= n_per_category for n in missing):
            break

    for name in missing:
        if not out[name]:
            raise FileNotFoundError(
                f"No CLEVR features extracted for category {name!r}. "
                "Falling back to synthetic_features() is recommended."
            )
        tensor = torch.stack(out[name])
        torch.save(tensor, os.path.join(feat_dir, f"{name}.pt"))
        cached[name] = tensor
    return cached


def load_features(clevr_root: str = "data"):
    """Load cached features. Raises FileNotFoundError listing missing categories."""
    feat_dir = _feature_dir(clevr_root)
    out = {}
    missing = []
    for name in SPELKE_SYSTEMS:
        path = os.path.join(feat_dir, f"{name}.pt")
        if os.path.exists(path):
            out[name] = torch.load(path, map_location="cpu")
        else:
            missing.append(name)
    if missing:
        raise FileNotFoundError(
            f"Cached features missing for categories {missing} under {feat_dir!r}. "
            "Run extract_clevr_features() or use synthetic_features() as a fallback."
        )
    return out


def synthetic_features(
    n_per_category: int = 500,
    seed_dim: int = 128,
    device: str = "cpu",
    category_weight: float = 1.0,
    identity_weight: float = 0.6,
):
    """Generate synthetic features with a two-component structure.

    Each item is::

        feature = category_weight * center[cat] + identity_weight * z_item

    where ``center[cat]`` is a per-category orthonormal direction (shared by all
    items in the category) and ``z_item`` is a unique random unit vector. The
    result is L2-normalized.

    This mirrors real visual features far better than a single tight Gaussian
    blob: items in a category share structure (so same-category cosine stays
    well above the Phase 2 threshold ~ w_c^2/(w_c^2 + w_i^2)) yet remain
    individually distinct (so they are separately storable, which Phase 3
    capacity depends on). Cross-category cosine stays near zero because the
    category centers are orthonormal. This is the automatic fallback when CLEVR
    is unavailable.
    """
    warnings.warn(
        "Using SYNTHETIC features (CLEVR not found). Results are for pipeline "
        "validation, not paper-grade numbers.",
        stacklevel=2,
    )
    gen = torch.Generator().manual_seed(42)
    # Canonical centers — same vectors as SpelkeSeedLayer uses for seeds.
    # This alignment is what lets S-DAM's residuals capture only within-category
    # noise, giving it a measurable advantage over the baseline in Phase 1.
    centers = canonical_spelke_directions(len(SPELKE_SYSTEMS), seed_dim)

    out = {}
    for name, idx in SPELKE_SYSTEMS.items():
        z = torch.randn(n_per_category, seed_dim, generator=gen)
        z = torch.nn.functional.normalize(z, dim=1)              # unique item identity
        samples = category_weight * centers[idx].unsqueeze(0) + identity_weight * z
        samples = torch.nn.functional.normalize(samples, dim=1)
        out[name] = samples.to(device)
    return out
