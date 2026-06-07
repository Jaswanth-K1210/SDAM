"""CLEVR scene-graph -> Spelke factor values (Milestone 1, 3 factors).

Factors are read from the CLEVR scene JSON (not the questions — the scene graph
is cleaner and gives exact ground truth):

  shape  -> Objectness   (dominant object shape; FLAGGED proxy for object
                          identity, NOT full Spelke permanence/cohesion/solidity)
  count  -> Numerosity    (number of objects in the scene)
  layout -> Geometry      (spatial spread: std of object image-x coordinates,
                          fallback to left/right relationship counts)

Agentness is intentionally ABSENT: CLEVR has no agents. This is a 3-factor probe.
"""
from __future__ import annotations

import json

import numpy as np

# Canonical shape -> class id. Tie-break order follows this mapping.
SHAPE_CLASSES = {"cube": 0, "sphere": 1, "cylinder": 2}


def shape_factor(scene: dict) -> int:
    """Dominant (most frequent) object shape in the scene -> {0,1,2}.

    Ties broken by SHAPE_CLASSES order (cube < sphere < cylinder).
    Maps to Spelke OBJECTNESS (FLAGGED: object-identity proxy, not full
    Objectness).
    """
    counts = {name: 0 for name in SHAPE_CLASSES}
    for obj in scene["objects"]:
        if obj["shape"] in counts:
            counts[obj["shape"]] += 1
    # Highest count wins; tie broken by smallest SHAPE_CLASSES index (negate it
    # in the key so a smaller index sorts higher under max()).
    best = max(SHAPE_CLASSES, key=lambda name: (counts[name], -SHAPE_CLASSES[name]))
    return SHAPE_CLASSES[best]


def count_factor(scene: dict) -> int:
    """Number of objects in the scene. Maps to Spelke NUMEROSITY."""
    return len(scene["objects"])


def _relationship_total(rel_list) -> int:
    """Total count of relationship entries. CLEVR stores relationships['left']
    as a list (one entry per object) of index lists; sum their lengths. Tolerate
    a flat list too."""
    if not rel_list:
        return 0
    if isinstance(rel_list[0], (list, tuple)):
        return int(sum(len(x) for x in rel_list))
    return len(rel_list)


def layout_factor(scene: dict) -> float:
    """Spatial spread = std of object image-plane x-coordinates.

    Falls back to (left + right) relationship counts when pixel_coords are
    unavailable. Maps to Spelke GEOMETRY.
    """
    xs = [
        float(o["pixel_coords"][0])
        for o in scene["objects"]
        if o.get("pixel_coords") is not None
    ]
    if len(xs) >= 2:
        return float(np.std(xs))
    rel = scene.get("relationships", {})
    return float(_relationship_total(rel.get("left", [])) + _relationship_total(rel.get("right", [])))


def load_clevr_scenes(scenes_json_path: str, max_scenes: int = 6000) -> list[dict]:
    """Load CLEVR_*_scenes.json -> list of scene dicts (truncated to max_scenes)."""
    with open(scenes_json_path) as f:
        data = json.load(f)
    return data["scenes"][:max_scenes]


def build_factor_arrays(scenes: list[dict]) -> dict:
    """Return {'shape': (n,) int, 'count': (n,) int, 'layout': (n,) float}.

    Order matches the scene list so it aligns with extracted features by index.
    """
    shape = np.array([shape_factor(s) for s in scenes], dtype=np.int64)
    count = np.array([count_factor(s) for s in scenes], dtype=np.int64)
    layout = np.array([layout_factor(s) for s in scenes], dtype=np.float64)
    return {"shape": shape, "count": count, "layout": layout}
