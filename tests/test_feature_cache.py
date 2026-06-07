"""Tests for probe/feature_cache.py (torch-free)."""
import numpy as np

from probe.feature_cache import cached_features, feature_cache_key


def test_key_deterministic():
    files = ["a/CLEVR_train_000001.png", "b/CLEVR_train_000002.png"]
    k1 = feature_cache_key("dinov2_vits14", files, "resize224-imagenetnorm")
    k2 = feature_cache_key("dinov2_vits14", files, "resize224-imagenetnorm")
    assert k1 == k2 and len(k1) == 16


def test_key_sensitive_to_model_preproc_and_filelist():
    files = ["CLEVR_train_000001.png", "CLEVR_train_000002.png"]
    base = feature_cache_key("dinov2_vits14", files, "p")
    assert feature_cache_key("imagenet_vits16", files, "p") != base   # model
    assert feature_cache_key("dinov2_vits14", files, "q") != base      # preprocessing
    assert feature_cache_key("dinov2_vits14", files[::-1], "p") != base  # order matters
    assert feature_cache_key("dinov2_vits14", files[:1], "p") != base    # membership


def test_cached_features_computes_then_loads(tmp_path):
    calls = {"n": 0}

    def fake_extract(paths):
        calls["n"] += 1
        return np.arange(len(paths) * 4, dtype=np.float32).reshape(len(paths), 4)

    paths = ["x1.png", "x2.png", "x3.png"]
    a = cached_features(str(tmp_path), "dinov2_vits14", paths, "p", fake_extract)
    b = cached_features(str(tmp_path), "dinov2_vits14", paths, "p", fake_extract)
    assert calls["n"] == 1                 # extractor ran once; second call hit cache
    assert np.array_equal(a, b)
    assert a.shape == (3, 4) and a.dtype == np.float32
