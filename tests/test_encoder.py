"""Encoder smoke test — runs ONLY on Colab (GPU + a few CLEVR images).

Skipped unless torch is importable, CUDA is available, and CLEVR_SAMPLE points
at a directory of CLEVR .png images. Never blocks the local/CI suite.
"""
import importlib
import os

import pytest

_has_torch = importlib.util.find_spec("torch") is not None


@pytest.mark.skipif(
    not _has_torch or os.environ.get("CLEVR_SAMPLE") is None,
    reason="needs torch + GPU and CLEVR_SAMPLE pointing at a CLEVR image dir",
)
def test_extract_features_shape():
    import glob

    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from probe.encoder import extract_features, get_dinov2_encoder

    paths = sorted(glob.glob(os.path.join(os.environ["CLEVR_SAMPLE"], "*.png")))[:8]
    assert paths, "no .png images found under CLEVR_SAMPLE"
    model = get_dinov2_encoder("cuda")
    feats = extract_features(model, paths, "cuda", batch_size=4)
    assert feats.shape == (len(paths), 384)
    assert feats.dtype.name == "float32"
