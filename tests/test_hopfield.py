import pytest
import torch

from sdam.hopfield import HopfieldLayer


def test_energy_non_increasing_across_iterations():
    torch.manual_seed(0)
    layer = HopfieldLayer(32, beta=4.0, max_iter=1)
    patterns = torch.randn(10, 32)
    layer.store(patterns)
    xi = patterns[0] + 0.1 * torch.randn(32)
    prev = layer.energy(xi)
    for _ in range(5):
        xi = layer.retrieve(xi)
        cur = layer.energy(xi)
        assert cur <= prev + 1e-5, "Hopfield energy must not increase across updates"
        prev = cur


def test_retrieve_raises_when_empty():
    layer = HopfieldLayer(8)
    with pytest.raises(RuntimeError, match="before any patterns"):
        layer.retrieve(torch.randn(8))


def test_retrieve_multi_clips_k():
    layer = HopfieldLayer(8, beta=2.0)
    layer.store(torch.randn(3, 8))
    out = layer.retrieve_multi(torch.randn(8), k=10)  # k > n_stored
    assert out.shape == (8,)


def test_reset_clears_patterns():
    layer = HopfieldLayer(8)
    layer.store(torch.randn(4, 8))
    assert layer.n_stored == 4
    layer.reset()
    assert layer.n_stored == 0
    assert layer.stored_patterns is None


def test_n_stored_counts_correctly():
    layer = HopfieldLayer(8)
    assert layer.n_stored == 0
    layer.store(torch.randn(2, 8))
    layer.store(torch.randn(3, 8))
    assert layer.n_stored == 5


def test_energy_distance_inf_when_empty():
    layer = HopfieldLayer(8)
    d = layer.energy_distance(torch.randn(8))
    assert torch.isinf(d)
