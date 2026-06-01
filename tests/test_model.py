import torch

from sdam.model import SDAM


def test_write_then_read_shape():
    model = SDAM(input_dim=64)
    model.reset_memory()
    x = torch.randn(5, 64)
    model.write(x)
    out = model.read(x)
    assert out.shape == x.shape


def test_above_tau_written_to_longterm():
    model = SDAM(input_dim=64, write_threshold=0.1)
    model.reset_memory()
    x = torch.randn(8, 64)  # residual norms ~ 1 >> 0.1
    diag = model.write(x)
    assert model.mem.n_stored > 0
    assert bool(diag["written"].any())


def test_below_tau_goes_to_provisional():
    model = SDAM(input_dim=64, write_threshold=0.1)
    model.reset_memory()
    # A vector lying inside the seed subspace has ~zero residual.
    x = model.ssl.seeds[0].detach().clone()
    diag = model.write(x)
    assert model.mem.n_stored == 0
    assert diag["n_provisional"] == 1
    assert len(model.provisional) == 1


def test_reset_memory_clears_everything():
    model = SDAM(input_dim=32, write_threshold=0.1)
    model.write(torch.randn(4, 32))
    model.write(model.ssl.seeds[0].detach().clone())  # one provisional
    model.reset_memory()
    assert model.mem.n_stored == 0
    assert model.provisional == []
    assert model.failure_counts == {}


def test_consolidate_promotes_high_norm_provisional():
    model = SDAM(input_dim=32, write_threshold=0.1)
    model.reset_memory()
    model.tau.data = torch.tensor(1e6)         # force everything provisional
    model.write(torch.randn(32))
    assert len(model.provisional) == 1
    model.tau.data = torch.tensor(0.1)         # now it clears tau
    summary = model.consolidate()
    assert summary["promoted"] == 1
    assert model.mem.n_stored == 1
    assert len(model.provisional) == 0


def test_new_attractor_at_failure_threshold():
    model = SDAM(input_dim=32, write_threshold=0.1, failure_threshold=3)
    model.reset_memory()
    model.tau.data = torch.tensor(1e6)         # pattern can never clear tau
    model.write(torch.randn(32))
    start_seeds = model.ssl.n_systems
    for _ in range(3):
        model.consolidate()
    assert model.ssl.n_systems == start_seeds + 1
    assert len(model.provisional) == 0


def test_diagnostics_keys():
    model = SDAM(input_dim=16)
    diag = model.diagnostics()
    for key in ["n_stored", "n_provisional", "n_seeds", "tau", "seeds_orthogonal"]:
        assert key in diag
