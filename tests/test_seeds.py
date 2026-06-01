import torch

from sdam.seeds import SpelkeSeedLayer


def test_seeds_orthogonal_after_init():
    ssl = SpelkeSeedLayer(seed_dim=128, n_systems=4)
    assert ssl.verify_orthogonality(), "seeds should be orthonormal right after init"


def test_residual_orthogonal_to_all_seeds():
    ssl = SpelkeSeedLayer(seed_dim=128, n_systems=4)
    x = torch.randn(8, 128)
    r = ssl.residual(x)
    dots = r @ ssl.seeds.t()  # (8, 4)
    assert torch.allclose(dots, torch.zeros_like(dots), atol=1e-4), (
        "residual must be orthogonal to every seed"
    )


def test_proj_plus_residual_equals_x():
    ssl = SpelkeSeedLayer(seed_dim=64, n_systems=4)
    x = torch.randn(5, 64)
    recon = ssl.project(x) + ssl.residual(x)
    assert torch.allclose(recon, x, atol=1e-5), "proj(x) + residual(x) must equal x"


def test_proj_plus_residual_unbatched():
    ssl = SpelkeSeedLayer(seed_dim=64, n_systems=4)
    x = torch.randn(64)
    recon = ssl.project(x) + ssl.residual(x)
    assert torch.allclose(recon, x, atol=1e-5)


def test_category_in_range():
    ssl = SpelkeSeedLayer(seed_dim=32, n_systems=4)
    x = torch.randn(20, 32)
    cats = ssl.category(x)
    assert cats.dtype in (torch.int64, torch.long)
    assert int(cats.min()) >= 0 and int(cats.max()) <= 3


def test_high_inertia_param_vs_buffer():
    ssl_param = SpelkeSeedLayer(seed_dim=16, n_systems=4, use_high_inertia=True)
    ssl_buf = SpelkeSeedLayer(seed_dim=16, n_systems=4, use_high_inertia=False)
    assert isinstance(ssl_param.seeds, torch.nn.Parameter)
    assert not isinstance(ssl_buf.seeds, torch.nn.Parameter)
    assert "seeds" in dict(ssl_buf.named_buffers())
