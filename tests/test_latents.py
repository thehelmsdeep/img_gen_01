import torch

from src.generation.latents import create_latents


def test_create_latents_shape_and_dtype():
    latents = create_latents(
        batch_size=1,
        channels=4,
        height=64,
        width=64,
        seed=42,
        device="cpu",
        dtype=torch.float32,
    )

    assert latents.shape == (1, 4, 64, 64)
    assert latents.dtype == torch.float32


def test_create_latents_seed_is_reproducible():
    first = create_latents(height=8, width=8, seed=42, device="cpu")
    second = create_latents(height=8, width=8, seed=42, device="cpu")

    assert torch.equal(first, second)


def test_create_latents_rejects_invalid_dimensions():
    import pytest

    with pytest.raises(ValueError):
        create_latents(height=0, width=64)
