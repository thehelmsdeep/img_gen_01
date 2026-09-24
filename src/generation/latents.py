import torch


def create_latents(
    batch_size: int = 1,
    channels: int = 4,
    height: int = 64,
    width: int = 64,
    seed: int | None = None,
    device: str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create the initial Gaussian noise tensor used by latent diffusion."""
    if batch_size < 1 or channels < 1 or height < 1 or width < 1:
        raise ValueError("Latent dimensions must be positive.")

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    return torch.randn(
        (batch_size, channels, height, width),
        generator=generator,
        device=device,
        dtype=dtype,
    )