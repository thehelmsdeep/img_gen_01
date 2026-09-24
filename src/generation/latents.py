import torch
from diffusers import DDPMScheduler


def create_scheduler() -> DDPMScheduler:
    """Create the diffusion scheduler used to control denoising steps."""
    return DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
    )


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


def add_noise(scheduler, latents, timestep, noise=None):
    """Add controlled noise at a selected diffusion timestep."""
    if noise is None:
        noise = torch.randn_like(latents)
    timestep_tensor = torch.tensor([timestep], device=latents.device)
    return scheduler.add_noise(latents, noise, timestep_tensor)