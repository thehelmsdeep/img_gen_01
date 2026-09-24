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


def predict_noise(pipe, latents, timestep, text_embeddings):
    """Use the pipeline UNet to predict noise for one timestep."""
    timestep_tensor = torch.tensor([timestep], device=latents.device)
    with torch.no_grad():
        return pipe.unet(
            latents,
            timestep_tensor,
            encoder_hidden_states=text_embeddings,
        ).sample


def scheduler_step(scheduler, noise_prediction, timestep, latents):
    """Apply one scheduler update using the UNet noise prediction."""
    scheduler.set_timesteps(scheduler.config.num_train_timesteps, device=latents.device)
    timestep_tensor = torch.tensor(timestep, device=latents.device)
    with torch.no_grad():
        return scheduler.step(noise_prediction, timestep_tensor, latents).prev_sample


def add_noise(scheduler, latents, timestep, noise=None):
    """Add controlled noise at a selected diffusion timestep."""
    if noise is None:
        noise = torch.randn_like(latents)
    timestep_tensor = torch.tensor([timestep], device=latents.device)
    return scheduler.add_noise(latents, noise, timestep_tensor)

def denoise(
    scheduler,
    unet,
    latents,
    text_embeddings,
    num_inference_steps: int = 20,
    negative_embeddings=None,
    guidance_scale: float = 7.5,
):
    """Run a small manual latent denoising loop."""
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be at least 1.")

    scheduler.set_timesteps(num_inference_steps, device=latents.device)

    with torch.no_grad():
        for timestep in scheduler.timesteps:
            if negative_embeddings is not None:
                noise_unconditional = unet(
                    latents,
                    timestep,
                    encoder_hidden_states=negative_embeddings,
                ).sample
                noise_conditional = unet(
                    latents,
                    timestep,
                    encoder_hidden_states=text_embeddings,
                ).sample
                noise_prediction = noise_unconditional + guidance_scale * (
                    noise_conditional - noise_unconditional
                )
            else:
                noise_prediction = unet(
                    latents,
                    timestep,
                    encoder_hidden_states=text_embeddings,
                ).sample
            latents = scheduler.step(
                noise_prediction,
                timestep,
                latents,
            ).prev_sample

    return latents


def decode_latents(pipe, latents):
    """Decode final latents into an RGB image using the pipeline VAE."""
    scaling_factor = pipe.vae.config.scaling_factor
    latents = latents / scaling_factor

    with torch.no_grad():
        image = pipe.vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    return image
