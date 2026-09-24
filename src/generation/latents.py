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
    """Create initial Gaussian noise in Stable Diffusion latent space."""
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


def denoise(
    scheduler,
    unet,
    latents,
    text_embeddings,
    num_inference_steps: int = 20,
    negative_embeddings=None,
    guidance_scale: float = 7.5,
):
    """Run the Stable Diffusion denoising loop manually."""
    if num_inference_steps < 1:
        raise ValueError("num_inference_steps must be at least 1.")

    scheduler.set_timesteps(num_inference_steps, device=latents.device)

    with torch.no_grad():
        for timestep in scheduler.timesteps:
            model_input = scheduler.scale_model_input(latents, timestep)

            if negative_embeddings is not None:
                noise_unconditional = unet(
                    model_input,
                    timestep,
                    encoder_hidden_states=negative_embeddings,
                ).sample
                noise_conditional = unet(
                    model_input,
                    timestep,
                    encoder_hidden_states=text_embeddings,
                ).sample
                noise_prediction = noise_unconditional + guidance_scale * (
                    noise_conditional - noise_unconditional
                )
            else:
                noise_prediction = unet(
                    model_input,
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
    """Decode Stable Diffusion latents into a batch of RGB NumPy images."""
    latents = latents / pipe.vae.config.scaling_factor

    with torch.no_grad():
        image = pipe.vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    return image.cpu().permute(0, 2, 3, 1).float().numpy()
