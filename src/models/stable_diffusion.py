from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from src.generation.latents import create_latents, denoise, decode_latents
from src.prompt import encode_cfg_prompts


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pipeline(
    model_id: str,
    cache_dir: str | None = None,
    local_dir: str | None = None,
) -> StableDiffusionPipeline:
    """Load a model from a local directory or download/cache it locally."""
    device = choose_device()
    dtype = torch.float16 if device == "cuda" else torch.float32

    if local_dir:
        source = Path(local_dir)
        if not source.exists():
            raise FileNotFoundError(f"Local model directory not found: {source}")
        pipe = StableDiffusionPipeline.from_pretrained(
            source,
            torch_dtype=dtype,
            use_safetensors=True,
            local_files_only=True,
        )
    else:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            use_safetensors=True,
            cache_dir=cache_dir,
        )

    return pipe.to(device)


def manual_generate(
    pipe: StableDiffusionPipeline,
    prompt: str,
    steps: int = 20,
    negative_prompt: str = "",
    guidance_scale: float = 7.5,
    seed: int | None = None,
):
    """Run a minimal educational text-to-image path using pipeline components."""
    negative_embeddings, embeddings = encode_cfg_prompts(pipe, prompt, negative_prompt)

    height = 512
    width = 512
    latent_height = height // 8
    latent_width = width // 8
    latents = create_latents(
        channels=pipe.unet.config.in_channels,
        height=latent_height,
        width=latent_width,
        seed=seed,
        device=str(pipe.device),
        dtype=pipe.unet.dtype,
    )

    latents = latents * pipe.scheduler.init_noise_sigma

    latents = denoise(
        pipe.scheduler,
        pipe.unet,
        latents,
        embeddings,
        num_inference_steps=steps,
        negative_embeddings=negative_embeddings,
        guidance_scale=guidance_scale,
    )

    return decode_latents(pipe, latents)
