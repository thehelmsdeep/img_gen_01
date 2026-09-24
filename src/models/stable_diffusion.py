from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from src.generation.latents import create_latents, denoise, decode_latents
from src.prompt import encode_prompt


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
    seed: int | None = None,
):
    """Run a minimal educational text-to-image path using pipeline components."""
    embeddings = encode_prompt(pipe, prompt).hidden_states

    height = 64
    width = 64
    latents = create_latents(
        height=height,
        width=width,
        seed=seed,
        device=str(pipe.device),
        dtype=pipe.unet.dtype,
    )

    latents = denoise(
        pipe.scheduler,
        pipe.unet,
        latents,
        embeddings,
        num_inference_steps=steps,
    )

    return decode_latents(pipe, latents)
