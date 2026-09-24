from pathlib import Path

import torch

from src.config import MODEL_CACHE_DIR, OUTPUT_DIR
from src.models.registry import get_model
from src.models.stable_diffusion import load_pipeline as load_sd_pipeline


def _load_flux_pipeline(model_id: str):
    from src.models.flux import load_pipeline

    return load_pipeline(model_id, cache_dir=str(MODEL_CACHE_DIR))


def generate(
    prompt: str,
    output_path: str = str(OUTPUT_DIR / "image.png"),
    model_name: str = "flux",
    negative_prompt: str | None = None,
    steps: int = 4,
    width: int = 512,
    height: int = 512,
    seed: int | None = None,
    guidance_scale: float = 0.0,
) -> Path:
    """Generate one image locally using a registered model."""
    prepared_prompt = " ".join(prompt.split()).strip()
    if not prepared_prompt:
        raise ValueError("Prompt cannot be empty.")

    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if width < 64 or height < 64:
        raise ValueError("width and height must be at least 64.")

    model = get_model(model_name)

    if model.pipeline == "flux":
        pipe = _load_flux_pipeline(model.model_id)

        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(seed)

        result = pipe(
            prompt=prepared_prompt,
            num_inference_steps=steps,
            guidance_scale=0.0,
            width=width,
            height=height,
            max_sequence_length=256,
            generator=generator,
        )
    else:
        pipe = load_sd_pipeline(
            model.model_id,
            cache_dir=str(MODEL_CACHE_DIR),
        )

        generator = None
        if seed is not None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            generator = torch.Generator(device=device).manual_seed(seed)

        result = pipe(
            prompt=prepared_prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            width=width,
            height=height,
            generator=generator,
        )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(path)

    return path
