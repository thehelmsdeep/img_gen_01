from pathlib import Path

from src.config import MODEL_CACHE_DIR, OUTPUT_DIR
from src.models.registry import get_model
from src.models.stable_diffusion import load_pipeline


def generate(
    prompt: str,
    output_path: str = str(OUTPUT_DIR / "image.png"),
    model_name: str = "sd15",
    steps: int = 30,
    width: int = 512,
    height: int = 512,
    seed: int | None = None,
) -> Path:
    """Generate one image locally using a registered model."""
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if width < 64 or height < 64:
        raise ValueError("width and height must be at least 64.")

    model = get_model(model_name)
    pipe = load_pipeline(model.model_id, cache_dir=str(MODEL_CACHE_DIR))

    generator = None
    if seed is not None:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt=prompt,
        num_inference_steps=steps,
        width=width,
        height=height,
        generator=generator,
    )

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.images[0].save(path)

    return path
