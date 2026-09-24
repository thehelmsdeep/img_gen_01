from pathlib import Path

from src.models.stable_diffusion import load_pipeline


def generate(
    prompt: str,
    output_path: str = "outputs/image.png",
    model_id: str | None = None,
    steps: int = 30,
    width: int = 512,
    height: int = 512,
    seed: int | None = None,
) -> Path:
    """Generate one image locally with basic generation controls."""
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty.")
    if steps < 1:
        raise ValueError("steps must be at least 1.")
    if width < 64 or height < 64:
        raise ValueError("width and height must be at least 64.")

    pipe = load_pipeline(model_id=model_id) if model_id else load_pipeline()

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
