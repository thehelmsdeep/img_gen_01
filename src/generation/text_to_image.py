from pathlib import Path

from src.models.stable_diffusion import load_pipeline


def generate(
    prompt: str,
    output_path: str = "outputs/image.png",
    model_id: str | None = None,
) -> Path:
    """Generate one image from a text prompt and save it locally."""
    pipe = load_pipeline(model_id=model_id) if model_id else load_pipeline()
    image = pipe(prompt).images[0]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)

    return path
