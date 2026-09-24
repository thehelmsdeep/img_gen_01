from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pipeline(
    model_id: str = DEFAULT_MODEL_ID,
    local_dir: str | None = None,
) -> StableDiffusionPipeline:
    """Load Stable Diffusion locally."""
    device = choose_device()
    source = Path(local_dir) if local_dir else model_id

    pipe = StableDiffusionPipeline.from_pretrained(
        source,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        use_safetensors=True,
    )

    return pipe.to(device)
