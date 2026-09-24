from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


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
