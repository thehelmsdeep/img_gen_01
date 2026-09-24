from pathlib import Path

import torch
from diffusers import FluxPipeline


def choose_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_pipeline(
    model_id: str,
    cache_dir: str | None = None,
    local_dir: str | None = None,
) -> FluxPipeline:
    """Load a FLUX pipeline from a local directory or Hugging Face cache."""
    device = choose_device()

    # FLUX.1 is large. On CUDA, CPU offload keeps most weights in system RAM
    # and moves only the needed parts to the GPU.
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    if local_dir:
        source = Path(local_dir)
        if not source.exists():
            raise FileNotFoundError(f"Local model directory not found: {source}")
        pipe = FluxPipeline.from_pretrained(
            source,
            torch_dtype=dtype,
            local_files_only=True,
        )
    else:
        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=cache_dir,
        )

    if device == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)

    return pipe
