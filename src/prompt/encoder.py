from dataclasses import dataclass

import torch
from diffusers import StableDiffusionPipeline


@dataclass(frozen=True)
class TextEncoding:
    hidden_states: torch.Tensor
    shape: tuple[int, ...]


def encode_prompt(
    pipe: StableDiffusionPipeline,
    prompt: str,
) -> TextEncoding:
    """Convert a prompt into the text embeddings used by the diffusion model."""
    inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )

    device = pipe.device
    input_ids = inputs.input_ids.to(device)

    with torch.no_grad():
        hidden_states = pipe.text_encoder(input_ids)[0]

    return TextEncoding(
        hidden_states=hidden_states,
        shape=tuple(hidden_states.shape),
    )
