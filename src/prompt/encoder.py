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


def encode_cfg_prompts(
    pipe: StableDiffusionPipeline,
    prompt: str,
    negative_prompt: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode negative and positive prompts for classifier-free guidance."""
    positive = encode_prompt(pipe, prompt).hidden_states
    negative = encode_prompt(pipe, negative_prompt).hidden_states
    return negative, positive


def apply_cfg(
    noise_unconditional: torch.Tensor,
    noise_conditional: torch.Tensor,
    guidance_scale: float = 7.5,
) -> torch.Tensor:
    """Combine unconditional and conditional noise predictions with CFG."""
    return noise_unconditional + guidance_scale * (
        noise_conditional - noise_unconditional
    )
