from diffusers import StableDiffusionPipeline


def tokenize_prompt(
    pipe: StableDiffusionPipeline,
    prompt: str,
) -> list[str]:
    """Return the token strings produced by the model's text tokenizer."""
    inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        truncation=True,
        max_length=pipe.tokenizer.model_max_length,
        return_tensors="pt",
    )

    token_ids = inputs.input_ids[0].tolist()
    return pipe.tokenizer.convert_ids_to_tokens(token_ids)
