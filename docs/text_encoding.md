# Text Encoding

The next stage after tokenization is the text encoder.

## Flow

    Prompt
       ↓
    Tokenizer
       ↓
    Token IDs
       ↓
    Text Encoder
       ↓
    Text Embeddings
       ↓
    Diffusion model

The text encoder converts token IDs into vectors called **embeddings**.

These vectors contain the numerical representation of the prompt that the diffusion model uses as conditioning.

For Stable Diffusion v1.5, the text encoder is part of the loaded pipeline. The project can inspect its output without generating an image.

Example:

    from src.models.registry import get_model
    from src.models.stable_diffusion import load_pipeline
    from src.prompt import encode_prompt

    model = get_model("sd15")
    pipe = load_pipeline(model.model_id)

    encoded = encode_prompt(pipe, "a small robot on a desk")
    print(encoded.shape)

The exact shape depends on the tokenizer/text-encoder configuration of the selected model.

This stage is important because the diffusion model does not work directly with human-readable words. It receives numerical representations produced by the text encoder.
