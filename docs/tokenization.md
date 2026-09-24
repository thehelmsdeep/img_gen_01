# Prompt Tokenization

After prompt processing, the text is converted into tokens by the text tokenizer used by the model.

## Flow

    Prompt
       ↓
    Prompt processor
       ↓
    Tokenizer
       ↓
    Token IDs
       ↓
    Text encoder
       ↓
    Conditioning for generation

Tokens are pieces of text, not necessarily complete words. The tokenizer converts those pieces into numeric token IDs that the model can process.

The current implementation uses the tokenizer that belongs to the loaded Stable Diffusion pipeline, so tokenization stays tied to the selected model.

The tokenizer can also be inspected directly with the project's Python API:

    from src.models.registry import get_model
    from src.models.stable_diffusion import load_pipeline
    from src.prompt import tokenize_prompt

    model = get_model("sd15")
    pipe = load_pipeline(model.model_id)
    tokens = tokenize_prompt(pipe, "a small robot on a desk")
    print(tokens)

This step does not generate an image. It lets us inspect the first transformation from human text into model input.
