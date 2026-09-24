# Dependency Setup

The project keeps its core Python dependencies in `requirements.txt`.

Install them with:

```bash
pip install -r requirements.txt
```

Core packages:

- `torch` — tensor operations and model execution
- `diffusers` — diffusion model components and pipelines
- `transformers` — text tokenizer and encoder components
- `accelerate` — device and inference utilities
- `safetensors` — safe model-weight loading
- `Pillow` — image file handling

The versions are intentionally not pinned yet because the exact PyTorch build depends on the user's operating system and CPU/GPU setup. Once the target environment is chosen, we can add a tested lock/pin strategy.
