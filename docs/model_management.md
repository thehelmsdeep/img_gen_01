# Model Management

The project now has a small model registry.

## Why

Instead of putting a model ID in many files, the project keeps model information in one place.

Current model:

- sd15 — Stable Diffusion v1.5

The model files are not stored in Git. They are downloaded/cached locally.

## Flow

    model name
        ↓
    model registry
        ↓
    model ID
        ↓
    Diffusers
        ↓
    local model cache
        ↓
    generation

As more models are added, the registry can contain multiple entries without changing the generation code.
