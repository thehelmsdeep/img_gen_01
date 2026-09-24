# img_gen_01

A self-hosted image-generation project focused on learning how modern image generation systems work while keeping the application independent from paid image-generation APIs.

## Goals

- Run image-generation models locally or on hardware controlled by the user.
- Download model weights instead of depending on a hosted inference API.
- Keep the application code modular and understandable.
- Build the project incrementally, from a simple prototype toward a complete generation pipeline.
- Document the architecture, models, pipeline, and development process.

## What this project is

The project is the software around an image-generation model. A model can be downloaded from a public model repository and stored locally; the project's own code is responsible for loading it, preparing inputs, running generation, processing the result, and eventually providing an interface.

~~~text
User Prompt
    |
    v
Prompt Processing
    |
    v
Model / Text Encoder
    |
    v
Image Generation Pipeline
    |
    v
Decoder / Image Processing
    |
    v
Generated Image
~~~

## Independence

This project is designed to avoid dependence on a paid hosted image-generation API.

That does not mean the project must train a large model from scratch. Publicly available model weights can be downloaded once and then used locally, subject to their individual licenses.

The long-term project can also experiment with implementing parts of the generation pipeline ourselves.

## Planned structure

~~~text
img_gen_01/
├── README.md
├── docs/
│   ├── architecture.md
│   ├── installation.md
│   ├── models.md
│   ├── pipeline.md
│   └── development.md
├── src/
│   ├── models/
│   ├── pipeline/
│   ├── generation/
│   ├── image/
│   └── utils/
├── models/
├── tests/
└── examples/
~~~

## Project stages

### Stage 1 — Local model execution
Download a compatible model and run inference without a paid image-generation API. The current default is FLUX.1 schnell.

### Stage 2 — Own pipeline
Separate model loading, prompt processing, generation, decoding, and image output into our own modules.

### Stage 3 — Image editing
Add image-to-image, inpainting, masking, and other editing workflows.

### Stage 4 — Deeper implementation
Implement and study individual components of the generation process rather than treating the model as a black box.

### Stage 5 — Experiments
Test smaller models, custom components, alternative samplers, conditioning methods, and eventually custom architectures.

## Current default model

The default model is `black-forest-labs/FLUX.1-schnell`. Diffusers documents this as the timestep-distilled FLUX variant, intended for few-step generation; its recommended usage uses `guidance_scale=0` and up to 256 prompt tokens. The model repository is gated, so access requires accepting its conditions on Hugging Face before the weights can be downloaded.

## Important

Model files can be very large and should normally not be committed to Git. Store them locally or use an appropriate model-storage mechanism. Always check the license of a model before using or redistributing it.

See the docs/ directory for project documentation.
