# Models

## Purpose

Models provide the learned parameters used by the generation pipeline.

The project does not require the model weights to be created by this repository. Public model weights may be downloaded and used locally when their license permits it.

## Storage

Large model files should not normally be committed to Git.

Recommended local layout:

~~~text
models/
├── model-a/
│   ├── ...
│   └── README.md
└── model-b/
    ├── ...
    └── README.md
~~~

## Model metadata

For every model used by the project, record:

- Model name
- Source
- Version / revision
- License
- Required hardware
- Expected memory usage
- Supported tasks
- Download instructions
- Any special configuration

## Long-term goal

The project should make it possible to understand the difference between:

- model weights,
- model architecture,
- tokenizer/text encoder,
- sampler or scheduler,
- decoder/VAE,
- and the application pipeline that connects them.
