# Prompt Processing

The project now has a small layer between the command line and the image-generation pipeline.

## Flow

    User text
        ↓
    Prompt processor
        ↓
    Prompt object
        ↓
    Generation pipeline
        ↓
    Model

The first version only cleans unnecessary whitespace and validates that the prompt is not empty.

A negative prompt can also be stored for models that support it.

The important idea is that prompt handling is now separate from model execution. This lets the project later add prompt templates, metadata, token inspection, prompt weighting, or other processing without putting that logic inside the model loader.
