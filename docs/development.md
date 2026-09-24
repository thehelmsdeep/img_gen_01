# Development

## Development philosophy

Build the project in small, testable layers.

### Recommended order

1. Project configuration
2. Model discovery and loading
3. Minimal inference
4. Image saving
5. Prompt handling
6. Pipeline abstraction
7. Image-to-image
8. Inpainting and masking
9. Hardware optimization
10. Deeper custom implementations

## Code organization

Keep model-specific code isolated from generic application code.

For example:

~~~text
src/
├── models/       # model adapters and loading
├── pipeline/     # generation orchestration
├── generation/  # inference logic
├── image/        # image input/output and processing
└── utils/        # shared utilities
~~~

## Testing

Tests should cover:

- configuration loading,
- model path validation,
- input validation,
- deterministic generation settings where supported,
- image output,
- and individual pipeline components.

## Git

Do not commit:

- model weights,
- generated image collections,
- virtual environments,
- cache directories,
- secrets,
- API keys,
- local configuration containing private paths.

Large artifacts should be handled outside normal source-code commits.
