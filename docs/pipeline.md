# Generation Pipeline

A typical text-to-image system can be understood as a sequence of transformations.

~~~text
Text
  |
  v
Tokenization
  |
  v
Text Representation / Conditioning
  |
  v
Generation Process
  |
  v
Latent Representation
  |
  v
Decoder
  |
  v
Pixel Image
~~~

The exact architecture depends on the model.

## Planned modules

### Prompt processing
Responsible for validating and preparing user input.

### Model loading
Responsible for locating model files and loading the required components.

### Generation
Responsible for the actual inference process and generation parameters.

### Decoding
Converts the model's internal representation into an image when required.

### Image output
Responsible for saving and exporting generated images.

## Future editing pipeline

~~~text
Input Image
    |
    +---- Prompt
    |
    +---- Mask (optional)
    |
    v
Image Editing Model
    |
    v
Output Image
~~~

This will eventually support workflows such as image-to-image and inpainting.
