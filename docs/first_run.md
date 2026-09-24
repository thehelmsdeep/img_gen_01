# First Run

The first executable version of img_gen_01 uses Stable Diffusion v1.5 through Diffusers.

## What happens

1. main.py receives a text prompt.
2. The generation module starts generation.
3. The model loader loads Stable Diffusion.
4. On the first run, model files are downloaded and cached locally.
5. The model generates an image.
6. The image is saved under outputs/.

Example:

    python -m src.main "a small robot sitting on a desk"

The project does not call a paid image-generation API. The first model download requires an internet connection; generation itself is performed by the local Python process.

Stable Diffusion v1.5 is the first learning target, not necessarily the final model choice. Check the model license before distributing a product or generated content.
