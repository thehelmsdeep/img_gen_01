# First Run

The first executable version of img_gen_01 uses Stable Diffusion v1.5 through Diffusers.

## What happens

1. `src/main.py` receives a text prompt and generation settings.
2. The generation module selects a registered model.
3. The model loader downloads the model on first use and caches it under the project's `models/` directory.
4. The local pipeline generates an image.
5. The image is saved under `outputs/`.

Example:

    python -m src.main "a small robot sitting on a desk" --model sd15 --steps 30 --width 512 --height 512 --seed 42

## Main controls

- `--model`: registered model name.
- `--steps`: number of denoising steps.
- `--width`: output width.
- `--height`: output height.
- `--seed`: optional seed for repeatable generation settings.

The project does not call a paid image-generation API. The first model download requires an internet connection; generation is performed by the local Python process.

Stable Diffusion v1.5 is the first learning target, not necessarily the final model choice. Check the model license before distributing a product or generated content.
