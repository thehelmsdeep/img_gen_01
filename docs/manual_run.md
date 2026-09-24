# Running the Manual Pipeline

The manual educational pipeline can now be selected from the CLI.

```bash
python -m src.main "a small robot on a desk" --manual --steps 20 --guidance-scale 7.5 --seed 42
```

Optional negative prompt:

```bash
python -m src.main "a small robot on a desk" --manual --negative-prompt "blurry, distorted" --steps 20
```

This mode connects prompt encoding, CFG, latent initialization, UNet denoising, scheduler updates, and VAE decoding without calling the high-level `pipe(prompt)` method.

It is intentionally educational and may not match the quality of the production Diffusers pipeline yet. The next work can focus on matching the official Stable Diffusion inference details, including latent initialization, scheduler configuration, and image normalization.
