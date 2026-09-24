# VAE Decoder

The final denoised latent is not an RGB image yet. The VAE decoder converts that latent representation into image pixels.

```text
Final Latent
    ↓
VAE Decoder
    ↓
Tensor of RGB values
    ↓
Pixel image
```

The new `decode_latents()` utility uses the pipeline VAE to decode the final latent and converts the result into a NumPy image array.

For Stable Diffusion v1.5, a 64×64 latent corresponds to a 512×512 image after decoding.

This completes the main conceptual path from text to pixels. The next step can combine the pieces into one small end-to-end pipeline instead of relying on the high-level Diffusers call.
