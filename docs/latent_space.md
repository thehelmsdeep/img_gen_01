# Latent Space

Stable Diffusion does not start generation directly from a full RGB image. It works in a smaller numerical representation called a **latent**.

For a 512x512 Stable Diffusion v1.5 image, the common latent size is approximately `4 x 64 x 64`.

The initial latent is random Gaussian noise.

## Flow

```text
Prompt
  ↓
Text Encoder
  ↓
Text Embeddings
  ↓
Random Latent Noise
  ↓
Denoising Process
  ↓
Clean Latent
  ↓
VAE Decoder
  ↓
RGB Image
```

The `create_latents()` utility lets the project create and inspect the initial noise tensor independently from the full Diffusers pipeline.

The existing `generate()` function still lets Diffusers manage the complete inference process.
