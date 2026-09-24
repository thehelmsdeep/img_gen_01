# Manual Text-to-Image Pipeline

The project now has an educational path that connects the major Stable Diffusion components directly.

```text
Prompt
  ↓
Tokenizer + Text Encoder
  ↓
Text Embeddings
  ↓
Random Latent
  ↓
UNet + Scheduler
  ↓
Denoising Loop
  ↓
Final Latent
  ↓
VAE Decoder
  ↓
RGB Image
```

`manual_generate()` performs these stages using components from the loaded pipeline instead of calling the high-level `pipe(prompt)` generation method.

This is intentionally a small learning implementation. It is not yet a feature-complete replacement for Diffusers inference: production features such as classifier-free guidance, scheduler-specific initialization, latent scaling details, safety handling, and optimized memory management still need to be added.
