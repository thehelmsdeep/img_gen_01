# Manual Denoising Loop

The project now connects the main pieces into a small manual denoising loop.

```text
Initial Latent
     ↓
┌───────────────┐
│ UNet predicts │
│ noise         │
└───────┬───────┘
        ↓
 Scheduler Step
        ↓
 Updated Latent
        ↓
 repeat N times
        ↓
 Final Latent
```

`denoise()` runs the loop for the requested number of inference steps.

This is an educational implementation. The normal text-to-image function still uses the complete Diffusers pipeline, while this function exposes the core iterative process directly.
