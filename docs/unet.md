# UNet Noise Prediction

The **UNet** is the main neural network used during Stable Diffusion denoising.

At each timestep it receives the current noisy latent, the timestep, and the text embeddings from the prompt. It predicts the noise that should be removed.

```text
Noisy Latent + Timestep + Text Embeddings
                    ↓
                   UNet
                    ↓
             Predicted Noise
                    ↓
                Scheduler
                    ↓
              Less Noise
```

The project now exposes `predict_noise()` so one UNet prediction can be inspected directly.

The normal `generate()` function still uses the complete Diffusers pipeline. This is an educational entry point for understanding the individual denoising components.
