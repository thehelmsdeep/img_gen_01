# Denoising and Scheduler

The diffusion process is controlled by a **scheduler**.

The scheduler defines how noise changes across diffusion timesteps.

```text
Random noise
    ↓
timestep
    ↓
predict noise
    ↓
remove part of noise
    ↓
next timestep
    ↓
repeat
    ↓
clean latent
```

The project now includes a scheduler utility and `add_noise()` so we can inspect controlled noise at different timesteps.

The full Stable Diffusion pipeline still manages its own scheduler during normal generation.

The next step is to connect the **UNet noise predictor** to this process.