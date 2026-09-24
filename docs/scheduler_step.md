# Scheduler Step

After the UNet predicts the noise, the scheduler uses that prediction to update the latent.

```text
Current Latent
      ↓
     UNet
      ↓
Predicted Noise
      ↓
  Scheduler Step
      ↓
Updated Latent
      ↓
Next timestep
```

The new `scheduler_step()` utility performs one update and returns the updated latent.

This is one building block of the iterative denoising loop. The next step is to connect these pieces into a small manual denoising loop.
