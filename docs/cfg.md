# Classifier-Free Guidance

CFG helps the generated image follow the prompt more strongly.

At each denoising step, the UNet can predict noise twice:

```text
Empty / negative prompt → UNet → unconditional noise
Positive prompt         → UNet → conditional noise
                         ↓
                       CFG
                         ↓
                 guided noise
```

The basic formula is:

`guided = unconditional + scale × (conditional - unconditional)`

The project now exposes `encode_cfg_prompts()` and `apply_cfg()` for this step.

A common guidance scale is around `7.5`, but the useful range depends on the model and prompt.
