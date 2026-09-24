from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    name: str
    model_id: str
    description: str
    pipeline: str


MODELS = {
    "flux": ModelInfo(
        name="FLUX.1 schnell",
        model_id="black-forest-labs/FLUX.1-schnell",
        description="Local FLUX.1 text-to-image model optimized for few-step generation.",
        pipeline="flux",
    ),
    "sd15": ModelInfo(
        name="Stable Diffusion v1.5",
        model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        description="Stable Diffusion v1.5 text-to-image model.",
        pipeline="stable_diffusion",
    ),
}


def get_model(name: str) -> ModelInfo:
    try:
        return MODELS[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODELS))
        raise ValueError(
            f"Unknown model '{name}'. Available models: {available}"
        ) from exc
