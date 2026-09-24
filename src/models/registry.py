from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    name: str
    model_id: str
    description: str


MODELS = {
    "sd15": ModelInfo(
        name="Stable Diffusion v1.5",
        model_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        description="First local text-to-image model for the project.",
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
