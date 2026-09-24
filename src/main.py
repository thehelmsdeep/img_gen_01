import argparse

from src.generation.text_to_image import generate
from src.models.registry import MODELS


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local text-to-image generation"
    )
    parser.add_argument("prompt", help="Text prompt for the image")
    parser.add_argument(
        "--model",
        default="flux",
        choices=sorted(MODELS),
        help="Registered model name (default: flux)",
    )
    parser.add_argument("--output", default="outputs/image.png")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=0.0,
        help="Used by SD 1.5; FLUX.1 schnell requires 0.0.",
    )
    args = parser.parse_args()

    path = generate(
        prompt=args.prompt,
        output_path=args.output,
        model_name=args.model,
        steps=args.steps,
        width=args.width,
        height=args.height,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
    )
    print(f"Image saved to: {path}")


if __name__ == "__main__":
    main()
