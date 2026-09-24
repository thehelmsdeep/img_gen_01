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
        default="sd15",
        choices=sorted(MODELS),
        help="Registered model name",
    )
    parser.add_argument("--output", default="outputs/image.png")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    path = generate(
        prompt=args.prompt,
        output_path=args.output,
        model_name=args.model,
        steps=args.steps,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    print(f"Image saved to: {path}")


if __name__ == "__main__":
    main()
