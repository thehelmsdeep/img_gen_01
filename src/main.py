import argparse

from src.generation.text_to_image import generate
from src.models.stable_diffusion import load_pipeline, manual_generate
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
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--manual", action="store_true", help="Run the educational manual pipeline")
    args = parser.parse_args()

    if args.manual:
        model = MODELS[args.model]
        pipe = load_pipeline(model.model_id, cache_dir="models")
        image = manual_generate(
            pipe=pipe,
            prompt=args.prompt,
            steps=args.steps,
            seed=args.seed,
            negative_prompt=args.negative_prompt,
            guidance_scale=args.guidance_scale,
        )
        from PIL import Image
        path = args.output
        Image.fromarray((image[0] * 255).astype("uint8")).save(path)
        print(f"Image saved to: {path}")
        return

    path = generate(
        prompt=args.prompt,
        output_path=args.output,
        model_name=args.model,
        steps=args.steps,
        width=args.width,
        height=args.height,
        seed=args.seed,
        negative_prompt=args.negative_prompt,
    )
    print(f"Image saved to: {path}")


if __name__ == "__main__":
    main()
