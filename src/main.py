import argparse

from src.generation.text_to_image import generate


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Local text-to-image generation"
    )
    parser.add_argument("prompt", help="Text prompt for the image")
    parser.add_argument(
        "--output",
        default="outputs/image.png",
        help="Output PNG path",
    )
    args = parser.parse_args()

    path = generate(args.prompt, args.output)
    print(f"Image saved to: {path}")


if __name__ == "__main__":
    main()
