from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    text: str
    negative: str | None = None


def prepare_prompt(text: str, negative: str | None = None) -> Prompt:
    text = " ".join(text.split()).strip()

    if not text:
        raise ValueError("Prompt cannot be empty.")

    negative = " ".join(negative.split()).strip() if negative else None

    return Prompt(text=text, negative=negative)
