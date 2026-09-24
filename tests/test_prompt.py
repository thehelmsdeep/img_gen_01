import pytest

from src.prompt.processor import prepare_prompt


def test_prepare_prompt_normalizes_whitespace():
    result = prepare_prompt("  a   red   car  ")
    assert result.text == "a red car"


def test_prepare_prompt_keeps_negative_prompt():
    result = prepare_prompt("a red car", "  blurry   distorted  ")
    assert result.text == "a red car"
    assert result.negative == "blurry distorted"


def test_prepare_prompt_rejects_empty_prompt():
    with pytest.raises(ValueError):
        prepare_prompt("   ")
