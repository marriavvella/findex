
import unicodedata

from hypothesis import given, strategies as st

from findex.tokenize import tokenize


@given(st.text(alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"))
def test_tokenize_case_insensitive(text):
    """Токенізація не залежить від регістру."""

    assert list(tokenize(text)) == list(
        tokenize(text.swapcase())
    )


@given(
    st.text(
        alphabet=[
            "a", "e", "i", "o", "u",
            "c", "f", "n", "r",
            "é", "á", "ü",
            "\u0301", "\u0308",
            " ", "'", "!",
        ]
    )
)
def test_tokenize_unicode_normalization(text):
    """Перевіряє однакову обробку Unicode в NFC."""

    normalized = unicodedata.normalize("NFC", text)

    assert list(tokenize(text)) == list(
        tokenize(normalized)
    )


@given(
    st.text(
        alphabet=(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "абвгдеєжзиіїйклмнопрстуфхцчшщьюяґ"
            "АБВГДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯҐ"
            "0123456789 .,!?'-"
        )
    )
)
def test_tokenize_idempotence(text):
    """Повторна токенізація не змінює результат."""

    first_result = list(tokenize(text))

    second_result = list(
        tokenize(" ".join(first_result))
    )

    assert first_result == second_result
