import re
import unicodedata
from collections.abc import Iterator

# Регулярний вираз для пошуку слів (підтримує латиницю, кирилицю та апострофи)
WORD_RE = re.compile(r"[a-zA-Zа-яА-ЯіІїЇєЄґҐ']+")


def tokenize(text: str) -> Iterator[str]:
    """Потоковий токенізатор із нормалізацією Unicode NFC та casefold."""
    # 1. Unicode NFC нормалізація для зведення різних варіантів символів до єдиного формату
    normalized = unicodedata.normalize("NFC", text)

    # 2. Casefold для надійного порівняння слів незалежно від регістру
    folded = normalized.casefold()

    # 3. Ледачий пошук слів через finditer замість завантаження всього списку в пам'ять
    for match in WORD_RE.finditer(folded):
        token = match.group().strip("'")
        if token:
            yield token
