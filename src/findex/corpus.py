from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple


class Document(NamedTuple):
    doc_id: str
    path: Path
    text: str

def iter_documents(root: Path) -> Iterator[Document]:
    """Генератор, який читає текстові файли (.txt) по одному з папки root."""
    for path in root.rglob("*.txt"):
        try:
            text = path.read_text(encoding="utf-8")
            yield Document(doc_id=path.stem, path=path, text=text)
        except (OSError, UnicodeDecodeError) as e:
            # Логуємо конкретні помилки читання файлу чи кодування
            print(f"Помилка читання файлу {path}: {e}")