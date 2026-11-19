
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple


class Document(NamedTuple):
    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """Генератор, який читає текстові файли (.txt) у стабільному порядку."""

    for path in sorted(root.rglob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")

            yield Document(
                doc_id=path.stem,
                path=path,
                text=text,
            )

        except (OSError, UnicodeDecodeError) as e:
            # Логуємо помилки читання файлу або кодування.
            print(f"Помилка читання файлу {path}: {e}")
