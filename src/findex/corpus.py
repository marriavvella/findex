
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple, cast

logger = logging.getLogger(__name__)


class Document(NamedTuple):
    doc_id: str
    path: Path
    text: str


def iter_documents(root: Path) -> Iterator[Document]:
    """Read TXT and JSONL documents lazily in a stable order."""

    if root.is_file():
        if root.suffix.lower() == ".jsonl":
            yield from _iter_jsonl(root)
        elif root.suffix.lower() == ".txt":
            yield from _iter_txt(root)
        return

    if not root.is_dir():
        logger.warning("Corpus path does not exist: %s", root)
        return

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() == ".txt":
            yield from _iter_txt(path)

        elif path.suffix.lower() == ".jsonl":
            yield from _iter_jsonl(path)


def _iter_txt(path: Path) -> Iterator[Document]:
    """Read a text document."""

    try:
        text = path.read_text(encoding="utf-8")

        yield Document(
            doc_id=path.stem,
            path=path,
            text=text,
        )

    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(
            "Cannot read text file %s: %s",
            path,
            exc,
        )


def _iter_jsonl(path: Path) -> Iterator[Document]:
    """Read JSONL documents one line at a time."""

    try:
        with path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    record = cast(object, json.loads(line))

                    if not isinstance(record, dict):
                        raise TypeError(
                            "JSONL record must be an object"
                        )

                    data = cast(dict[str, object], record)

                    doc_id = data.get("doc_id")
                    text = data.get("text")

                    if doc_id is None or not isinstance(text, str):
                        raise TypeError(
                            "JSONL record requires doc_id and text"
                        )

                    yield Document(
                        doc_id=str(doc_id),
                        path=path,
                        text=text,
                    )

                except (
                    json.JSONDecodeError,
                    ValueError,
                    TypeError,
                ) as exc:
                    logger.warning(
                        "Invalid JSONL record in %s at line %d: %s",
                        path,
                        line_number,
                        exc,
                    )

    except (OSError, UnicodeDecodeError) as exc:
        logger.warning(
            "Cannot read JSONL file %s: %s",
            path,
            exc,
        )
