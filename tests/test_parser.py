
from pathlib import Path

from findex.index import InvertedIndex
from findex.query_parser import (
    And,
    Or,
    Phrase,
    Term,
    parse_query,
)


def test_ast_parsing():
    q1 = parse_query('python AND (async OR await)')
    expected1 = And(
        Term('python'),
        Or(Term('async'), Term('await'))
    )
    assert q1 == expected1, (
        f"Expected {expected1}, got {q1}"
    )

    q2 = parse_query('"event loop"')
    expected2 = Phrase('event loop')
    assert q2 == expected2, (
        f"Expected {expected2}, got {q2}"
    )


def test_exact_phrase_search(tmp_path):
    # Створюємо три тестові документи
    (tmp_path / "doc1.txt").write_text(
        "alice in wonderland",
        encoding="utf-8"
    )

    (tmp_path / "doc2.txt").write_text(
        "alice was walking in wonderland",
        encoding="utf-8"
    )

    (tmp_path / "doc3.txt").write_text(
        "wonderland in alice",
        encoding="utf-8"
    )

    # Будуємо інвертований індекс
    index = InvertedIndex()
    index.build(str(tmp_path))

    # Шукаємо точну фразу
    result = parse_query(
        '"alice in wonderland"'
    ).evaluate(index)

    # Отримуємо назви знайдених документів
    matching_files = {
        Path(index.metadata[doc_id].filename).name
        for doc_id in result
    }

    # Фраза має бути лише в першому документі
    assert matching_files == {"doc1.txt"}


def test_phrase_with_boolean_and(tmp_path):
    # Створюємо два тестові документи
    (tmp_path / "doc1.txt").write_text(
        "python event loop asyncio",
        encoding="utf-8"
    )

    (tmp_path / "doc2.txt").write_text(
        "python event driven loop",
        encoding="utf-8"
    )

    # Будуємо інвертований індекс
    index = InvertedIndex()
    index.build(str(tmp_path))

    # Поєднуємо AND із пошуком точної фрази
    result = parse_query(
        'python AND "event loop"'
    ).evaluate(index)

    matching_files = {
        Path(index.metadata[doc_id].filename).name
        for doc_id in result
    }

    # Лише перший документ містить точну фразу
    assert matching_files == {"doc1.txt"}


if __name__ == "__main__":
    test_ast_parsing()
    print("All AST Parser tests passed successfully!")
