
import pytest

from findex.index import InvertedIndex


def test_object_model(tmp_path):
    # Створюємо тестовий документ
    (tmp_path / "doc1.txt").write_text(
        "python search python",
        encoding="utf-8"
    )

    index = InvertedIndex()
    index.build(str(tmp_path))

    # Перевірка __len__
    assert len(index) == 2

    # Перевірка __contains__
    assert "python" in index
    assert "search" in index
    assert "missing" not in index

    # Перевірка __getitem__
    postings = index["python"]
    assert postings["doc_ids"].tolist() == [1]
    assert postings["tfs"].tolist() == [2]

    # Перевірка __iter__
    assert set(iter(index)) == {"python", "search"}

    # Перевірка __repr__
    assert repr(index) == (
        "InvertedIndex(documents=1, terms=2)"
    )

    # Перевірка помилки для відсутнього терміна
    with pytest.raises(KeyError):
        index["missing"]
