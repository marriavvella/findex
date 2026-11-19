
import pytest

from findex.index import InvertedIndex
from findex.search import execute_search
from findex.scoring import BM25Scorer, TfIdfScorer


@pytest.fixture
def sample_index(tmp_path):
    """Створює невеликий тестовий індекс."""

    (tmp_path / "doc1.txt").write_text(
        "apple banana apple garden",
        encoding="utf-8",
    )

    (tmp_path / "doc2.txt").write_text(
        "banana orange garden",
        encoding="utf-8",
    )

    (tmp_path / "doc3.txt").write_text(
        "apple orange fruit",
        encoding="utf-8",
    )

    index = InvertedIndex()
    index.build(str(tmp_path))

    return index


def test_search_with_bm25(sample_index):
    """Перевіряє пошук із ранжуванням BM25."""

    results = execute_search(
        sample_index,
        "apple",
        BM25Scorer(),
        top_k=5,
    )

    assert len(results) == 2
    assert all(result.doc_id in (1, 3) for result in results)
    assert all(result.score >= 0 for result in results)


def test_search_with_tfidf(sample_index):
    """Перевіряє пошук із ранжуванням TF-IDF."""

    results = execute_search(
        sample_index,
        "apple",
        TfIdfScorer(),
        top_k=5,
    )

    assert len(results) == 2
    assert all(result.doc_id in (1, 3) for result in results)


def test_search_top_k(sample_index):
    """Перевіряє обмеження кількості результатів."""

    results = execute_search(
        sample_index,
        "apple OR banana",
        BM25Scorer(),
        top_k=2,
    )

    assert len(results) == 2


def test_search_empty_results(sample_index):
    """Перевіряє запит без знайдених документів."""

    results = execute_search(
        sample_index,
        "unexistingword",
        BM25Scorer(),
    )

    assert results == []


def test_search_invalid_top_k(sample_index):
    """Перевіряє обробку неправильного top_k."""

    with pytest.raises(ValueError):
        execute_search(
            sample_index,
            "apple",
            BM25Scorer(),
            top_k=0,
        )


def test_search_snippets(sample_index):
    """Перевіряє формування сніпетів із підсвічуванням."""

    results = execute_search(
        sample_index,
        "apple",
        BM25Scorer(),
    )

    assert len(results) == 2

    for result in results:
        assert result.snippet
        assert "**apple**" in result.snippet.lower()


def test_search_result_fields(sample_index):
    """Перевіряє структуру результатів пошуку."""

    results = execute_search(
        sample_index,
        "banana",
        BM25Scorer(),
    )

    assert len(results) == 2

    for result in results:
        assert isinstance(result.doc_id, int)
        assert isinstance(result.filename, str)
        assert isinstance(result.score, float)
        assert isinstance(result.snippet, str)
