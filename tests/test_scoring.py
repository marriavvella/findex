
import pytest

from findex.index import InvertedIndex
from findex.scoring import (
    BM25Scorer,
    TfIdfScorer,
    get_scorer,
)


@pytest.fixture
def search_index(tmp_path):
    # Перший документ містить більше входжень python
    (tmp_path / "doc1.txt").write_text(
        "python python python search",
        encoding="utf-8"
    )

    # Другий документ містить одне входження python
    (tmp_path / "doc2.txt").write_text(
        "python search example",
        encoding="utf-8"
    )

    # Третій документ не містить python
    (tmp_path / "doc3.txt").write_text(
        "database storage example",
        encoding="utf-8"
    )

    index = InvertedIndex()
    index.build(str(tmp_path))

    return index


def test_tfidf_ranking(search_index):
    scorer = TfIdfScorer()

    results = scorer.score(
        search_index,
        ["python"],
        [1, 2]
    )

    assert len(results) == 2
    assert results[0].doc_id == 1
    assert results[0].score > results[1].score


def test_bm25_ranking(search_index):
    scorer = BM25Scorer()

    results = scorer.score(
        search_index,
        ["python"],
        [1, 2]
    )

    assert len(results) == 2
    assert results[0].doc_id == 1
    assert results[0].score > results[1].score


def test_scorer_factory():
    assert isinstance(get_scorer("bm25"), BM25Scorer)
    assert isinstance(get_scorer("tfidf"), TfIdfScorer)
    assert isinstance(get_scorer("tf-idf"), TfIdfScorer)


def test_unknown_scorer():
    with pytest.raises(ValueError):
        get_scorer("unknown")


def test_empty_results(search_index):
    scorer = BM25Scorer()

    results = scorer.score(
        search_index,
        ["nonexistent"],
        []
    )

    assert results == []
