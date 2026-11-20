"""Tests for serial and parallel inverted-index construction."""

from pathlib import Path

import numpy as np
import pytest

from findex.index import InvertedIndex, build_index_parallel


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """Create a small corpus for testing."""
    documents = {
        "doc1.txt": "python search index python",
        "doc2.txt": "parallel python threads",
        "doc3.txt": "processes search index",
        "doc4.txt": "python multiprocessing index",
    }

    for filename, content in documents.items():
        (tmp_path / filename).write_text(
            content,
            encoding="utf-8",
        )

    return tmp_path


def assert_indexes_equal(
    first: InvertedIndex,
    second: InvertedIndex,
) -> None:
    """Compare metadata, terms, postings and document lengths."""
    assert first.metadata == second.metadata
    assert first.index.keys() == second.index.keys()

    np.testing.assert_array_equal(
        first.doc_lengths,
        second.doc_lengths,
    )

    for term in first.index:
        np.testing.assert_array_equal(
            first.index[term]["doc_ids"],
            second.index[term]["doc_ids"],
        )
        np.testing.assert_array_equal(
            first.index[term]["tfs"],
            second.index[term]["tfs"],
        )


@pytest.mark.parametrize(
    ("executor_type", "workers"),
    [
        ("serial", 1),
        ("serial", 2),
        ("threads", 1),
        ("threads", 2),
        ("processes", 1),
        ("processes", 2),
    ],
)
def test_parallel_matches_sequential(
    corpus_dir: Path,
    executor_type: str,
    workers: int,
) -> None:
    """All execution modes must produce the same index."""
    sequential = InvertedIndex()
    sequential.build(str(corpus_dir))

    parallel = build_index_parallel(
        str(corpus_dir),
        workers=workers,
        executor_type=executor_type,
    )

    assert_indexes_equal(sequential, parallel)


def test_postings_sorted(corpus_dir: Path) -> None:
    """Posting lists must be sorted by document ID."""
    index = build_index_parallel(
        str(corpus_dir),
        workers=2,
        executor_type="threads",
    )

    for posting in index.index.values():
        doc_ids = posting["doc_ids"]
        assert np.all(doc_ids[:-1] <= doc_ids[1:])


def test_unique_document_ids(corpus_dir: Path) -> None:
    """Every document must receive a unique ID."""
    index = build_index_parallel(
        str(corpus_dir),
        workers=4,
        executor_type="processes",
    )

    doc_ids = list(index.metadata)

    assert len(doc_ids) == len(set(doc_ids))
    assert len(doc_ids) == 4


def test_invalid_workers(corpus_dir: Path) -> None:
    """Worker count must be positive."""
    with pytest.raises(ValueError, match="workers"):
        build_index_parallel(
            str(corpus_dir),
            workers=0,
        )


def test_invalid_executor(corpus_dir: Path) -> None:
    """Unknown executor types must be rejected."""
    with pytest.raises(ValueError, match="executor"):
        build_index_parallel(
            str(corpus_dir),
            workers=2,
            executor_type="unknown",
        )