
import numpy as np
import pytest

from findex.index import (
    InvertedIndex,
    open_index,
    build_partial,
    merge,
    build_index_parallel,
)
from findex.models import DocMeta


def test_inverted_index_build(tmp_path):
    (tmp_path / "doc1.txt").write_text(
        "apple banana apple", encoding="utf-8"
    )
    (tmp_path / "doc2.txt").write_text(
        "banana orange", encoding="utf-8"
    )

    idx = InvertedIndex()
    idx.build(str(tmp_path))

    assert len(idx.metadata) == 2
    assert len(idx.index) == 3

    np.testing.assert_array_equal(
        idx.index["apple"]["doc_ids"],
        np.array([1], dtype=np.int32)
    )
    np.testing.assert_array_equal(
        idx.index["apple"]["tfs"],
        np.array([2], dtype=np.int32)
    )

    np.testing.assert_array_equal(
        idx.index["banana"]["doc_ids"],
        np.array([1, 2], dtype=np.int32)
    )


def test_pickle_roundtrip(tmp_path):
    (tmp_path / "doc.txt").write_text(
        "apple banana apple", encoding="utf-8"
    )

    idx = InvertedIndex()
    idx.build(str(tmp_path))

    filepath = tmp_path / "index.pkl"
    idx.save_pickle(str(filepath))

    restored = InvertedIndex.load_pickle(str(filepath))

    assert restored.metadata == idx.metadata
    assert set(restored.index) == set(idx.index)

    np.testing.assert_array_equal(
        restored.index["apple"]["tfs"],
        idx.index["apple"]["tfs"]
    )


def test_json_roundtrip(tmp_path):
    (tmp_path / "doc.txt").write_text(
        "apple banana apple", encoding="utf-8"
    )

    idx = InvertedIndex()
    idx.build(str(tmp_path))

    filepath = tmp_path / "index.json"
    idx.save_json(str(filepath))

    restored = InvertedIndex.load_json(str(filepath))

    assert restored.metadata == idx.metadata
    assert set(restored.index) == set(idx.index)

    np.testing.assert_array_equal(
        restored.index["apple"]["doc_ids"],
        idx.index["apple"]["doc_ids"]
    )


# Перевірка спеціальних методів об'єкта

def test_index_object_methods(tmp_path):
    (tmp_path / "doc.txt").write_text(
        "apple banana apple", encoding="utf-8"
    )

    idx = InvertedIndex()
    idx.build(str(tmp_path))

    assert len(idx) == 2
    assert "apple" in idx
    assert "unknown" not in idx

    assert idx["apple"]["doc_ids"].tolist() == [1]
    assert set(iter(idx)) == {"apple", "banana"}

    assert repr(idx) == "InvertedIndex(documents=1, terms=2)"


def test_index_context_manager():
    idx = InvertedIndex()

    with idx as opened:
        assert opened is idx


def test_index_context_manager_does_not_suppress_errors():
    idx = InvertedIndex()

    with pytest.raises(ValueError):
        with idx:
            raise ValueError("Test exception")


# Перевірка статистики індексу

def test_document_statistics(tmp_path):
    (tmp_path / "doc1.txt").write_text(
        "apple banana apple", encoding="utf-8"
    )
    (tmp_path / "doc2.txt").write_text(
        "banana orange", encoding="utf-8"
    )

    idx = InvertedIndex()
    idx.build(str(tmp_path))

    assert idx.num_docs == 2
    assert idx.avg_doc_length == 2.5

    assert idx.doc_length(1) == 3
    assert idx.doc_length(2) == 2

    assert idx.df("apple") == 1
    assert idx.df("banana") == 2
    assert idx.df("unknown") == 0

    with pytest.raises(KeyError):
        idx.doc_length(100)


def test_empty_index_statistics():
    idx = InvertedIndex()

    assert idx.num_docs == 0
    assert idx.avg_doc_length == 0.0
    assert len(idx) == 0
    assert idx.df("apple") == 0


def test_build_resets_previous_index(tmp_path):
    (tmp_path / "first.txt").write_text(
        "apple banana", encoding="utf-8"
    )

    idx = InvertedIndex()
    idx.build(str(tmp_path))

    assert idx.avg_doc_length == 2

    (tmp_path / "first.txt").unlink()

    (tmp_path / "second.txt").write_text(
        "orange orange orange", encoding="utf-8"
    )

    idx.build(str(tmp_path))

    assert idx.num_docs == 1
    assert set(idx.index) == {"orange"}
    assert idx.avg_doc_length == 3
    assert idx.doc_length(1) == 3


# Перевірка відкриття індексу

def test_open_index_pickle_and_json(tmp_path):
    (tmp_path / "doc.txt").write_text(
        "apple banana apple", encoding="utf-8"
    )

    idx = InvertedIndex()
    idx.build(str(tmp_path))

    pickle_path = tmp_path / "index.pkl"
    json_path = tmp_path / "index.json"

    idx.save_pickle(str(pickle_path))
    idx.save_json(str(json_path))

    pickle_index = open_index(str(pickle_path))
    json_index = open_index(str(json_path))

    assert pickle_index.num_docs == 1
    assert json_index.num_docs == 1

    assert set(pickle_index.index) == {"apple", "banana"}
    assert set(json_index.index) == {"apple", "banana"}


# Перевірка часткової індексації

def test_build_partial(tmp_path):
    doc1 = tmp_path / "doc1.txt"
    doc2 = tmp_path / "doc2.txt"

    doc1.write_text("apple banana apple", encoding="utf-8")
    doc2.write_text("banana orange", encoding="utf-8")

    partial_index, metadata = build_partial(
        [str(doc1), str(doc2)],
        start_doc_id=5
    )

    assert set(metadata) == {5, 6}
    assert metadata[5].length == 3
    assert metadata[6].length == 2

    assert partial_index["apple"]["doc_ids"] == [5]
    assert partial_index["apple"]["tfs"] == [2]

    assert partial_index["banana"]["doc_ids"] == [5, 6]


# Перевірка об'єднання часткових індексів

def test_merge_partial_indexes():
    partial1 = (
        {
            "apple": {
                "doc_ids": [1],
                "tfs": [2]
            }
        },
        {
            1: DocMeta(
                doc_id=1,
                filename="doc1.txt",
                length=3
            )
        }
    )

    partial2 = (
        {
            "apple": {
                "doc_ids": [2],
                "tfs": [1]
            },
            "banana": {
                "doc_ids": [2],
                "tfs": [3]
            }
        },
        {
            2: DocMeta(
                doc_id=2,
                filename="doc2.txt",
                length=4
            )
        }
    )

    merged_index, merged_metadata = merge(
        [partial1, partial2]
    )

    assert set(merged_metadata) == {1, 2}

    np.testing.assert_array_equal(
        merged_index["apple"]["doc_ids"],
        np.array([1, 2], dtype=np.int32)
    )

    np.testing.assert_array_equal(
        merged_index["apple"]["tfs"],
        np.array([2, 1], dtype=np.int32)
    )

    np.testing.assert_array_equal(
        merged_index["banana"]["doc_ids"],
        np.array([2], dtype=np.int32)
    )


def test_merge_empty_partials():
    merged_index, merged_metadata = merge([])

    assert merged_index == {}
    assert merged_metadata == {}


# Перевірка послідовної побудови індексу

def test_build_index_parallel_serial(tmp_path):
    (tmp_path / "doc1.txt").write_text(
        "apple banana", encoding="utf-8"
    )
    (tmp_path / "doc2.txt").write_text(
        "banana orange", encoding="utf-8"
    )

    idx = build_index_parallel(
        str(tmp_path),
        workers=1,
        executor_type="serial"
    )

    assert idx.num_docs == 2
    assert set(idx.index) == {"apple", "banana", "orange"}

    assert idx.doc_length(1) == 2
    assert idx.doc_length(2) == 2


def test_build_index_parallel_threads(tmp_path):
    (tmp_path / "doc1.txt").write_text(
        "apple banana", encoding="utf-8"
    )
    (tmp_path / "doc2.txt").write_text(
        "banana orange", encoding="utf-8"
    )

    idx = build_index_parallel(
        str(tmp_path),
        workers=2,
        executor_type="threads"
    )

    assert idx.num_docs == 2
    assert set(idx.index) == {"apple", "banana", "orange"}

    np.testing.assert_array_equal(
        idx.index["banana"]["doc_ids"],
        np.array([1, 2], dtype=np.int32)
    )


def test_build_index_parallel_empty_directory(tmp_path):
    idx = build_index_parallel(
        str(tmp_path),
        workers=2,
        executor_type="serial"
    )

    assert idx.num_docs == 0
    assert len(idx.index) == 0


def test_build_index_parallel_invalid_workers(tmp_path):
    with pytest.raises(ValueError, match="workers must be >= 1"):
        build_index_parallel(
            str(tmp_path),
            workers=0
        )


def test_build_index_parallel_invalid_executor(tmp_path):
    (tmp_path / "doc.txt").write_text(
        "apple banana", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Unknown executor type"):
        build_index_parallel(
            str(tmp_path),
            workers=1,
            executor_type="invalid"
        )
