
import numpy as np

from findex.index import InvertedIndex


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
