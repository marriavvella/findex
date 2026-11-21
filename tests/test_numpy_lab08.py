"""Check NumPy representation and deterministic ranking."""
import numpy as np
import pytest
from findex.index import InvertedIndex
from findex.scoring import BM25Scorer, TfIdfScorer


@pytest.mark.parametrize("scorer", [BM25Scorer(), TfIdfScorer()])
def test_numpy_ranking(scorer, tmp_path):
    for i, content in enumerate(["python python search", "python search", "engine storage"], 1):
        (tmp_path / f"doc{i}.txt").write_text(content, encoding="utf-8")
    idx = InvertedIndex()
    idx.build(str(tmp_path))
    assert idx.doc_lengths.dtype == np.int32
    assert idx.index["python"]["doc_ids"].dtype == np.int32
    assert idx.index["python"]["tfs"].dtype == np.int32
    results = scorer.score(idx, ["python"], [1, 2])
    assert [r.doc_id for r in results] == [1, 2]
    assert all(isinstance(r.score, float) for r in results)
