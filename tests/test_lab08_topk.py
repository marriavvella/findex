import pytest
from findex.index import InvertedIndex
from findex.scoring import BM25Scorer, TfIdfScorer

@pytest.mark.parametrize("scorer", [BM25Scorer(), TfIdfScorer()])
def test_topk_same_as_full(scorer, tmp_path):
    for i in range(15):
        (tmp_path / f"{i}.txt").write_text(("python " * (i % 4 + 1)) + "search " + ("index " * (i % 3 + 1)), encoding="utf-8")
    idx = InvertedIndex()
    idx.build(str(tmp_path))
    ids = list(idx.metadata)
    for k in (1, 2, 5, 10, 15):
        full = scorer.score(idx, ["python", "search", "index"], ids)[:k]
        partial = scorer.score(idx, ["python", "search", "index"], ids, top_k=k)
        assert [(r.doc_id, r.score) for r in partial] == [(r.doc_id, r.score) for r in full]
