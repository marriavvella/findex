"""pytest-benchmark checks used in CI."""
from findex.index import InvertedIndex
from findex.scoring import BM25Scorer


def test_build_index(benchmark, tmp_path):
    (tmp_path / "sample.txt").write_text("python search engine python", encoding="utf-8")
    benchmark(lambda: InvertedIndex().build(str(tmp_path)))


def test_search(benchmark, tmp_path):
    (tmp_path / "sample.txt").write_text("python search engine python", encoding="utf-8")
    idx = InvertedIndex()
    idx.build(str(tmp_path))
    benchmark(lambda: BM25Scorer().score(idx, ["python", "search", "engine"], [1]))
