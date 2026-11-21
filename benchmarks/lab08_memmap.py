"""10x JSONL corpus and NumPy memmap experiment. Run: uv run python benchmarks/lab08_memmap.py"""
import json
import time
from pathlib import Path
import numpy as np
from findex.index import InvertedIndex

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)
source = ROOT / "corpus.jsonl"
if not source.exists():
    raise SystemExit("corpus.jsonl missing")
records = [json.loads(line) for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
if not records:
    raise SystemExit("corpus.jsonl empty")
expanded = DOCS / "corpus_10x.jsonl"
with expanded.open("w", encoding="utf-8") as output:
    for copy in range(10):
        for record in records:
            row = dict(record)
            row["doc_id"] = f"{copy}_{record.get('doc_id', '')}"
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
start = time.perf_counter()
index = InvertedIndex()
index.build(str(expanded))
build_ms = (time.perf_counter() - start) * 1000
terms = sorted(index.index)
ids = np.concatenate([index.index[t]["doc_ids"] for t in terms]).astype(np.int32)
tfs = np.concatenate([index.index[t]["tfs"] for t in terms]).astype(np.int32)
packed = np.column_stack((ids, tfs))
path = DOCS / "postings_10x.npy"
np.save(path, packed)
start = time.perf_counter()
mapped = np.load(path, mmap_mode="r")
load_ms = (time.perf_counter() - start) * 1000
assert isinstance(mapped, np.memmap)
assert np.array_equal(mapped, packed)
report = f"10x corpus documents: {len(index.metadata)}\nTerms: {len(terms)}\nBuild time ms: {build_ms:.3f}\nMemmap open ms: {load_ms:.3f}\nNumpy file bytes: {path.stat().st_size}\nCorpus bytes: {expanded.stat().st_size}\nNote: memmap is a separate benchmark artifact; main CLI still uses pickle.\n"
(DOCS / "memmap_results.txt").write_text(report, encoding="utf-8")
print(report)
