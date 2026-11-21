
"""Lab 08: compare Python-loop and NumPy ranking implementations."""

import math
import statistics
import tempfile
import time
from pathlib import Path

from findex.index import InvertedIndex
from findex.scoring import BM25Scorer, TfIdfScorer


def python_score(index, terms, matching_docs, method, top_k=10):
    """Reference implementation using ordinary Python loops."""
    ids = list(dict.fromkeys(int(i) for i in matching_docs))
    n = len(index.metadata)

    if not n or not ids:
        return []

    scores = {doc_id: 0.0 for doc_id in ids}
    avgdl = max(
        sum(meta.length for meta in index.metadata.values()) / n,
        1.0,
    )

    for term in terms:
        postings = index.index.get(term)
        if postings is None:
            continue

        doc_ids = postings["doc_ids"]
        frequencies = postings["tfs"]
        df = len(doc_ids)

        if not df:
            continue

        for doc_id, frequency in zip(doc_ids, frequencies):
            doc_id = int(doc_id)
            if doc_id not in scores:
                continue

            tf = float(frequency)
            length = float(index.metadata[doc_id].length)

            if method == "tfidf":
                idf = math.log((n + 1) / (df + 0.5)) + 1.0
                value = tf / max(length, 1.0) * idf
            else:
                idf = math.log(
                    (n - df + 0.5) / (df + 0.5) + 1.0
                )
                denominator = tf + 1.5 * (
                    1.0 - 0.75 + 0.75 * length / avgdl
                )
                value = (
                    idf * tf * 2.5 / denominator
                    if denominator > 0
                    else 0.0
                )

            scores[doc_id] += value

    # Python's sort is stable: equal scores keep input order.
    ranked = sorted(
        ids,
        key=lambda doc_id: -round(scores[doc_id], 4),
    )

    return [
        (doc_id, round(scores[doc_id], 4))
        for doc_id in ranked[:top_k]
    ]


def measure(function, repetitions=100):
    """Measure median execution time in milliseconds."""
    for _ in range(10):
        function()

    measurements = []

    for _ in range(repetitions):
        start = time.perf_counter_ns()
        function()
        elapsed = time.perf_counter_ns() - start
        measurements.append(elapsed / 1_000_000)

    return statistics.median(measurements)


def create_corpus(folder, count=500):
    """Create a reproducible corpus with frequent and rare terms."""
    for i in range(count):
        words = ["python"] * (i % 4 + 1)
        words += ["search", "engine", "document"]

        if i % 5 == 0:
            words += ["crawler", "async"]

        if i % 10 == 0:
            words += ["benchmark"]

        if i == 17:
            words += ["uniquerareterm"]

        (folder / f"doc_{i:04d}.txt").write_text(
            " ".join(words),
            encoding="utf-8",
        )


def main():
    with tempfile.TemporaryDirectory() as temp:
        corpus_path = Path(temp)
        create_corpus(corpus_path)

        index = InvertedIndex()
        index.build(str(corpus_path))

        ids = list(index.metadata)

        queries = {
            "Frequent": ["python"],
            "Rare": ["uniquerareterm"],
            "Three terms": ["async", "crawler", "benchmark"],
        }

        scorers = {
            "BM25": (BM25Scorer(), "bm25"),
            "TF-IDF": (TfIdfScorer(), "tfidf"),
        }

        rows = []

        print(f"Documents: {len(ids)}")
        print(f"Unique terms: {len(index.index)}")
        print()

        print(
            "| Algorithm | Query | Python (ms) | "
            "NumPy (ms) | Speedup |"
        )
        print("|---|---|---:|---:|---:|")

        for scorer_name, (scorer, method) in scorers.items():
            for query_name, terms in queries.items():

                def old_run():
                    return python_score(
                        index, terms, ids, method, top_k=10
                    )

                def new_run():
                    return scorer.score(
                        index, terms, ids, top_k=10
                    )

                old_results = old_run()
                new_results = [
                    (r.doc_id, r.score)
                    for r in new_run()
                ]

                if len(old_results) != len(new_results):
                    raise AssertionError(
                        f"Different result counts: "
                        f"{scorer_name}, {query_name}"
                    )

                for old, new in zip(old_results, new_results):
                    if old[0] != new[0] or not math.isclose(
                        old[1], new[1], abs_tol=0.0001
                    ):
                        raise AssertionError(
                            f"Different ranking: "
                            f"{scorer_name}, {query_name}: "
                            f"{old} != {new}"
                        )

                python_ms = measure(old_run)
                numpy_ms = measure(new_run)

                speedup = (
                    python_ms / numpy_ms
                    if numpy_ms > 0
                    else float("inf")
                )

                row = (
                    f"| {scorer_name} | {query_name} | "
                    f"{python_ms:.4f} | {numpy_ms:.4f} | "
                    f"{speedup:.2f}x |"
                )

                rows.append(row)
                print(row)

        output = Path("docs/lab08_comparison.md")
        output.parent.mkdir(parents=True, exist_ok=True)

        content = (
            "# Lab 08: Python loops vs NumPy\n\n"
            "Synthetic corpus: 500 documents.\n\n"
            "Median of 100 timed runs after 10 warm-ups. "
            "Times are in milliseconds.\n\n"
            "This compares a Python-loop reference implementation "
            "with the current NumPy implementation. "
            "It is not a measurement of an earlier Git revision.\n\n"
            "| Algorithm | Query | Python (ms) | "
            "NumPy (ms) | Speedup |\n"
            "|---|---|---:|---:|---:|\n"
            + "\n".join(rows)
            + "\n"
        )

        output.write_text(content, encoding="utf-8")
        print()
        print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
