
import math
import heapq
from typing import Protocol, List, Dict, Sequence, Literal
from findex.index import Index
from findex.models import SearchResult

ScorerType = Literal["tf-idf", "tfidf", "bm25"]


class Scorer(Protocol):
    def score(
        self,
        index: Index,
        query_terms: Sequence[str],
        matching_docs: Sequence[int],
    ) -> List[SearchResult]:
        ...


class TfIdfScorer:
    def score(
        self,
        index: Index,
        query_terms: Sequence[str],
        matching_docs: Sequence[int],
    ) -> List[SearchResult]:

        scores: Dict[int, float] = {
            int(doc_id): 0.0 for doc_id in matching_docs
        }

        N = len(index.metadata)

        if N == 0:
            return []

        for term in query_terms:
            postings = index.index.get(term)

            if postings is None:
                continue

            doc_ids = postings["doc_ids"]
            tfs = postings["tfs"]

            df = len(doc_ids)

            if df == 0:
                continue

            idf = math.log((N + 1) / (df + 0.5)) + 1.0

            for doc_id, tf_value in zip(doc_ids, tfs):
                doc_id = int(doc_id)

                if doc_id in scores:
                    doc_len = index.metadata[doc_id].length
                    tf = int(tf_value) / max(doc_len, 1)
                    scores[doc_id] += tf * idf

        results = [
            SearchResult(
                score=round(score, 4),
                doc_id=doc_id,
                filename=index.metadata[doc_id].filename,
                snippet="",
            )
            for doc_id, score in scores.items()
        ]

        return heapq.nlargest(
            len(results),
            results,
            key=lambda result: result.score,
        )


class BM25Scorer:
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b

    def score(
        self,
        index: Index,
        query_terms: Sequence[str],
        matching_docs: Sequence[int],
    ) -> List[SearchResult]:

        scores: Dict[int, float] = {
            int(doc_id): 0.0 for doc_id in matching_docs
        }

        N = len(index.metadata)

        if N == 0:
            return []

        avgdl = (
            sum(meta.length for meta in index.metadata.values()) / N
        )

        avgdl = max(avgdl, 1.0)

        for term in query_terms:
            postings = index.index.get(term)

            if postings is None:
                continue

            doc_ids = postings["doc_ids"]
            tfs = postings["tfs"]

            df = len(doc_ids)

            if df == 0:
                continue

            idf = math.log(
                (N - df + 0.5) / (df + 0.5) + 1.0
            )

            for doc_id, tf_value in zip(doc_ids, tfs):
                doc_id = int(doc_id)

                if doc_id in scores:
                    doc_len = index.metadata[doc_id].length
                    tf = int(tf_value)

                    numerator = tf * (self.k1 + 1)

                    denominator = tf + self.k1 * (
                        1 - self.b
                        + self.b * (doc_len / avgdl)
                    )

                    if denominator > 0:
                        scores[doc_id] += (
                            idf * numerator / denominator
                        )

        results = [
            SearchResult(
                score=round(score, 4),
                doc_id=doc_id,
                filename=index.metadata[doc_id].filename,
                snippet="",
            )
            for doc_id, score in scores.items()
        ]

        return heapq.nlargest(
            len(results),
            results,
            key=lambda result: result.score,
        )


def get_scorer(scorer_type: ScorerType) -> Scorer:
    if scorer_type in ("tf-idf", "tfidf"):
        return TfIdfScorer()

    if scorer_type == "bm25":
        return BM25Scorer()

    raise ValueError(
        f"Unknown scorer algorithm: {scorer_type}"
    )
