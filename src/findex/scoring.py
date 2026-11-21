
"""Vectorized BM25 and TF-IDF with stable legacy-compatible ranking."""

import math
from collections.abc import Sequence
from typing import Literal, Protocol

import numpy as np

from findex.index import Index
from findex.models import SearchResult


ScorerType = Literal["tf-idf", "tfidf", "bm25"]


class Scorer(Protocol):
    def score(
        self,
        index: Index,
        query_terms: Sequence[str],
        matching_docs: Sequence[int],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        ...


def _score(
    index: Index,
    query_terms: Sequence[str],
    matching_docs: Sequence[int],
    *,
    method: str,
    top_k: int | None = None,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[SearchResult]:
    """Calculate vectorized scores and return stable ranking."""

    ids = np.asarray(
        list(dict.fromkeys(int(i) for i in matching_docs)),
        dtype=np.int32,
    )

    n = len(index.metadata)

    if not n or len(ids) == 0:
        return []

    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be greater than zero")

    scores = np.zeros(len(ids), dtype=np.float64)

    # Sort IDs only for efficient vectorized membership checks.
    order = np.argsort(ids)
    sorted_ids = ids[order]

    lengths = index.doc_lengths

    if len(lengths) <= int(ids.max()):
        lengths = np.zeros(
            int(ids.max()) + 1,
            dtype=np.int32,
        )

        for doc_id, meta in index.metadata.items():
            lengths[doc_id] = meta.length

    avgdl = max(
        sum(meta.length for meta in index.metadata.values()) / n,
        1.0,
    )

    for term in query_terms:
        postings = index.index.get(term)

        if postings is None:
            continue

        posting_ids = np.asarray(
            postings["doc_ids"],
            dtype=np.int32,
        )

        tf = np.asarray(
            postings["tfs"],
            dtype=np.float64,
        )

        df = len(posting_ids)

        if df == 0:
            continue

        positions = np.searchsorted(
            sorted_ids,
            posting_ids,
        )

        in_bounds = positions < len(sorted_ids)

        valid = np.zeros(
            len(posting_ids),
            dtype=bool,
        )

        valid[in_bounds] = (
            sorted_ids[positions[in_bounds]]
            == posting_ids[in_bounds]
        )

        if not np.any(valid):
            continue

        doc_ids = posting_ids[valid]
        tf = tf[valid]

        dest = order[positions[valid]]

        doc_lengths = lengths[doc_ids].astype(
            np.float64
        )

        if method == "tfidf":
            idf = (
                math.log((n + 1) / (df + 0.5))
                + 1.0
            )

            values = (
                tf / np.maximum(doc_lengths, 1.0)
            ) * idf

        else:
            idf = math.log(
                (n - df + 0.5) / (df + 0.5) + 1.0
            )

            denominator = (
                tf
                + k1
                * (
                    1.0
                    - b
                    + b * doc_lengths / avgdl
                )
            )

            values = np.divide(
                idf * tf * (k1 + 1),
                denominator,
                out=np.zeros_like(tf),
                where=denominator > 0,
            )

        np.add.at(scores, dest, values)

    rounded = np.round(scores, 4)

    # Preserve the original input order for equal scores.
    if top_k is not None and top_k < len(rounded):

        # Use argpartition to determine the top-k threshold.
        partitioned = np.argpartition(
            -rounded,
            top_k - 1,
        )

        cutoff = rounded[
            partitioned[top_k - 1]
        ]

        # Include every document above the threshold.
        above = np.flatnonzero(
            rounded > cutoff
        )

        # Equal-score documents retain their original order.
        equal = np.flatnonzero(
            rounded == cutoff
        )

        remaining = top_k - len(above)

        candidates = np.concatenate(
            (
                above,
                equal[:remaining],
            )
        )

        # Stable sorting restores ranking and tie order.
        ranked = candidates[
            np.argsort(
                -rounded[candidates],
                kind="stable",
            )
        ]

    else:
        ranked = np.argsort(
            -rounded,
            kind="stable",
        )

    return [
        SearchResult(
            score=float(rounded[i]),
            doc_id=int(ids[i]),
            filename=index.metadata[int(ids[i])].filename,
            snippet="",
        )
        for i in ranked
    ]


class TfIdfScorer:
    def score(
        self,
        index: Index,
        query_terms: Sequence[str],
        matching_docs: Sequence[int],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        return _score(
            index,
            query_terms,
            matching_docs,
            method="tfidf",
            top_k=top_k,
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
        top_k: int | None = None,
    ) -> list[SearchResult]:
        return _score(
            index,
            query_terms,
            matching_docs,
            method="bm25",
            top_k=top_k,
            k1=self.k1,
            b=self.b,
        )


def get_scorer(scorer_type: ScorerType) -> Scorer:
    if scorer_type in ("tf-idf", "tfidf"):
        return TfIdfScorer()

    if scorer_type == "bm25":
        return BM25Scorer()

    raise ValueError(
        f"Unknown scorer algorithm: {scorer_type}"
    )
