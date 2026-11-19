
import argparse
import tracemalloc

from findex.index import Index, open_index
from findex.models import SearchResult
from findex.query_parser import (
    And,
    ASTNode,
    Not,
    Or,
    Phrase,
    Term,
    parse_query,
)
from findex.scoring import (
    BM25Scorer,
    Scorer,
    TfIdfScorer,
)
from findex.utils import (
    cached_search_ids,
    generate_snippet,
    timed,
)


def extract_query_terms(node: ASTNode) -> list[str]:
    """Витягує терміни з AST для ранжування."""

    if isinstance(node, Term):
        return [node.term]

    if isinstance(node, Phrase):
        return node.tokens

    if isinstance(node, (And, Or)):
        return (
            extract_query_terms(node.left)
            + extract_query_terms(node.right)
        )

    if isinstance(node, Not):
        return []

    return []


@timed
def execute_search(
    index: Index,
    query_str: str,
    scorer: Scorer,
    top_k: int = 5,
) -> list[SearchResult]:
    """Виконує Boolean-пошук із ранжуванням."""

    if top_k < 1:
        raise ValueError(
            "top_k must be greater than zero"
        )

    # Отримуємо результати з кешу або обчислюємо їх
    matching_docs = cached_search_ids(
        index, query_str
    )

    if not matching_docs:
        return []

    # Отримуємо терміни для ранжування
    ast_root = parse_query(query_str)
    terms = extract_query_terms(ast_root)

    # Ранжування BM25 або TF-IDF
    results = scorer.score(
        index,
        terms,
        matching_docs
    )

    # Формування результатів зі сніпетами
    final_results: list[SearchResult] = []

    for res in results[:top_k]:
        try:
            with open(
                res.filename,
                "r",
                encoding="utf-8"
            ) as file:
                content = file.read()

            snippet = generate_snippet(
                content, terms
            )

        except (OSError, UnicodeError):
            snippet = "Текст недоступний"

        final_results.append(
            SearchResult(
                score=res.score,
                doc_id=res.doc_id,
                filename=res.filename,
                snippet=snippet
            )
        )

    return final_results


def main() -> None:
    """Запускає пошук із командного рядка."""

    parser = argparse.ArgumentParser(
        description=(
            "Ranked Boolean Search with Snippets"
        )
    )

    parser.add_argument(
        "index_path",
        type=str,
        help="Path to saved index (.pkl or .json)"
    )

    parser.add_argument(
        "query",
        type=str,
        help="Search query"
    )

    parser.add_argument(
        "--scorer",
        choices=["bm25", "tfidf"],
        default="bm25",
        help="Ranking algorithm"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return"
    )

    args = parser.parse_args()

    if args.top_k < 1:
        parser.error(
            "--top-k must be greater than zero"
        )

    tracemalloc.start()

    try:
        index = open_index(args.index_path)

        scorer: Scorer = (
            BM25Scorer()
            if args.scorer == "bm25"
            else TfIdfScorer()
        )

        results = execute_search(
            index,
            args.query,
            scorer,
            top_k=args.top_k
        )

        _, peak_mem = (
            tracemalloc.get_traced_memory()
        )

    finally:
        tracemalloc.stop()

    print(
        f"\nFound {len(results)} top results | "
        f"Peak memory: "
        f"{peak_mem / 1024 / 1024:.2f} MB\n"
    )

    for result in results:
        print(
            f"Score: {result.score:.4f} | "
            f"[Doc {result.doc_id}] "
            f"{result.filename}"
        )
        print(f"Snippet: {result.snippet}\n")


if __name__ == "__main__":
    main()
