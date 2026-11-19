
import argparse
import tracemalloc
from typing import List

from findex.index import open_index, Index
from findex.query_parser import parse_query, Term, Phrase, And, Or, Not
from findex.scoring import BM25Scorer, TfIdfScorer, Scorer
from findex.utils import timed, generate_snippet, cached_search_ids
from findex.models import SearchResult


def extract_query_terms(node) -> List[str]:
    """Витягує терміни з AST-дерева для ранжування та сніпетів."""

    if isinstance(node, Term):
        return [node.term]

    elif isinstance(node, Phrase):
        return node.tokens

    elif isinstance(node, (And, Or)):
        return (
            extract_query_terms(node.left)
            + extract_query_terms(node.right)
        )

    elif isinstance(node, Not):
        return extract_query_terms(node.child)

    return []


@timed
def execute_search(
    index: Index,
    query_str: str,
    scorer: Scorer,
    top_k: int = 5
) -> List[SearchResult]:
    """Виконує Boolean-пошук із ранжуванням результатів."""

    if top_k < 1:
        raise ValueError("top_k must be greater than zero")

    # 1. Розбираємо запит у дерево AST
    ast_root = parse_query(query_str)

    # 2. Знаходимо документи, які відповідають запиту
    matching_docs = list(ast_root.evaluate(index))

    if not matching_docs:
        return []

    # 3. Отримуємо терміни для ранжування
    terms = extract_query_terms(ast_root)

    # 4. Ранжуємо результати за BM25 або TF-IDF
    results = scorer.score(index, terms, matching_docs)

    # 5. Викликаємо функцію кешування
    cached_search_ids(query_str, id(index))

    # 6. Формуємо сніпети для знайдених документів
    final_results: List[SearchResult] = []

    for res in results[:top_k]:
        try:
            with open(
                res.filename,
                "r",
                encoding="utf-8"
            ) as f:
                content = f.read()

            snippet = generate_snippet(content, terms)

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
        description="Ranked Boolean Search with Snippets"
    )

    parser.add_argument(
        "index_path",
        type=str,
        help="Path to saved index file (.pkl or .json)"
    )

    parser.add_argument(
        "query",
        type=str,
        help="Search query, e.g. 'python AND (async OR await)'"
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
        parser.error("--top-k must be greater than zero")

    # Починаємо вимірювання використання пам'яті
    tracemalloc.start()

    try:
        # Завантажуємо індекс без використання with
        index = open_index(args.index_path)

        # Вибираємо алгоритм ранжування
        scorer = (
            BM25Scorer()
            if args.scorer == "bm25"
            else TfIdfScorer()
        )

        # Виконуємо пошук
        results = execute_search(
            index,
            args.query,
            scorer,
            top_k=args.top_k
        )

        # Отримуємо пікове використання пам'яті
        _, peak_mem = tracemalloc.get_traced_memory()

    finally:
        tracemalloc.stop()

    print(
        f"\nFound {len(results)} top results | "
        f"Peak memory: {peak_mem / 1024 / 1024:.2f} MB\n"
    )

    for result in results:
        print(
            f"Score: {result.score:.4f} | "
            f"[Doc {result.doc_id}] {result.filename}"
        )

        print(f"Snippet: {result.snippet}\n")


if __name__ == "__main__":
    main()
