
import csv
import json
from pathlib import Path

from findex.index import open_index
from findex.search import execute_search
from findex.scoring import BM25Scorer, TfIdfScorer


INDEX_PATH = "index.pkl"
TOP_K = 5

RESULTS_PATH = Path("evaluation_results.csv")
JUDGMENTS_PATH = Path("relevance_judgments.json")

# Десять запитів для оцінювання
QUERIES = [
    "alice",
    "wonderland",
    "rabbit",
    "queen",
    "king",
    "cat",
    "garden",
    "tea",
    "alice AND rabbit",
    '"alice in wonderland"',
]


def precision_at_k(retrieved, relevant, k=5):
    """
    Обчислює Precision@k.

    retrieved — список знайдених документів.
    relevant — список вручну визначених
               релевантних документів.
    """

    if k < 1:
        raise ValueError("k must be greater than zero")

    relevant_set = set(relevant)

    relevant_count = sum(
        doc_id in relevant_set
        for doc_id in retrieved[:k]
    )

    return relevant_count / k


def load_judgments():
    """Завантажує збережені оцінки релевантності."""

    if not JUDGMENTS_PATH.exists():
        return {}

    with JUDGMENTS_PATH.open(
        "r", encoding="utf-8"
    ) as file:
        return json.load(file)


def save_judgments(judgments):
    """Зберігає ручні оцінки релевантності."""

    with JUDGMENTS_PATH.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(
            judgments,
            file,
            ensure_ascii=False,
            indent=4
        )


def collect_judgments(index, queries):
    """
    Запитує користувача, які документи
    є релевантними для кожного запиту.

    Оцінки зберігаються у JSON,
    тому їх не потрібно вводити повторно.
    """

    judgments = load_judgments()

    valid_ids = set(index.metadata.keys())

    print("\nСПИСОК ДОКУМЕНТІВ:")

    for doc_id, meta in sorted(index.metadata.items()):
        print(f"{doc_id}: {meta.filename}")

    print("\nРУЧНЕ ОЦІНЮВАННЯ РЕЛЕВАНТНОСТІ")
    print(
        "Вводь номери релевантних документів "
        "через пробіл."
    )
    print(
        "Наприклад: 1 5"
    )
    print(
        "Якщо релевантних документів немає, "
        "введи: none"
    )
    print(
        "Якщо ще не можеш оцінити запит, "
        "натисни Enter — він залишиться без оцінки."
    )

    for query in queries:

        if query in judgments:
            print(
                f"\nЗапит '{query}' уже оцінено: "
                f"{judgments[query]}"
            )
            continue

        while True:
            print(f"\nЗапит: {query}")

            answer = input(
                "Релевантні документи: "
            ).strip()

            if not answer:
                print(
                    "Оцінку пропущено. "
                    "Її можна додати пізніше."
                )
                break

            if answer.lower() == "none":
                judgments[query] = []
                save_judgments(judgments)
                break

            try:
                relevant_ids = sorted(
                    set(
                        int(value)
                        for value in answer.split()
                    )
                )
            except ValueError:
                print(
                    "Помилка: введи номери "
                    "документів через пробіл."
                )
                continue

            if not set(relevant_ids).issubset(valid_ids):
                print(
                    "Помилка: можна вводити лише "
                    "номери наявних документів."
                )
                continue

            judgments[query] = relevant_ids

            save_judgments(judgments)

            print(
                f"Збережено: {relevant_ids}"
            )

            break

    return judgments


def main():

    index = open_index(INDEX_PATH)

    print("=" * 55)
    print("ОЦІНЮВАННЯ ЯКОСТІ ПОШУКУ FINDEX")
    print("=" * 55)

    print(f"Документів: {index.num_docs}")
    print(f"Унікальних термінів: {len(index)}")
    print(f"Кількість запитів: {len(QUERIES)}")
    print(f"Метрика: Precision@{TOP_K}")

    # Отримуємо ручні оцінки релевантності.
    judgments = collect_judgments(
        index,
        QUERIES
    )

    scorers = {
        "BM25": BM25Scorer(),
        "TF-IDF": TfIdfScorer(),
    }

    scores_by_scorer = {
        name: []
        for name in scorers
    }

    evaluated_queries = 0

    with RESULTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "query",
            "scorer",
            "retrieved_doc_ids",
            "relevant_doc_ids",
            "precision_at_5",
        ])

        for query in QUERIES:

            print("\n" + "-" * 55)
            print(f"Запит: {query}")

            is_evaluated = query in judgments

            relevant = judgments.get(query, [])

            if is_evaluated:
                evaluated_queries += 1

            for scorer_name, scorer in scorers.items():

                results = execute_search(
                    index,
                    query,
                    scorer,
                    top_k=TOP_K
                )

                retrieved = [
                    result.doc_id
                    for result in results
                ]

                if is_evaluated:
                    precision = precision_at_k(
                        retrieved,
                        relevant,
                        TOP_K
                    )

                    scores_by_scorer[
                        scorer_name
                    ].append(precision)

                    precision_display = (
                        f"{precision:.4f}"
                    )

                else:
                    precision = None
                    precision_display = ""

                print(
                    f"{scorer_name}: "
                    f"документи {retrieved}"
                )

                if precision is not None:
                    print(
                        f"Precision@{TOP_K}: "
                        f"{precision:.4f}"
                    )
                else:
                    print(
                        "Precision: не оцінено"
                    )

                writer.writerow([
                    query,
                    scorer_name,
                    " ".join(
                        map(str, retrieved)
                    ),
                    (
                        " ".join(
                            map(str, relevant)
                        )
                        if is_evaluated
                        else ""
                    ),
                    precision_display,
                ])

    print("\n" + "=" * 55)
    print("ПІДСУМКИ ОЦІНЮВАННЯ")
    print("=" * 55)

    print(
        f"Оцінено запитів: "
        f"{evaluated_queries}/{len(QUERIES)}"
    )

    for scorer_name, scores in scores_by_scorer.items():

        if scores:
            mean_precision = (
                sum(scores) / len(scores)
            )

            print(
                f"{scorer_name}: "
                f"Mean Precision@{TOP_K} = "
                f"{mean_precision:.4f}"
            )

        else:
            print(
                f"{scorer_name}: "
                "ще немає оцінених запитів"
            )

    print(
        f"\nРезультати збережено: "
        f"{RESULTS_PATH}"
    )

    print(
        f"Оцінки релевантності: "
        f"{JUDGMENTS_PATH}"
    )


if __name__ == "__main__":
    main()
