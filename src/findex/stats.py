import sys
import time
import tracemalloc
from collections import Counter
from itertools import islice
from pathlib import Path

from findex.corpus import iter_documents
from findex.tokenize import tokenize


def main():
    if len(sys.argv) < 2:
        print("Використання: python -m findex.stats <corpus_dir> [--limit N]")
        sys.exit(1)

    root = Path(sys.argv[1])
    limit = None

    # Перевірка наявності аргументу --limit для обмеження обробки
    if "--limit" in sys.argv:
        limit_idx = sys.argv.index("--limit")
        if limit_idx + 1 < len(sys.argv):
            limit = int(sys.argv[limit_idx + 1])

    # Запускаємо трекер пам'яті та таймер
    tracemalloc.start()
    start_time = time.perf_counter()

    doc_count = 0
    total_tokens = 0
    term_counter = Counter()

    # Створюємо ледачий потік документів
    docs_stream = iter_documents(root)
    if limit is not None:
        docs_stream = islice(docs_stream, limit)

    # Проходимо по корпусу в один прохід, не завантажуючи все в пам'ять
    for doc in docs_stream:
        doc_count += 1
        for token in tokenize(doc.text):
            total_tokens += 1
            term_counter[token] += 1

    # Зупиняємо вимірювання
    elapsed_time = time.perf_counter() - start_time
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("=== Статистика корпусу (findex) ===")
    print(f"Оброблено документів: {doc_count}")
    print(f"Загальна кількість токенів: {total_tokens}")
    print(f"Розмір словника (унікальних термів): {len(term_counter)}")
    print(f"Час виконання: {elapsed_time:.4f} сек.")
    print(f"Пікова пам'ять: {peak / 1024 / 1024:.2f} МБ")
    print("\nТоп-50 термів:")
    for term, freq in term_counter.most_common(50):
        print(f"  {term}: {freq}")


if __name__ == "__main__":
    main()
