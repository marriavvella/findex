
import os
import time
from pathlib import Path

from findex.index import open_index


def main():
    pickle_path = Path("index.pkl")
    json_path = Path("index.json")

    if not pickle_path.exists():
        print("Error: index.pkl not found")
        print("Run: uv run python build_index.py")
        return

    # Завантаження початкового індексу
    index = open_index(str(pickle_path))

    # Вимірювання часу збереження Pickle
    start = time.perf_counter()
    index.save_pickle(str(pickle_path))
    pickle_save_time = time.perf_counter() - start

    # Вимірювання часу збереження JSON
    start = time.perf_counter()
    index.save_json(str(json_path))
    json_save_time = time.perf_counter() - start

    # Вимірювання часу завантаження Pickle
    start = time.perf_counter()
    pickle_index = open_index(str(pickle_path))
    pickle_load_time = time.perf_counter() - start

    # Вимірювання часу завантаження JSON
    start = time.perf_counter()
    json_index = open_index(str(json_path))
    json_load_time = time.perf_counter() - start

    # Перевірка кількості документів і термінів
    assert len(pickle_index.metadata) == len(json_index.metadata)
    assert set(pickle_index.index) == set(json_index.index)

    # Порівняння розмірів файлів
    pickle_size = os.path.getsize(pickle_path)
    json_size = os.path.getsize(json_path)

    saving = (1 - json_size / pickle_size) * 100

    print("\nSerialization benchmark")
    print("-" * 55)
    print(f"{'Metric':<25} {'Pickle':>12} {'JSON':>12}")
    print("-" * 55)
    print(f"{'File size (bytes)':<25} {pickle_size:>12} {json_size:>12}")
    print(f"{'Save time (s)':<25} {pickle_save_time:>12.4f} {json_save_time:>12.4f}")
    print(f"{'Load time (s)':<25} {pickle_load_time:>12.4f} {json_load_time:>12.4f}")
    print("-" * 55)

    print(f"Documents: {len(json_index.metadata)}")
    print(f"Unique terms: {len(json_index.index)}")
    print(f"JSON size reduction: {saving:.2f}%")
    print("Serialization verification: PASSED")


if __name__ == "__main__":
    main()
