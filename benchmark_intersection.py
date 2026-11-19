
import argparse
import statistics
import time
from pathlib import Path

from findex.index import open_index


def intersect_set(a, b):
    """Перетин списків через множини Python."""
    return sorted(set(a) & set(b))


def intersect_merge(a, b):
    """Перетин двох відсортованих списків двома вказівниками."""
    i = 0
    j = 0
    result = []

    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            result.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1

    return result


def benchmark(func, a, b, repeats=1000):
    """Вимірювання середнього часу виконання алгоритму."""
    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        func(a, b)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return statistics.mean(times) * 1000


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark: set intersection vs two-pointer merge"
    )

    parser.add_argument(
        "--index",
        default="index.pkl",
        help="Path to saved index"
    )

    parser.add_argument(
        "--term1",
        default="alice",
        help="First search term"
    )

    parser.add_argument(
        "--term2",
        default="wonderland",
        help="Second search term"
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=1000,
        help="Number of benchmark repetitions"
    )

    args = parser.parse_args()

    if args.repeats < 1:
        parser.error("--repeats must be greater than zero")

    if not Path(args.index).is_file():
        parser.error(f"Index file not found: {args.index}")

    index = open_index(args.index)

    postings1 = index.index.get(args.term1)
    postings2 = index.index.get(args.term2)

    if postings1 is None or postings2 is None:
        print("One or both terms were not found in the index.")
        return

    a = [int(x) for x in postings1["doc_ids"]]
    b = [int(x) for x in postings2["doc_ids"]]

    # Для merge потрібні відсортовані списки.
    a = sorted(set(a))
    b = sorted(set(b))

    result_set = intersect_set(a, b)
    result_merge = intersect_merge(a, b)

    assert result_set == result_merge, (
        "ERROR: Intersection algorithms returned different results!"
    )

    set_time = benchmark(
        intersect_set, a, b, args.repeats
    )

    merge_time = benchmark(
        intersect_merge, a, b, args.repeats
    )

    print("\nINTERSECTION BENCHMARK")
    print("-" * 45)
    print(f"Term 1: {args.term1}")
    print(f"Term 2: {args.term2}")
    print(f"Posting list 1: {len(a)} documents")
    print(f"Posting list 2: {len(b)} documents")
    print(f"Intersection: {len(result_set)} documents")
    print(f"Repetitions: {args.repeats}")

    print("\nAverage execution time:")
    print(f"Set intersection:   {set_time:.6f} ms")
    print(f"Merge intersection: {merge_time:.6f} ms")

    if merge_time > 0:
        ratio = set_time / merge_time
        print(f"Set / Merge ratio:  {ratio:.2f}x")

    print("\nVerification: PASSED")
    print("Both algorithms returned identical results.")


if __name__ == "__main__":
    main()
