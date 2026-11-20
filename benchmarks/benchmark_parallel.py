"""Benchmark serial, threaded and process-based index building."""

from __future__ import annotations

import csv
import os
import statistics
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
import psutil

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from findex.index import InvertedIndex, build_index_parallel

REPEATS = 3
WORKER_COUNTS = sorted({1, 2, 4, 8, os.cpu_count() or 1})
EXECUTORS = ("serial", "threads", "processes")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS_DIR / "parallel_benchmark.csv"
CHART_PATH = RESULTS_DIR / "parallel_speedup.png"


def measure(
    operation: Callable[[], InvertedIndex],
) -> dict[str, float]:
    """Measure wall time, CPU time, peak parent RSS and merge time."""
    process = psutil.Process(os.getpid())

    peak_rss = process.memory_info().rss
    stop_event = threading.Event()

    def monitor_memory() -> None:
        nonlocal peak_rss

        while not stop_event.wait(0.01):
            try:
                current_rss = process.memory_info().rss
                peak_rss = max(peak_rss, current_rss)
            except psutil.Error:
                break

    monitor = threading.Thread(
        target=monitor_memory,
        daemon=True,
    )

    start_wall = time.perf_counter()
    start_cpu = time.process_time()

    monitor.start()

    try:
        index = operation()
    finally:
        end_wall = time.perf_counter()
        end_cpu = time.process_time()

        stop_event.set()
        monitor.join()

        peak_rss = max(
            peak_rss,
            process.memory_info().rss,
        )

    return {
        "wall_seconds": end_wall - start_wall,
        "cpu_seconds": end_cpu - start_cpu,
        "peak_parent_rss_mb": peak_rss / (1024 * 1024),
        "merge_seconds": index.merge_seconds,
    }


def run_benchmarks(corpus_dir: Path) -> list[dict[str, Any]]:
    """Run every configuration at least three times."""
    results: list[dict[str, Any]] = []

    for executor_type in EXECUTORS:
        counts = [1] if executor_type == "serial" else WORKER_COUNTS

        for workers in counts:
            measurements: list[dict[str, float]] = []

            print(
                f"\nExecutor: {executor_type}, workers: {workers}",
                flush=True,
            )

            for repeat in range(1, REPEATS + 1):

                def build_operation(
                    current_workers: int = workers,
                    current_executor: str = executor_type,
                ) -> InvertedIndex:
                    return build_index_parallel(
                        str(corpus_dir),
                        workers=current_workers,
                        executor_type=current_executor,
                    )

                metrics = measure(build_operation)
                measurements.append(metrics)

                print(
                    f"  Repeat {repeat}/{REPEATS}: "
                    f"wall={metrics['wall_seconds']:.3f}s, "
                    f"cpu={metrics['cpu_seconds']:.3f}s, "
                    f"rss={metrics['peak_parent_rss_mb']:.1f}MB, "
                    f"merge={metrics['merge_seconds']:.3f}s",
                    flush=True,
                )

            result: dict[str, Any] = {
                "executor": executor_type,
                "workers": workers,
            }

            for key in (
                "wall_seconds",
                "cpu_seconds",
                "peak_parent_rss_mb",
                "merge_seconds",
            ):
                result[key] = statistics.median(
                    measurement[key]
                    for measurement in measurements
                )

            results.append(result)

    return results


def calculate_speedups(
    results: list[dict[str, Any]],
) -> None:
    """Calculate speedup relative to serial execution."""
    baseline = next(
        row["wall_seconds"]
        for row in results
        if row["executor"] == "serial"
    )

    for row in results:
        row["speedup"] = baseline / row["wall_seconds"]


def save_csv(results: list[dict[str, Any]]) -> None:
    """Save median benchmark measurements."""
    fieldnames = [
        "executor",
        "workers",
        "wall_seconds",
        "cpu_seconds",
        "peak_parent_rss_mb",
        "merge_seconds",
        "speedup",
    ]

    with CSV_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(results)


def save_chart(results: list[dict[str, Any]]) -> None:
    """Plot measured speedup against ideal linear speedup."""
    fig, ax = plt.subplots(figsize=(9, 6))

    for executor_type in ("threads", "processes"):
        rows = sorted(
            (
                row
                for row in results
                if row["executor"] == executor_type
            ),
            key=lambda row: row["workers"],
        )

        ax.plot(
            [row["workers"] for row in rows],
            [row["speedup"] for row in rows],
            marker="o",
            label=executor_type.capitalize(),
        )

    ax.plot(
        WORKER_COUNTS,
        WORKER_COUNTS,
        linestyle="--",
        label="Ideal speedup",
    )

    ax.axhline(
        1.0,
        linestyle=":",
        label="Serial baseline",
    )

    ax.set_title("Parallel Indexing: Actual vs Ideal Speedup")
    ax.set_xlabel("Number of workers")
    ax.set_ylabel("Speedup")
    ax.set_xticks(WORKER_COUNTS)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=160)
    plt.close(fig)


def main() -> None:
    """Execute benchmarks and generate output files."""
    corpus_dir = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path("data")
    )

    if not corpus_dir.is_dir():
        raise SystemExit(
            f"Corpus directory does not exist: {corpus_dir}"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Corpus: {corpus_dir.resolve()}")
    print(f"Logical CPUs: {os.cpu_count()}")
    print(f"Worker counts: {WORKER_COUNTS}")
    print(f"Repeats: {REPEATS}")

    results = run_benchmarks(corpus_dir)
    calculate_speedups(results)

    save_csv(results)
    save_chart(results)

    print("\nBenchmark summary:")
    print(
        f"{'Executor':<12} {'Workers':>7} "
        f"{'Wall (s)':>10} {'CPU (s)':>10} "
        f"{'RSS (MB)':>10} {'Merge (s)':>10} "
        f"{'Speedup':>9}"
    )

    for row in results:
        print(
            f"{row['executor']:<12} "
            f"{row['workers']:>7} "
            f"{row['wall_seconds']:>10.3f} "
            f"{row['cpu_seconds']:>10.3f} "
            f"{row['peak_parent_rss_mb']:>10.1f} "
            f"{row['merge_seconds']:>10.3f} "
            f"{row['speedup']:>9.3f}"
        )

    print(f"\nCSV saved to: {CSV_PATH}")
    print(f"Chart saved to: {CHART_PATH}")


if __name__ == "__main__":
    main()