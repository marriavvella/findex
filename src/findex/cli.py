import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Annotated, Literal, NoReturn

import typer
import uvicorn
from rich.console import Console
from rich.live import Live
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from findex.crawler import CrawlStats, crawl_site
from findex.index import InvertedIndex, build_index_parallel, open_index
from findex.scoring import BM25Scorer, Scorer, TfIdfScorer

app = typer.Typer(help="Findex - Python search engine CLI")
console = Console()
error_console = Console(stderr=True)
logger = logging.getLogger(__name__)

ScorerName = Literal["bm25", "tfidf"]
ExecutorName = Literal["serial", "threads", "processes"]


@app.callback()
def main(
    verbose: Annotated[
        int,
        typer.Option(
            "-v",
            "--verbose",
            count=True,
            help="Increase logging verbosity (-v or -vv)",
        ),
    ] = 0,
) -> None:
    """Configure CLI logging."""
    level = logging.WARNING

    if verbose == 1:
        level = logging.INFO
    elif verbose >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


def fail(message: str) -> NoReturn:
    """Print one-line error to stderr and exit."""
    error_console.print(f"[red]Error: {message}[/red]")
    raise typer.Exit(code=1)


def create_index(
    corpus_dir: Path,
    out: Path,
    workers: int = 1,
    executor_type: ExecutorName = "serial",
) -> None:
    """Build and save an inverted index with progress indication."""
    if not corpus_dir.exists():
        fail("Corpus directory not found.")

    if corpus_dir.is_file() and corpus_dir.suffix.lower() != ".jsonl":
        fail("Corpus file must have a .jsonl extension.")

    if not corpus_dir.is_dir() and not corpus_dir.is_file():
        fail("Corpus path must be a directory or a JSONL file.")

    if out.suffix.lower() not in (".pkl", ".json"):
        fail("Output must be .pkl or .json.")

    if workers < 1:
        fail("Workers must be at least 1.")

    start = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Building index...", total=None)

        if corpus_dir.is_file():
            # JSONL is read lazily by iter_documents().
            # Keep indexing outside the asyncio event loop.
            index = InvertedIndex()
            index.build(str(corpus_dir))
        elif executor_type == "serial" and workers == 1:
            index = InvertedIndex()
            index.build(str(corpus_dir))
        else:
            index = build_index_parallel(
                str(corpus_dir),
                workers=workers,
                executor_type=executor_type,
            )

    try:
        if out.suffix.lower() == ".json":
            index.save_json(str(out))
        else:
            index.save_pickle(str(out))
    except OSError as exc:
        fail(f"Cannot save index: {exc}")

    elapsed = time.perf_counter() - start

    console.print("[bold green]Index built successfully![/bold green]")
    console.print(f"Documents: {len(index.metadata)}")
    console.print(f"Unique terms: {len(index.index)}")
    console.print(f"Saved to: {out}")
    console.print(f"Elapsed: {elapsed:.3f} seconds")
    logger.info("Executor: %s; workers: %d", executor_type, workers)


@app.command("index")
def index_command(
    corpus_dir: Annotated[
        Path,
        typer.Argument(help="Corpus directory or JSONL file"),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="Output index file (.pkl or .json)",
        ),
    ] = Path("index.pkl"),
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-w",
            min=1,
            help="Number of workers",
        ),
    ] = 1,
    executor_type: Annotated[
        ExecutorName,
        typer.Option(
            "--executor",
            help="serial, threads or processes",
        ),
    ] = "serial",
) -> None:
    """Build and save an inverted index."""
    create_index(corpus_dir, out, workers, executor_type)


@app.command("build")
def build(
    corpus_dir: Annotated[
        Path,
        typer.Argument(help="Corpus directory or JSONL file"),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="Output index file (.pkl or .json)",
        ),
    ] = Path("index.pkl"),
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-w",
            min=1,
            help="Number of workers",
        ),
    ] = 1,
    executor_type: Annotated[
        ExecutorName,
        typer.Option(
            "--executor",
            help="serial, threads or processes",
        ),
    ] = "serial",
) -> None:
    """Build an index (legacy command)."""
    create_index(corpus_dir, out, workers, executor_type)


@app.command()
def search(
    query: Annotated[
        str,
        typer.Argument(help="Search query"),
    ],
    index_path: Annotated[
        Path,
        typer.Option(
            "--index",
            "-i",
            help="Saved index file",
        ),
    ] = Path("index.pkl"),
    scorer: Annotated[
        str,
        typer.Option(
            "--scorer",
            help="Ranking algorithm: bm25 or tfidf",
        ),
    ] = "bm25",
    top_k: Annotated[
        int,
        typer.Option(
            "--top-k",
            "-k",
            min=1,
            help="Maximum number of results",
        ),
    ] = 5,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output search results as JSON",
        ),
    ] = False,
) -> None:
    """Search indexed documents."""
    if not index_path.is_file():
        fail("Index file not found.")

    from findex.search import execute_search

    try:
        index = open_index(str(index_path))
    except (OSError, ValueError, EOFError) as exc:
        fail(f"Cannot load index: {exc}")

    scorer_name = scorer.lower()

    if scorer_name not in ("bm25", "tfidf"):
        fail("Scorer must be bm25 or tfidf.")

    ranking: Scorer = BM25Scorer() if scorer_name == "bm25" else TfIdfScorer()

    results = execute_search(
        index,
        query,
        ranking,
        top_k=top_k,
    )

    if json_output:
        payload = {
            "query": query,
            "scorer": scorer,
            "results": [
                {
                    "doc_id": result.doc_id,
                    "score": result.score,
                    "filename": result.filename,
                    "snippet": result.snippet,
                }
                for result in results
            ],
        }

        sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
        return

    table = Table(title=f"Search results: {query}")
    table.add_column("Document", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Filename", style="green")
    table.add_column("Snippet")

    for result in results:
        table.add_row(
            f"Doc {result.doc_id}",
            f"{result.score:.4f}",
            result.filename,
            result.snippet,
        )

    console.print(table)
    console.print(f"Found {len(results)} results")


@app.command()
def stats(
    index_path: Annotated[
        Path,
        typer.Option(
            "--index",
            "-i",
            help="Saved index file",
        ),
    ] = Path("index.pkl"),
) -> None:
    """Show inverted index statistics."""
    if not index_path.is_file():
        fail("Index file not found.")

    try:
        index = open_index(str(index_path))
    except (OSError, ValueError, EOFError) as exc:
        fail(f"Cannot load index: {exc}")

    documents = len(index.metadata)
    terms = len(index.index)

    total_tokens = sum(meta.length for meta in index.metadata.values())

    total_postings = sum(len(data["doc_ids"]) for data in index.index.values())

    average_length = total_tokens / documents if documents else 0.0

    table = Table(title="Findex Statistics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row(f"Documents: {documents}", str(documents))
    table.add_row("Unique terms:", str(terms))
    table.add_row("Total tokens:", str(total_tokens))
    table.add_row("Total postings", str(total_postings))
    table.add_row(
        "Average document length",
        f"{average_length:.2f}",
    )
    table.add_row(
        "Index file size",
        f"{index_path.stat().st_size} bytes",
    )

    console.print(table)


def make_crawl_table(
    stats: CrawlStats,
    max_pages: int,
    concurrency: int,
) -> Table:
    """Build a Rich table with current crawler statistics."""
    table = Table(title="Findex Async Crawler")

    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")

    table.add_row(
        "Pages collected",
        f"{stats.pages}/{max_pages}",
    )
    table.add_row(
        "Elapsed time",
        f"{stats.elapsed:.2f} s",
    )
    table.add_row(
        "Pages per second",
        f"{stats.pages_per_second:.2f}",
    )
    table.add_row(
        "Active requests",
        str(stats.active_requests),
    )
    table.add_row(
        "Queue depth",
        str(stats.queue_depth),
    )
    table.add_row(
        "Errors",
        str(stats.errors),
    )
    table.add_row(
        "Concurrency limit",
        str(concurrency),
    )

    return table


@app.command()
def crawl(
    start_url: Annotated[
        str,
        typer.Argument(help="Starting URL to crawl"),
    ],
    out_jsonl: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="Output JSONL file path",
        ),
    ] = Path("corpus.jsonl"),
    max_pages: Annotated[
        int,
        typer.Option(
            "--max-pages",
            "-m",
            min=1,
            help="Maximum pages to collect",
        ),
    ] = 200,
    concurrency: Annotated[
        int,
        typer.Option(
            "--concurrency",
            "-c",
            min=1,
            help="Global concurrency limit",
        ),
    ] = 5,
    per_host_concurrency: Annotated[
        int,
        typer.Option(
            "--per-host-concurrency",
            min=1,
            help="Maximum concurrent requests per host",
        ),
    ] = 2,
    host_delay: Annotated[
        float,
        typer.Option(
            "--host-delay",
            min=0.0,
            help="Minimum delay between requests to one host",
        ),
    ] = 0.2,
    log_file: Annotated[
        Path,
        typer.Option(
            "--log-file",
            help="Path for crawler HTTP log",
        ),
    ] = Path("crawl.log"),
) -> None:
    """Run asynchronous crawler and save pages to JSONL."""
    console.print(
        f"[bold cyan]Starting crawler for {start_url} "
        f"(max_pages={max_pages}, concurrency={concurrency})"
        f"[/bold cyan]"
    )

    stats = CrawlStats()

    crawler_logger = logging.getLogger("findex.crawler")
    previous_level = crawler_logger.level

    file_handler: logging.FileHandler | None = None

    try:
        log_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_handler = logging.FileHandler(
            log_file,
            mode="w",
            encoding="utf-8",
        )

        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        )

        crawler_logger.addHandler(file_handler)
        crawler_logger.setLevel(logging.INFO)

    except OSError as exc:
        fail(f"Cannot create crawler log: {exc}")

    async def run() -> None:
        start = time.perf_counter()

        try:
            out_jsonl.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            await asyncio.to_thread(
                out_jsonl.write_text,
                "",
                encoding="utf-8",
            )

        except OSError as exc:
            fail(f"Cannot create output file: {exc}")

        try:
            with Live(
                make_crawl_table(
                    stats,
                    max_pages,
                    concurrency,
                ),
                console=console,
                refresh_per_second=4,
                transient=False,
            ) as live:
                async for page in crawl_site(
                    start_url,
                    max_pages=max_pages,
                    concurrency=concurrency,
                    per_host_concurrency=per_host_concurrency,
                    host_delay=host_delay,
                    stats=stats,
                ):
                    line = (
                        json.dumps(
                            {
                                "doc_id": page["doc_id"],
                                "filename": page["url"],
                                "text": page["html"],
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    await asyncio.to_thread(
                        append_line,
                        out_jsonl,
                        line,
                    )

                    live.update(
                        make_crawl_table(
                            stats,
                            max_pages,
                            concurrency,
                        )
                    )

                live.update(
                    make_crawl_table(
                        stats,
                        max_pages,
                        concurrency,
                    )
                )

        except OSError as exc:
            fail(f"Cannot write crawler output: {exc}")

        elapsed = time.perf_counter() - start

        console.print(
            f"[bold green]Crawler finished in {elapsed:.2f} seconds.[/bold green]"
        )
        console.print(f"Pages collected: {stats.pages}")
        console.print(f"Errors: {stats.errors}")
        console.print(f"Output: {out_jsonl}")
        console.print(f"Log: {log_file}")

    try:
        asyncio.run(run())
    finally:
        if file_handler is not None:  # pyright: ignore[reportUnnecessaryComparison]
            crawler_logger.removeHandler(file_handler)
            file_handler.close()

        crawler_logger.setLevel(previous_level)


def append_line(path: Path, line: str) -> None:
    """Append one JSONL record to a file."""
    with path.open("a", encoding="utf-8") as file:
        file.write(line)


@app.command()
def serve(
    port: Annotated[
        int,
        typer.Option(
            "--port",
            "-p",
            min=1,
            max=65535,
            help="Port to run server",
        ),
    ] = 8000,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            "-w",
            min=1,
            help="Number of uvicorn workers",
        ),
    ] = 1,
) -> None:
    """Start FastAPI web server via uvicorn."""
    uvicorn.run(
        "findex.web.app:app",
        host=__import__("os").environ.get("HOST", "0.0.0.0"),
        port=port,
        workers=workers,
        reload=False,
    )


if __name__ == "__main__":
    app()
