
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
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from findex.crawler import crawl_site
from findex.index import InvertedIndex, open_index
from findex.scoring import BM25Scorer, Scorer, TfIdfScorer

app = typer.Typer(help="Findex - Python search engine CLI")
console = Console()
error_console = Console(stderr=True)

ScorerName = Literal["bm25", "tfidf"]


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


def create_index(corpus_dir: Path, out: Path) -> None:
    """Build and save an inverted index with progress indication."""
    if not corpus_dir.is_dir():
        fail("Corpus directory not found.")

    if out.suffix.lower() not in (".pkl", ".json"):
        fail("Output must be .pkl or .json.")

    index = InvertedIndex()
    start = time.perf_counter()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task("Building index...", total=None)
        index.build(str(corpus_dir))

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


@app.command("index")
def index_command(
    corpus_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing corpus documents"),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="Output index file (.pkl or .json)",
        ),
    ] = Path("index.pkl"),
) -> None:
    """Build and save an inverted index."""
    create_index(corpus_dir, out)


@app.command("build")
def build(
    corpus_dir: Annotated[
        Path,
        typer.Argument(help="Directory containing corpus documents"),
    ],
    out: Annotated[
        Path,
        typer.Option(
            "--out",
            "-o",
            help="Output index file (.pkl or .json)",
        ),
    ] = Path("index.pkl"),
) -> None:
    """Build an index (legacy command)."""
    create_index(corpus_dir, out)


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

    ranking: Scorer = (
        BM25Scorer() if scorer_name == "bm25" else TfIdfScorer()
    )

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

        sys.stdout.write(
            json.dumps(payload, ensure_ascii=False) + "\n"
        )
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

    total_tokens = sum(
        meta.length for meta in index.metadata.values()
    )

    total_postings = sum(
        len(data["doc_ids"])
        for data in index.index.values()
    )

    average_length = (
        total_tokens / documents if documents else 0.0
    )

    table = Table(title="Findex Statistics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    table.add_row(f"Documents: {documents}", str(documents))
    table.add_row(f"Unique terms: {terms}", str(terms))
    table.add_row(f"Total tokens: {total_tokens}", str(total_tokens))
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
            help="Concurrency level",
        ),
    ] = 5,
) -> None:
    """Run async web crawler and save pages to JSONL."""
    console.print(
        f"[bold cyan]Starting crawler for {start_url} "
        f"(max_pages={max_pages}, concurrency={concurrency})"
        f"[/bold cyan]"
    )

    async def run() -> None:
        start = time.perf_counter()

        try:
            await asyncio.to_thread(
                out_jsonl.write_text,
                "",
                encoding="utf-8",
            )
        except OSError as exc:
            fail(f"Cannot create output file: {exc}")

        async for page in crawl_site(
            start_url,
            max_pages=max_pages,
            concurrency=concurrency,
        ):
            line = json.dumps(
                {
                    "doc_id": page["doc_id"],
                    "filename": page["url"],
                    "text": page["html"],
                },
                ensure_ascii=False,
            ) + "\n"

            try:
                await asyncio.to_thread(
                    append_line,
                    out_jsonl,
                    line,
                )
            except OSError as exc:
                fail(f"Cannot write output file: {exc}")

            console.print(
                f"[{page['doc_id']}/{max_pages}] "
                f"Collected: {page['url']}"
            )

        elapsed = time.perf_counter() - start

        console.print(
            f"[bold green]Crawled successfully "
            f"in {elapsed:.2f}s. "
            f"Saved to {out_jsonl}[/bold green]"
        )

    asyncio.run(run())


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
        host="127.0.0.1",
        port=port,
        workers=workers,
        reload=False,
    )


if __name__ == "__main__":
    app()
