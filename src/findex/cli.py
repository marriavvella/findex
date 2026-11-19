
import asyncio
import json
import time
from pathlib import Path

import typer
import uvicorn
from rich.console import Console

from findex.crawler import crawl_site
from findex.index import InvertedIndex, open_index
from findex.scoring import BM25Scorer, TfIdfScorer

app = typer.Typer(help="Findex - Python search engine CLI")
console = Console()


@app.command()
def build(
    corpus_dir: Path = typer.Argument(
        ..., help="Directory containing corpus documents"
    ),
    out: Path = typer.Option(
        Path("index.pkl"), "--out", "-o",
        help="Output index file (.pkl or .json)"
    ),
) -> None:
    """Build and save an inverted index."""

    if not corpus_dir.is_dir():
        console.print("[red]Corpus directory not found.[/red]")
        raise typer.Exit(code=1)

    if out.suffix.lower() not in (".pkl", ".json"):
        console.print("[red]Output must be .pkl or .json[/red]")
        raise typer.Exit(code=1)

    index = InvertedIndex()

    start = time.perf_counter()
    index.build(str(corpus_dir))

    if out.suffix.lower() == ".json":
        index.save_json(str(out))
    else:
        index.save_pickle(str(out))

    elapsed = time.perf_counter() - start

    console.print("[bold green]Index built successfully![/bold green]")
    console.print(f"Documents: {len(index.metadata)}")
    console.print(f"Unique terms: {len(index.index)}")
    console.print(f"Saved to: {out}")
    console.print(f"Elapsed: {elapsed:.3f} seconds")


@app.command()
def search(
    query: str = typer.Argument(
        ..., help="Search query"
    ),
    index_path: Path = typer.Option(
        Path("index.pkl"), "--index", "-i",
        help="Saved index file"
    ),
    scorer: str = typer.Option(
        "bm25", "--scorer",
        help="Ranking: bm25 or tfidf"
    ),
    top_k: int = typer.Option(
        5, "--top-k", "-k",
        min=1, help="Maximum number of results"
    ),
) -> None:
    """Search indexed documents."""

    if not index_path.is_file():
        console.print("[red]Index file not found.[/red]")
        raise typer.Exit(code=1)

    scorer = scorer.lower()

    if scorer not in ("bm25", "tfidf"):
        console.print("[red]Scorer must be bm25 or tfidf.[/red]")
        raise typer.Exit(code=1)

    # Import here to avoid importing the search module
    # when running unrelated CLI commands.
    from findex.search import execute_search

    index = open_index(str(index_path))

    ranking = (
        BM25Scorer()
        if scorer == "bm25"
        else TfIdfScorer()
    )

    results = execute_search(
        index, query, ranking, top_k=top_k
    )

    console.print(
        f"\n[bold green]Found {len(results)} results[/bold green]\n"
    )

    for result in results:
        console.print(
            f"[cyan]Doc {result.doc_id}[/cyan] | "
            f"Score: {result.score:.4f}"
        )
        console.print(f"File: {result.filename}")
        console.print(f"Snippet: {result.snippet}\n")


@app.command()
def stats(
    index_path: Path = typer.Option(
        Path("index.pkl"), "--index", "-i",
        help="Saved index file"
    ),
) -> None:
    """Show inverted index statistics."""

    if not index_path.is_file():
        console.print("[red]Index file not found.[/red]")
        raise typer.Exit(code=1)

    index = open_index(str(index_path))

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
        total_tokens / documents if documents else 0
    )

    console.print("\n[bold cyan]Findex Statistics[/bold cyan]")
    console.print(f"Documents: {documents}")
    console.print(f"Unique terms: {terms}")
    console.print(f"Total tokens: {total_tokens}")
    console.print(f"Total postings: {total_postings}")
    console.print(f"Average document length: {average_length:.2f}")
    console.print(f"Index file size: {index_path.stat().st_size} bytes")


@app.command()
def crawl(
    start_url: str = typer.Argument(
        ..., help="Starting URL to crawl"
    ),
    out_jsonl: Path = typer.Option(
        Path("corpus.jsonl"), "--out", "-o",
        help="Output JSONL file path"
    ),
    max_pages: int = typer.Option(
        200, "--max-pages", "-m",
        help="Maximum pages to collect"
    ),
    concurrency: int = typer.Option(
        5, "--concurrency", "-c",
        help="Concurrency level"
    ),
) -> None:
    """Run async web crawler and save pages to JSONL."""

    console.print(
        f"[bold cyan]Starting crawler for {start_url} "
        f"(max_pages={max_pages}, concurrency={concurrency})"
        f"[/bold cyan]"
    )

    async def run() -> None:
        t0 = time.perf_counter()

        async def write_line(line: str) -> None:
            await asyncio.to_thread(
                append_line,
                out_jsonl,
                line,
            )

        # Створюємо або очищуємо файл перед початком збору.
        await asyncio.to_thread(
            out_jsonl.write_text,
            "",
            encoding="utf-8",
        )

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

            await write_line(line)

            console.print(
                f"[{page['doc_id']}/{max_pages}] "
                f"Collected: {page['url']}"
            )

        elapsed = time.perf_counter() - t0

        console.print(
            f"[bold green]Crawled successfully "
            f"in {elapsed:.2f}s. "
            f"Saved to {out_jsonl}[/bold green]"
        )

    asyncio.run(run())


def append_line(path: Path, line: str) -> None:
    """Додає один JSONL-запис до файлу."""

    with path.open("a", encoding="utf-8") as file:
        file.write(line)


@app.command()
def serve(
    port: int = typer.Option(
        8000, "--port", "-p",
        help="Port to run server"
    ),
    workers: int = typer.Option(
        1, "--workers", "-w",
        help="Number of uvicorn workers"
    ),
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
