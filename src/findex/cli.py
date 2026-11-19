
import asyncio
import json
import time
from pathlib import Path

import typer
import uvicorn
from rich.console import Console

from findex.crawler import crawl_site

app = typer.Typer(help="Findex - Python search engine CLI")
console = Console()


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
        with open(out_jsonl, "w", encoding="utf-8") as f:
            t0 = time.perf_counter()

            async for page in crawl_site(
                start_url,
                max_pages=max_pages,
                concurrency=concurrency,
            ):
                f.write(
                    json.dumps(
                        {
                            "doc_id": page["doc_id"],
                            "filename": page["url"],
                            "text": page["html"],
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )

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
