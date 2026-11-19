
import logging
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
)
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from findex.index import InvertedIndex
from findex.scoring import (
    BM25Scorer,
    Scorer,
    TfIdfScorer,
)
from findex.search import execute_search
from findex.web.config import settings

logger = logging.getLogger("findex.web")


# Сховище моделей із чітко визначеним типом.
ml_models: dict[str, InvertedIndex | None] = {}


@asynccontextmanager
async def lifespan(
    _app: FastAPI,
) -> AsyncGenerator[None, None]:
    """Завантажує індекс під час запуску вебсервера."""

    idx_path = Path(settings.index_path)

    if idx_path.exists():
        ml_models["index"] = InvertedIndex.load_pickle(
            str(idx_path)
        )
    else:
        ml_models["index"] = None

    yield

    ml_models.clear()


app = FastAPI(
    title="Findex Web API",
    lifespan=lifespan,
)

templates = Jinja2Templates(
    directory="src/findex/web/templates"
)


def get_index() -> InvertedIndex:
    """Повертає завантажений індекс."""

    idx = ml_models.get("index")

    if idx is None:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded",
        )

    return idx


@app.get("/health")
def health(
    _idx: InvertedIndex = Depends(get_index),
) -> dict[str, str]:
    """Перевіряє доступність пошукової системи."""

    return {"status": "ok"}


@app.get("/stats")
def stats(
    idx: InvertedIndex = Depends(get_index),
) -> dict[str, int]:
    """Повертає статистику індексу."""

    return {
        "total_terms": len(idx.index),
        "total_documents": len(idx.metadata),
    }


@app.get("/search")
async def search(
    q: str = Query(..., min_length=1),
    k: int = Query(5, ge=1, le=50),
    scorer_name: Literal["bm25", "tfidf"] = "bm25",
    page: int = Query(1, ge=1),
    idx: InvertedIndex = Depends(get_index),
) -> dict[str, object]:
    """Виконує пошук із ранжуванням та пагінацією."""

    try:
        scorer: Scorer = (
            BM25Scorer()
            if scorer_name == "bm25"
            else TfIdfScorer()
        )

        results = execute_search(
            idx,
            q,
            scorer,
            top_k=k * page,
        )

        # Проста пагінація.
        start = (page - 1) * k
        paginated = results[start:start + k]

        return {
            "query": q,
            "page": page,
            "results": [
                {
                    "doc_id": result.doc_id,
                    "score": result.score,
                    "filename": result.filename,
                }
                for result in paginated
            ],
        }

    except Exception as exc:
        req_id = uuid.uuid4()

        logger.exception(
            "Search request failed. Req ID: %s",
            req_id,
        )

        raise HTTPException(
            status_code=500,
            detail=f"Internal Server Error (ID: {req_id})",
        ) from exc


@app.get("/docs/{doc_id}")
def get_doc(
    doc_id: int,
    idx: InvertedIndex = Depends(get_index),
) -> dict[str, int | str]:
    """Повертає вміст документа за його ідентифікатором."""

    meta = idx.metadata.get(doc_id)

    if meta is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    # Читаємо вміст файлу документа.
    doc_path = Path(meta.filename)

    text = (
        doc_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )
        if doc_path.exists()
        else "File not found on disk."
    )

    return {
        "doc_id": meta.doc_id,
        "filename": meta.filename,
        "text": text,
    }


@app.get("/", response_class=HTMLResponse)
def index_page(request: Request) -> HTMLResponse:
    """Повертає головну HTML-сторінку."""

    return templates.TemplateResponse(
        request,
        "index.html",
        {},
    )
