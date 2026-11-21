
import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from findex.index import InvertedIndex
from findex.scoring import BM25Scorer, TfIdfScorer
from findex.search import execute_search
from findex.web.config import settings

logger = logging.getLogger("findex.web")


class Hit(BaseModel):
    doc_id: int
    filename: str
    score: float
    snippet: str = ""


class SearchResponse(BaseModel):
    query: str
    page: int
    k: int
    total: int
    results: list[Hit]


class DocumentResponse(BaseModel):
    doc_id: int
    filename: str
    text: str


class StatsResponse(BaseModel):
    total_terms: int
    total_documents: int


class HealthResponse(BaseModel):
    status: str


def resolve_document_path(filename: str) -> Path | None:
    """Find a document locally or inside the Docker container."""
    original = Path(filename.replace("\\", "/"))

    candidates = [original]

    # The index stores paths such as data/Dracula.txt.
    # On Render the documents are stored in /app/data.
    candidates.append(Path("/app") / original)

    # Also support indexes containing only the document name.
    candidates.append(Path("data") / original.name)
    candidates.append(Path("/app/data") / original.name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


def read_document(filename: str) -> str | None:
    path = resolve_document_path(filename)

    if path is None:
        logger.warning("Document file not found: %s", filename)
        return None

    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        logger.exception("Cannot read document: %s", path)
        return None


def create_snippet(text: str, query: str, length: int = 220) -> str:
    if not text:
        return "Текст недоступний"

    words = re.findall(r"\w+", query, flags=re.UNICODE)
    position = 0

    for word in words:
        match = re.search(re.escape(word), text, flags=re.IGNORECASE)
        if match:
            position = max(0, match.start() - 60)
            break

    snippet = text[position:position + length].replace("\n", " ")
    snippet = re.sub(r"\s+", " ", snippet).strip()

    if position > 0:
        snippet = "..." + snippet

    if position + length < len(text):
        snippet += "..."

    return snippet


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = Path(settings.index_path)
    app.state.index = None

    if path.is_file():
        try:
            app.state.index = InvertedIndex.load_pickle(str(path))
            logger.info("Index loaded from %s", path)
        except Exception:
            logger.exception("Unable to load index from %s", path)
    else:
        logger.error("Index file not found: %s", path)

    yield
    app.state.index = None


app = FastAPI(
    title="Findex Web API",
    version="0.7.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(
    directory=str(Path(__file__).parent / "templates")
)


def get_index(request: Request) -> InvertedIndex:
    idx = getattr(request.app.state, "index", None)

    if idx is None:
        raise HTTPException(status_code=503, detail="Index not loaded")

    return idx


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get(
        "X-Request-ID", str(uuid.uuid4())
    )

    try:
        response = await call_next(request)
    except Exception:
        logger.exception(
            "Unhandled exception request_id=%s", request_id
        )
        response = JSONResponse(
            status_code=500,
            content={
                "detail": "Internal Server Error",
                "request_id": request_id,
            },
        )

    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health", response_model=HealthResponse)
def health(idx: InvertedIndex = Depends(get_index)):
    return HealthResponse(status="ok")


@app.get("/stats", response_model=StatsResponse)
def stats(idx: InvertedIndex = Depends(get_index)):
    return StatsResponse(
        total_terms=len(idx.index),
        total_documents=len(idx.metadata),
    )


@app.get("/search", response_model=SearchResponse)
def search(
    q: str = Query(min_length=1, max_length=300),
    k: int = Query(5, ge=1, le=50),
    scorer: Literal["bm25", "tfidf"] = "bm25",
    page: int = Query(1, ge=1, le=10000),
    idx: InvertedIndex = Depends(get_index),
):
    # A synchronous route runs in FastAPI's thread pool.
    ranker = BM25Scorer() if scorer == "bm25" else TfIdfScorer()

    ranked = execute_search(
        idx,
        q,
        ranker,
        top_k=max(1, len(idx.metadata)),
    )

    start = (page - 1) * k
    selected = ranked[start:start + k]

    results = []

    for result in selected:
        document_text = read_document(result.filename)

        snippet = (
            create_snippet(document_text, q)
            if document_text is not None
            else "Текст недоступний"
        )

        results.append(
            Hit(
                doc_id=result.doc_id,
                filename=result.filename,
                score=result.score,
                snippet=snippet,
            )
        )

    return SearchResponse(
        query=q,
        page=page,
        k=k,
        total=len(ranked),
        results=results,
    )


@app.get("/docs/{doc_id}", response_model=DocumentResponse)
def get_doc(
    doc_id: int,
    idx: InvertedIndex = Depends(get_index),
):
    meta = idx.metadata.get(doc_id)

    if meta is None:
        raise HTTPException(
            status_code=404,
            detail="Document not found",
        )

    content = read_document(meta.filename)

    if content is None:
        raise HTTPException(
            status_code=404,
            detail="Document text not found",
        )

    return DocumentResponse(
        doc_id=meta.doc_id,
        filename=meta.filename,
        text=content,
    )


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {},
    )
