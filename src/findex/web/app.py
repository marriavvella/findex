import logging
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

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
    yield
    app.state.index = None

app = FastAPI(title="Findex Web API", version="0.7.0", lifespan=lifespan)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

def get_index(request: Request) -> InvertedIndex:
    idx = getattr(request.app.state, "index", None)
    if idx is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    return idx

@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled exception request_id=%s", request_id)
        response = JSONResponse(status_code=500, content={"detail": "Internal Server Error", "request_id": request_id})
    response.headers["X-Request-ID"] = request_id
    return response

@app.get("/health", response_model=HealthResponse)
def health(idx: InvertedIndex = Depends(get_index)):
    return HealthResponse(status="ok")

@app.get("/stats", response_model=StatsResponse)
def stats(idx: InvertedIndex = Depends(get_index)):
    return StatsResponse(total_terms=len(idx.index), total_documents=len(idx.metadata))

@app.get("/search", response_model=SearchResponse)
def search(q: str = Query(min_length=1, max_length=300), k: int = Query(5, ge=1, le=50), scorer: Literal["bm25", "tfidf"] = "bm25", page: int = Query(1, ge=1, le=10000), idx: InvertedIndex = Depends(get_index)):
    # CPU-bound search runs in FastAPI's thread pool rather than on the event loop.
    ranker = BM25Scorer() if scorer == "bm25" else TfIdfScorer()
    ranked = execute_search(idx, q, ranker, top_k=max(1, len(idx.metadata)))
    start = (page - 1) * k
    return SearchResponse(query=q, page=page, k=k, total=len(ranked), results=[Hit(doc_id=r.doc_id, filename=r.filename, score=r.score, snippet=r.snippet) for r in ranked[start:start+k]])

@app.get("/docs/{doc_id}", response_model=DocumentResponse)
def get_doc(doc_id: int, idx: InvertedIndex = Depends(get_index)):
    meta = idx.metadata.get(doc_id)
    if meta is None:
        raise HTTPException(status_code=404, detail="Document not found")
    path = Path(meta.filename)
    content = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else "Текст документа відсутній на сервері."
    return DocumentResponse(doc_id=meta.doc_id, filename=meta.filename, text=content)

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {})
