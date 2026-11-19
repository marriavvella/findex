
import functools
import logging
import re
import time
import weakref
from collections.abc import Callable, Sequence
from typing import Protocol

from findex.index import Index
from findex.query_parser import parse_query

logger = logging.getLogger(__name__)


def timed[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Декоратор для вимірювання часу виконання функції."""

    @functools.wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        t0 = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = (time.perf_counter() - t0) * 1000

        logger.debug(
            "[LOG @timed] %s executed in %.3f ms",
            func.__name__,
            elapsed,
        )

        return result

    return wrapper


class CacheInfo(Protocol):
    """Інтерфейс статистики LRU-кешу."""

    @property
    def hits(self) -> int:
        ...


class SearchCache(Protocol):
    """Інтерфейс функції пошуку з LRU-кешем."""

    def __call__(self, query_str: str) -> tuple[int, ...]:
        ...

    def cache_info(self) -> CacheInfo:
        ...


_search_cache: weakref.WeakKeyDictionary[
    Index,
    tuple[object, SearchCache],
] = weakref.WeakKeyDictionary()

CACHE_SIZE = 256


def _create_cache(index: Index) -> SearchCache:
    """
    Створює LRU-кеш для конкретного індексу.

    Слабке посилання не дозволяє кешу
    утримувати об'єкт індексу в пам'яті.
    """

    index_ref: weakref.ReferenceType[Index] = weakref.ref(index)

    @functools.lru_cache(maxsize=CACHE_SIZE)
    def search(query_str: str) -> tuple[int, ...]:
        current_index = index_ref()

        if current_index is None:
            return ()

        ast_root = parse_query(query_str)

        return tuple(sorted(ast_root.evaluate(current_index)))

    return search


def cached_search_ids(
    index: Index,
    query_str: str,
) -> list[int]:
    """
    Повертає ID документів із використанням LRU-кешу.

    Повторні запити обслуговуються з кешу.
    Після перебудови індексу кеш оновлюється.
    """

    index_version = index.index
    state = _search_cache.get(index)

    if state is None or state[0] is not index_version:
        search = _create_cache(index)

        _search_cache[index] = (
            index_version,
            search,
        )
    else:
        search = state[1]

    hits_before = search.cache_info().hits

    doc_ids = search(query_str)

    hits_after = search.cache_info().hits

    if hits_after > hits_before:
        logger.debug("[CACHE HIT] %s", query_str)
    else:
        logger.debug("[CACHE MISS] %s", query_str)

    return list(doc_ids)


def generate_snippet(
    text: str,
    query_terms: Sequence[str],
    window: int = 80,
) -> str:
    """
    Створює фрагмент документа навколо знайденого слова.

    Підсвічує лише цілі слова, а не їхні частини.
    Пошук виконується без урахування регістру.
    """

    if not text:
        return ""

    if window < 1:
        raise ValueError(
            "window must be greater than zero"
        )

    terms: list[str] = list(
        dict.fromkeys(
            term.strip()
            for term in query_terms
            if term and term.strip()
        )
    )

    if not terms:
        snippet = text[:window * 2]

        if len(text) > window * 2:
            snippet += "..."

        return snippet

    escaped_terms: list[str] = [
        re.escape(term)
        for term in sorted(
            terms,
            key=len,
            reverse=True,
        )
    ]

    pattern = re.compile(
        r"(?<!\w)(?:"
        + "|".join(escaped_terms)
        + r")(?!\w)",
        flags=re.IGNORECASE,
    )

    match = pattern.search(text)

    if match is None:
        snippet = text[:window * 2]

        if len(text) > window * 2:
            snippet += "..."

        return snippet

    start = max(
        0,
        match.start() - window,
    )

    end = min(
        len(text),
        match.end() + window,
    )

    snippet = text[start:end]

    highlighted = pattern.sub(
        lambda found: f"**{found.group(0)}**",
        snippet,
    )

    if start > 0:
        highlighted = "..." + highlighted

    if end < len(text):
        highlighted += "..."

    return highlighted
