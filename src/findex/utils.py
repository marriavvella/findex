
import time
import functools
import weakref
import re
from typing import List

from findex.query_parser import parse_query


def timed(func):
    """Декоратор для вимірювання часу виконання функції."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()

        result = func(*args, **kwargs)

        elapsed = (time.perf_counter() - t0) * 1000

        print(
            f"[LOG @timed] {func.__name__} "
            f"executed in {elapsed:.3f} ms"
        )

        return result

    return wrapper


# Кеші прив'язані до окремих об'єктів індексу.
# WeakKeyDictionary не утримує непотрібні індекси.
_search_cache = weakref.WeakKeyDictionary()

CACHE_SIZE = 256


def _create_cache(index):
    """
    Створює LRU-кеш для конкретного індексу.

    Слабке посилання не дозволяє кешу
    утримувати об'єкт індексу в пам'яті.
    """

    index_ref = weakref.ref(index)

    @functools.lru_cache(maxsize=CACHE_SIZE)
    def search(query_str: str):
        current_index = index_ref()

        if current_index is None:
            return ()

        ast_root = parse_query(query_str)

        return tuple(
            sorted(ast_root.evaluate(current_index))
        )

    return search


def cached_search_ids(index, query_str: str) -> List[int]:
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
            search
        )
    else:
        search = state[1]

    hits_before = search.cache_info().hits

    doc_ids = search(query_str)

    hits_after = search.cache_info().hits

    if hits_after > hits_before:
        print(f"[CACHE HIT] {query_str}")
    else:
        print(f"[CACHE MISS] {query_str}")

    return list(doc_ids)


def generate_snippet(
    text: str,
    query_terms: List[str],
    window: int = 80
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

    # Видаляємо порожні та повторювані терміни.
    terms = list(
        dict.fromkeys(
            term.strip()
            for term in query_terms
            if term and term.strip()
        )
    )

    # Якщо термінів немає, повертаємо початок тексту.
    if not terms:
        snippet = text[:window * 2]

        if len(text) > window * 2:
            snippet += "..."

        return snippet

    # Формуємо регулярний вираз для пошуку цілих слів.
    # Довші терміни перевіряємо першими.
    escaped_terms = [
        re.escape(term)
        for term in sorted(
            terms,
            key=len,
            reverse=True
        )
    ]

    pattern = re.compile(
        r"(?<!\w)(?:"
        + "|".join(escaped_terms)
        + r")(?!\w)",
        flags=re.IGNORECASE
    )

    # Знаходимо перший збіг у документі.
    match = pattern.search(text)

    if match is None:
        snippet = text[:window * 2]

        if len(text) > window * 2:
            snippet += "..."

        return snippet

    # Визначаємо межі фрагмента навколо збігу.
    start = max(
        0,
        match.start() - window
    )

    end = min(
        len(text),
        match.end() + window
    )

    snippet = text[start:end]

    # Підсвічуємо всі цілі слова у фрагменті.
    highlighted = pattern.sub(
        lambda found: f"**{found.group(0)}**",
        snippet
    )

    # Позначаємо пропущений текст.
    if start > 0:
        highlighted = "..." + highlighted

    if end < len(text):
        highlighted += "..."

    return highlighted
