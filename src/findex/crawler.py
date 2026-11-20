
"""Asynchronous web crawler for the findex project."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections import defaultdict
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC
from email.utils import parsedate_to_datetime
from typing import Any, TypedDict
from urllib.parse import (
    parse_qsl,
    urlencode,
    urljoin,
    urlsplit,
    urlunsplit,
)
from urllib.robotparser import RobotFileParser

import httpx
from selectolax.lexbor import LexborHTMLParser as HTMLParser

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = "FindexBot/1.0 (educational crawler)"
RETRY_STATUSES = {429, 500, 502, 503, 504}


class FetchResult(TypedDict):
    url: str
    status: int | None
    html: str
    duration: float
    error: str | None
    bytes_received: int


class CrawledPage(TypedDict):
    doc_id: int
    url: str
    html: str
    links_count: int


@dataclass
class CrawlStats:
    """Mutable statistics for a crawling session."""

    started_at: float = field(default_factory=time.perf_counter)
    pages: int = 0
    errors: int = 0
    active_requests: int = 0
    queue_depth: int = 0

    @property
    def elapsed(self) -> float:
        return max(0.0, time.perf_counter() - self.started_at)

    @property
    def pages_per_second(self) -> float:
        elapsed = self.elapsed
        return self.pages / elapsed if elapsed > 0 else 0.0


def canonicalize_url(
    base_url: str,
    link: str,
) -> str | None:
    """Resolve and normalize an HTTP(S) URL."""

    try:
        absolute = urljoin(base_url, link.strip())
        parsed = urlsplit(absolute)

        scheme = parsed.scheme.lower()

        if scheme not in {"http", "https"}:
            return None

        hostname = parsed.hostname

        if not hostname:
            return None

        hostname = hostname.lower()

        try:
            port = parsed.port
        except ValueError:
            return None

        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"

        if (
            port is not None
            and not (
                (scheme == "http" and port == 80)
                or (scheme == "https" and port == 443)
            )
        ):
            netloc = f"{hostname}:{port}"
        else:
            netloc = hostname

        path = parsed.path or "/"

        if path != "/":
            path = path.rstrip("/")

        query_items = parse_qsl(
            parsed.query,
            keep_blank_values=True,
        )
        query_items.sort()
        query = urlencode(query_items)

        return urlunsplit(
            (scheme, netloc, path, query, "")
        )

    except (TypeError, ValueError):
        return None


class RobotsCache:
    """Cache robots.txt rules separately for each origin."""

    def __init__(self) -> None:
        self._cache: dict[str, str | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def _origin(url: str) -> str:
        parsed = urlsplit(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def _check_txt(
        self,
        robots_text: str,
        path: str,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> bool:
        parser = RobotFileParser()
        parser.parse(robots_text.splitlines())

        return parser.can_fetch(user_agent, path)

    async def allowed(
        self,
        client: httpx.AsyncClient,
        url: str,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> bool:
        """Return False if robots.txt forbids or cannot be checked."""

        origin = self._origin(url)

        if origin not in self._locks:
            self._locks[origin] = asyncio.Lock()

        async with self._locks[origin]:
            if origin not in self._cache:
                robots_url = f"{origin}/robots.txt"

                try:
                    async with asyncio.timeout(10):
                        response = await client.get(
                            robots_url,
                            headers={"User-Agent": user_agent},
                            follow_redirects=True,
                        )

                    if response.status_code == 404:
                        self._cache[origin] = ""
                    elif response.status_code == 200:
                        self._cache[origin] = response.text
                    else:
                        logger.warning(
                            "robots.txt unavailable: %s status=%s",
                            robots_url,
                            response.status_code,
                        )
                        self._cache[origin] = None

                except (
                    httpx.HTTPError,
                    TimeoutError,
                    OSError,
                ) as exc:
                    logger.warning(
                        "robots.txt request failed: %s: %s",
                        robots_url,
                        exc,
                    )
                    self._cache[origin] = None

            robots_text = self._cache[origin]

        if robots_text is None:
            return False

        return self._check_txt(
            robots_text,
            url,
            user_agent,
        )


def _retry_after_seconds(
    value: str | None,
) -> float | None:
    """Parse Retry-After seconds or an HTTP date."""

    if not value:
        return None

    try:
        return max(0.0, float(value))
    except ValueError:
        pass

    try:
        from datetime import datetime

        retry_date = parsedate_to_datetime(value)

        if retry_date.tzinfo is None:
            retry_date = retry_date.replace(
                tzinfo=UTC
            )

        now = datetime.now(UTC)

        return max(
            0.0,
            (retry_date - now).total_seconds(),
        )

    except (TypeError, ValueError, OverflowError):
        return None


async def fetch_page(
    client: httpx.AsyncClient,
    url: str,
    user_agent: str = DEFAULT_USER_AGENT,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
) -> FetchResult:
    """Fetch HTML with bounded retries for temporary failures."""

    started = time.perf_counter()
    last_status: int | None = None
    last_error: str | None = None

    for attempt in range(max_retries + 1):
        retry_after: float | None = None

        try:
            async with asyncio.timeout(timeout):
                response = await client.get(
                    url,
                    headers={"User-Agent": user_agent},
                    follow_redirects=True,
                )

            status = response.status_code
            last_status = status

            size = len(response.content)

            logger.info(
                "FETCH url=%s status=%s bytes=%s elapsed=%.3f",
                url,
                status,
                size,
                time.perf_counter() - started,
            )

            if status == 200:
                return {
                    "url": str(response.url),
                    "status": status,
                    "html": response.text,
                    "duration": time.perf_counter() - started,
                    "error": None,
                    "bytes_received": size,
                }

            if status not in RETRY_STATUSES:
                return {
                    "url": url,
                    "status": status,
                    "html": "",
                    "duration": time.perf_counter() - started,
                    "error": None,
                    "bytes_received": size,
                }

            retry_after = _retry_after_seconds(
                response.headers.get("Retry-After")
            )

            last_error = f"HTTP {status}"

        except (
            TimeoutError,
            httpx.TimeoutException,
            httpx.TransportError,
        ) as exc:
            last_error = str(exc) or type(exc).__name__

            logger.warning(
                "FETCH url=%s error=%s attempt=%s",
                url,
                last_error,
                attempt + 1,
            )

        if attempt >= max_retries:
            break

        backoff = min(8.0, 0.25 * (2 ** attempt))
        jitter = random.uniform(0.0, 0.1)

        delay = (
            retry_after
            if retry_after is not None
            else backoff + jitter
        )

        logger.info(
            "RETRY url=%s attempt=%s delay=%.3f",
            url,
            attempt + 1,
            delay,
        )

        await asyncio.sleep(delay)

    return {
        "url": url,
        "status": last_status,
        "html": "",
        "duration": time.perf_counter() - started,
        "error": last_error,
        "bytes_received": 0,
    }


def _extract_links(
    html: str,
    base_url: str,
) -> list[str]:
    """Extract unique normalized HTTP(S) links."""

    document = HTMLParser(html)
    links: list[str] = []
    seen: set[str] = set()

    for node in document.css("a[href]"):
        href = node.attributes.get("href")

        if not href:
            continue

        normalized = canonicalize_url(
            base_url,
            href,
        )

        if normalized and normalized not in seen:
            seen.add(normalized)
            links.append(normalized)

    return links


async def crawl_site(
    start_url: str,
    *,
    max_pages: int = 200,
    concurrency: int = 5,
    per_host_concurrency: int = 2,
    host_delay: float = 0.2,
    user_agent: str = DEFAULT_USER_AGENT,
    client: httpx.AsyncClient | None = None,
    stats: CrawlStats | None = None,
) -> AsyncIterator[CrawledPage]:
    """Crawl one origin using an asyncio TaskGroup and URL queue.

    Only use this function on sites you own or are permitted to crawl.
    """

    start = canonicalize_url(
        start_url,
        start_url,
    )

    if start is None:
        raise ValueError("start_url must be HTTP or HTTPS")

    if max_pages < 1:
        raise ValueError("max_pages must be positive")

    if concurrency < 1:
        raise ValueError("concurrency must be positive")

    if per_host_concurrency < 1:
        raise ValueError(
            "per_host_concurrency must be positive"
        )

    if host_delay < 0:
        raise ValueError("host_delay cannot be negative")

    origin = urlsplit(start).netloc

    own_client = client is None

    if own_client:
        client = httpx.AsyncClient(
            follow_redirects=True,
            timeout=10.0,
        )

    assert client is not None

    crawl_stats = stats or CrawlStats()
    robots = RobotsCache()

    pending: asyncio.Queue[str | None] = asyncio.Queue()
    results: asyncio.Queue[
        CrawledPage | object
    ] = asyncio.Queue()

    finished = object()

    seen: set[str] = {start}
    pending.put_nowait(start)

    global_limit = asyncio.Semaphore(concurrency)
    host_limits: defaultdict[
        str, asyncio.Semaphore
    ] = defaultdict(
        lambda: asyncio.Semaphore(
            per_host_concurrency
        )
    )

    host_locks: defaultdict[
        str, asyncio.Lock
    ] = defaultdict(asyncio.Lock)

    last_request: dict[str, float] = {}
    page_number = 0

    async def worker() -> None:
        nonlocal page_number

        while True:
            url = await pending.get()
            crawl_stats.queue_depth = pending.qsize()

            try:
                if url is None:
                    return

                allowed = await robots.allowed(
                    client,
                    url,
                    user_agent,
                )

                if not allowed:
                    logger.info(
                        "SKIP url=%s reason=robots",
                        url,
                    )
                    continue

                host = urlsplit(url).netloc

                async with global_limit, host_limits[host]:
                    async with host_locks[host]:
                        previous = last_request.get(
                            host,
                            0.0,
                        )
                        remaining = (
                            host_delay
                            - (
                                time.perf_counter()
                                - previous
                            )
                        )

                        if remaining > 0:
                            await asyncio.sleep(remaining)

                        last_request[host] = (
                            time.perf_counter()
                        )

                    crawl_stats.active_requests += 1

                    try:
                        result = await fetch_page(
                            client,
                            url,
                            user_agent,
                        )
                    finally:
                        crawl_stats.active_requests -= 1

                if result["status"] != 200:
                    crawl_stats.errors += 1
                    logger.warning(
                        "CRAWL url=%s status=%s error=%s",
                        url,
                        result["status"],
                        result["error"],
                    )
                    continue

                page_url = canonicalize_url(
                    result["url"],
                    result["url"],
                ) or url

                if urlsplit(page_url).netloc != origin:
                    logger.info(
                        "SKIP redirected outside origin: %s",
                        page_url,
                    )
                    continue

                links = _extract_links(
                    result["html"],
                    page_url,
                )

                page_number += 1
                crawl_stats.pages += 1

                await results.put(
                    {
                        "doc_id": page_number,
                        "url": page_url,
                        "html": result["html"],
                        "links_count": len(links),
                    }
                )

                for link in links:
                    if urlsplit(link).netloc != origin:
                        continue

                    if link in seen:
                        continue

                    if len(seen) >= max_pages:
                        break

                    seen.add(link)
                    pending.put_nowait(link)

                crawl_stats.queue_depth = pending.qsize()

            except asyncio.CancelledError:
                raise

            except Exception:
                crawl_stats.errors += 1
                logger.exception(
                    "Unexpected crawler error: %s",
                    url,
                )

            finally:
                pending.task_done()

    async def run_workers() -> None:
        try:
            async with asyncio.TaskGroup() as group:
                workers = [
                    group.create_task(worker())
                    for _ in range(concurrency)
                ]

                await pending.join()

                for _ in workers:
                    pending.put_nowait(None)

        finally:
            await results.put(finished)

    runner: asyncio.Task[Any] | None = None

    try:
        runner = asyncio.create_task(run_workers())

        while True:
            item = await results.get()

            if item is finished:
                break

            yield item  # type: ignore[misc]

        await runner

    finally:
        if runner is not None and not runner.done():
            runner.cancel()

            try:
                await runner
            except asyncio.CancelledError:
                pass

        if own_client:
            await client.aclose()

