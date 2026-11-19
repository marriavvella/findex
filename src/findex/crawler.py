
import asyncio
import logging
import random
import time
from collections.abc import AsyncIterator
from typing import TypedDict
from urllib.parse import urljoin, urlparse, urlunparse

import httpx
from selectolax.lexbor import LexborHTMLParser as HTMLParser


logger = logging.getLogger("findex.crawler")


class FetchResult(TypedDict):
    url: str
    status: int
    html: str
    duration: float
    error: str | None


class CrawledPage(TypedDict):
    doc_id: int
    url: str
    html: str
    links_count: int


class RobotsCache:
    def __init__(self) -> None:
        self._cache: dict[str, str] = {}

    async def allowed(
        self,
        client: httpx.AsyncClient,
        url: str,
        user_agent: str,
    ) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"

        if base in self._cache:
            return self._check_txt(self._cache[base], parsed.path)

        robots_url = f"{base}/robots.txt"

        try:
            resp = await client.get(robots_url, timeout=5.0)

            if resp.status_code == 200:
                txt = resp.text
                self._cache[base] = txt
                return self._check_txt(txt, parsed.path)

        except Exception:
            pass

        return True

    def _check_txt(self, txt: str, path: str) -> bool:
        for line in txt.splitlines():
            line = line.strip()

            if line.lower().startswith("disallow:"):
                disallow_path = line.split(":", 1)[1].strip()

                if disallow_path and path.startswith(disallow_path):
                    return False

        return True


def canonicalize_url(base_url: str, href: str) -> str | None:
    try:
        joined = urljoin(base_url, href)
        parsed = urlparse(joined)

        if parsed.scheme not in ("http", "https"):
            return None

        clean = (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path or "/",
            parsed.params,
            parsed.query,
            "",
        )

        return urlunparse(clean)

    except Exception:
        return None


async def fetch_page(
    client: httpx.AsyncClient,
    url: str,
    user_agent: str,
) -> FetchResult:
    headers = {"User-Agent": user_agent}
    retries = 3
    delay = 1.0

    for attempt in range(retries + 1):
        t0 = time.perf_counter()

        try:
            async with asyncio.timeout(10.0):
                resp = await client.get(
                    url,
                    headers=headers,
                    follow_redirects=True,
                )

                duration = time.perf_counter() - t0

                if resp.status_code in (429, 500, 502, 503, 504):
                    if attempt < retries:
                        retry_after = resp.headers.get("Retry-After")

                        sleep_time = (
                            float(retry_after)
                            if retry_after and retry_after.isdigit()
                            else delay + random.uniform(0, 0.5)
                        )

                        logger.warning(
                            "Status %s for %s. Retrying in %.2fs...",
                            resp.status_code,
                            url,
                            sleep_time,
                        )

                        await asyncio.sleep(sleep_time)
                        delay *= 2
                        continue

                return {
                    "url": url,
                    "status": resp.status_code,
                    "html": resp.text if resp.status_code == 200 else "",
                    "duration": duration,
                    "error": None,
                }

        except asyncio.TimeoutError:
            if attempt == retries:
                return {
                    "url": url,
                    "status": 0,
                    "html": "",
                    "duration": 10.0,
                    "error": "TimeoutError",
                }

        except Exception as exc:
            if attempt == retries:
                return {
                    "url": url,
                    "status": 0,
                    "html": "",
                    "duration": 0.0,
                    "error": str(exc),
                }

        await asyncio.sleep(delay + random.uniform(0, 0.5))
        delay *= 2

    return {
        "url": url,
        "status": 0,
        "html": "",
        "duration": 0.0,
        "error": "Max retries exceeded",
    }


async def crawl_site(
    start_url: str,
    max_pages: int = 200,
    concurrency: int = 5,
    user_agent: str = "FindexBot/1.0 (contact: student@kpi.ua)",
) -> AsyncIterator[CrawledPage]:
    seen: set[str] = set()

    queue: asyncio.Queue[str] = asyncio.Queue()
    queue.put_nowait(start_url)

    seen.add(start_url)

    robots_cache = RobotsCache()
    global_semaphore = asyncio.Semaphore(concurrency)
    host_locks: dict[str, asyncio.Lock] = {}

    pages_collected = 0

    async with httpx.AsyncClient() as client:
        while not queue.empty() and pages_collected < max_pages:
            try:
                url = queue.get_nowait()

            except asyncio.QueueEmpty:
                break

            parsed_host = urlparse(url).netloc

            if parsed_host not in host_locks:
                host_locks[parsed_host] = asyncio.Lock()

            async with global_semaphore:
                if not await robots_cache.allowed(
                    client,
                    url,
                    user_agent,
                ):
                    queue.task_done()
                    continue

                async with host_locks[parsed_host]:
                    res = await fetch_page(
                        client,
                        url,
                        user_agent,
                    )

                    await asyncio.sleep(0.2)

                if res["error"] or res["status"] != 200:
                    logger.error(
                        "Failed URL: %s | Status: %s | Error: %s",
                        url,
                        res["status"],
                        res["error"],
                    )

                    queue.task_done()
                    continue

                pages_collected += 1
                html_content = res["html"]

                parser = HTMLParser(html_content)

                links: list[str] = []

                for node in parser.css("a"):
                    href = node.attributes.get("href")

                    if href:
                        canon = canonicalize_url(url, href)

                        if canon and canon not in seen:
                            seen.add(canon)
                            queue.put_nowait(canon)
                            links.append(canon)

                yield {
                    "doc_id": pages_collected,
                    "url": url,
                    "html": html_content,
                    "links_count": len(links),
                }

                queue.task_done()
