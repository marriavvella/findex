
import httpx
import pytest

from findex import crawler


@pytest.fixture
def anyio_backend():
    return "asyncio"


def make_client(handler):
    """Create a mock HTTP client without real network access."""
    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler)
    )


def test_canonicalize_relative_url():
    result = crawler.canonicalize_url(
        "https://example.com/page/",
        "../about",
    )
    assert result == "https://example.com/about"


def test_canonicalize_removes_fragment():
    result = crawler.canonicalize_url(
        "https://example.com/page",
        "https://example.com/page#section",
    )
    assert result == "https://example.com/page"


def test_canonicalize_rejects_invalid_scheme():
    result = crawler.canonicalize_url(
        "https://example.com",
        "mailto:test@example.com",
    )
    assert result is None


def test_canonicalize_sorts_query():
    result = crawler.canonicalize_url(
        "https://example.com",
        "/page?z=2&a=1",
    )
    assert result == "https://example.com/page?a=1&z=2"


def test_canonicalize_normalizes_host_and_slash():
    result = crawler.canonicalize_url(
        "https://EXAMPLE.COM",
        "https://EXAMPLE.COM/about/",
    )
    assert result == "https://example.com/about"


def test_robots_check_disallow():
    cache = crawler.RobotsCache()
    robots = "User-agent: *\nDisallow: /private"

    assert cache._check_txt(robots, "/private/page") is False
    assert cache._check_txt(robots, "/public/page") is True


def test_robots_check_empty():
    cache = crawler.RobotsCache()
    assert cache._check_txt("", "/any-page") is True


@pytest.mark.anyio
async def test_robots_cache_reuses_response():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(
            200,
            text="User-agent: *\nDisallow: /private",
        )

    async with make_client(handler) as client:
        cache = crawler.RobotsCache()

        first = await cache.allowed(
            client,
            "https://example.com/public",
            "FindexBot",
        )
        second = await cache.allowed(
            client,
            "https://example.com/private/page",
            "FindexBot",
        )

    assert first is True
    assert second is False
    assert len(calls) == 1


@pytest.mark.anyio
async def test_robots_denies_when_request_fails():
    def handler(request):
        raise httpx.ConnectError(
            "Connection failed",
            request=request,
        )

    async with make_client(handler) as client:
        result = await crawler.RobotsCache().allowed(
            client,
            "https://example.com/page",
            "FindexBot",
        )

    assert result is False


@pytest.mark.anyio
async def test_fetch_page_success():
    def handler(request):
        return httpx.Response(
            200,
            text="<html><body>Hello</body></html>",
        )

    async with make_client(handler) as client:
        result = await crawler.fetch_page(
            client,
            "https://example.com",
            "FindexBot",
        )

    assert result["status"] == 200
    assert "Hello" in result["html"]
    assert result["error"] is None
    assert result["duration"] >= 0
    assert result["bytes_received"] > 0


@pytest.mark.anyio
async def test_fetch_page_http_error():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(404, text="Not found")

    async with make_client(handler) as client:
        result = await crawler.fetch_page(
            client,
            "https://example.com/missing",
            "FindexBot",
        )

    assert result["status"] == 404
    assert result["html"] == ""
    assert result["error"] is None
    assert len(calls) == 1


@pytest.mark.anyio
async def test_fetch_page_retries_429(monkeypatch):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if len(calls) == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "0"},
            )
        return httpx.Response(200, text="Recovered")

    async def no_sleep(delay):
        return None

    monkeypatch.setattr(crawler.asyncio, "sleep", no_sleep)

    async with make_client(handler) as client:
        result = await crawler.fetch_page(
            client,
            "https://example.com",
            "FindexBot",
        )

    assert result["status"] == 200
    assert len(calls) == 2


@pytest.mark.anyio
async def test_crawl_site_extracts_links():
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(
                200,
                text="User-agent: *",
            )

        return httpx.Response(
            200,
            text="""
            <html>
                <body>
                    <a href="/about">About</a>
                    <a href="https://example.com/contact">
                        Contact
                    </a>
                </body>
            </html>
            """,
        )

    pages = []

    async with make_client(handler) as client:
        async for page in crawler.crawl_site(
            "https://example.com",
            max_pages=1,
            client=client,
        ):
            pages.append(page)

    assert len(pages) == 1
    assert pages[0]["doc_id"] == 1
    assert pages[0]["url"] == "https://example.com/"
    assert pages[0]["links_count"] == 2
    assert "<html>" in pages[0]["html"]


@pytest.mark.anyio
async def test_crawl_respects_max_pages():
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(
                200,
                text="User-agent: *",
            )

        links = "".join(
            f'<a href="/page/{i}">Page {i}</a>'
            for i in range(20)
        )

        return httpx.Response(
            200,
            text=f"<html><body>{links}</body></html>",
        )

    pages = []

    async with make_client(handler) as client:
        async for page in crawler.crawl_site(
            "https://example.com",
            max_pages=5,
            concurrency=5,
            host_delay=0,
            client=client,
        ):
            pages.append(page)

    assert len(pages) == 5
    assert len({page["url"] for page in pages}) == 5
