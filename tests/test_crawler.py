
import pytest
import httpx

from findex import crawler


@pytest.fixture
def anyio_backend():
    return "asyncio"


class FakeResponse:
    def __init__(self, status_code, text="", headers=None):
        self.status_code = status_code
        self.text = text
        self.headers = headers or {}


class FakeClient:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    async def get(self, url, **kwargs):
        self.calls.append(url)

        response = self.responses.pop(0)

        if isinstance(response, Exception):
            raise response

        return response


def test_canonicalize_relative_url():
    """Перевіряє перетворення відносного URL."""

    result = crawler.canonicalize_url(
        "https://example.com/page/",
        "../about",
    )

    assert result == "https://example.com/about"


def test_canonicalize_removes_fragment():
    """Перевіряє видалення фрагмента URL."""

    result = crawler.canonicalize_url(
        "https://example.com/page",
        "https://example.com/page#section",
    )

    assert result == "https://example.com/page"


def test_canonicalize_rejects_invalid_scheme():
    """Перевіряє відхилення непідтримуваних схем."""

    result = crawler.canonicalize_url(
        "https://example.com",
        "mailto:test@example.com",
    )

    assert result is None


def test_robots_check_disallow():
    """Перевіряє заборону доступу через robots.txt."""

    cache = crawler.RobotsCache()

    robots = """
User-agent: *
Disallow: /private
"""

    assert cache._check_txt(robots, "/private/page") is False
    assert cache._check_txt(robots, "/public/page") is True


def test_robots_check_empty():
    """Перевіряє доступ за відсутності заборон."""

    cache = crawler.RobotsCache()

    assert cache._check_txt("", "/any-page") is True


@pytest.mark.anyio
async def test_robots_cache_reuses_response():
    """Перевіряє повторне використання robots.txt."""

    cache = crawler.RobotsCache()

    client = FakeClient([
        FakeResponse(
            200,
            "User-agent: *\nDisallow: /private",
        )
    ])

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
    assert len(client.calls) == 1


@pytest.mark.anyio
async def test_robots_allows_when_request_fails():
    """Перевіряє доступність сторінки при помилці robots.txt."""

    cache = crawler.RobotsCache()

    client = FakeClient([
        RuntimeError("Connection failed")
    ])

    result = await cache.allowed(
        client,
        "https://example.com/page",
        "FindexBot",
    )

    assert result is True


@pytest.mark.anyio
async def test_fetch_page_success():
    """Перевіряє успішне отримання HTML-сторінки."""

    client = FakeClient([
        FakeResponse(200, "<html><body>Hello</body></html>")
    ])

    result = await crawler.fetch_page(
        client,
        "https://example.com",
        "FindexBot",
    )

    assert result["status"] == 200
    assert "Hello" in result["html"]
    assert result["error"] is None
    assert result["duration"] >= 0


@pytest.mark.anyio
async def test_fetch_page_http_error():
    """Перевіряє обробку HTTP-помилки без повтору."""

    client = FakeClient([
        FakeResponse(404, "Not found")
    ])

    result = await crawler.fetch_page(
        client,
        "https://example.com/missing",
        "FindexBot",
    )

    assert result["status"] == 404
    assert result["html"] == ""
    assert result["error"] is None


@pytest.mark.anyio
async def test_crawl_site_extracts_links(monkeypatch):
    """Перевіряє збір сторінки та витягування посилань."""

    class FakeAsyncClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def get(self, url, **kwargs):
            if url.endswith("/robots.txt"):
                return FakeResponse(200, "User-agent: *")

            return FakeResponse(
                200,
                """
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

    async def no_sleep(delay):
        return None

    monkeypatch.setattr(
        crawler.httpx,
        "AsyncClient",
        FakeAsyncClient,
    )

    monkeypatch.setattr(
        crawler.asyncio,
        "sleep",
        no_sleep,
    )

    pages = []

    async for page in crawler.crawl_site(
        "https://example.com",
        max_pages=1,
    ):
        pages.append(page)

    assert len(pages) == 1
    assert pages[0]["doc_id"] == 1
    assert pages[0]["url"] == "https://example.com"
    assert pages[0]["links_count"] == 2
    assert "<html>" in pages[0]["html"]
