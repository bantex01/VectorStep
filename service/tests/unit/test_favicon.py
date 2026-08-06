"""GET /favicon.svg — serves the VectorStep mark so it shows up as the browser
tab icon; base.html links to it via <link rel="icon">."""
import httpx
import pytest

from src.main import app


@pytest.fixture
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


async def test_favicon_served_as_svg(client):
    resp = await client.get("/favicon.svg")

    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("image/svg+xml")
    assert "<svg" in resp.text
