import httpx

from app.be_core.config import settings
from app.tools.system_tools.schemas.websearch_schema import (
    WebSearchInput,
    WebSearchOutput,
    WebSearchResult,
)


async def perform_web_search(input: WebSearchInput) -> WebSearchOutput:
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": settings.BRAVE_SEARCH_API_KEY,
    }

    params = {
        "q": input.query,
        "count": 5,
    }

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers=headers,
            params=params,
        )

    if response.status_code != 200:
        raise Exception(f"Brave API error: {response.status_code}, {response.text}")

    data = response.json()
    results = [
        WebSearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=item.get("description", ""),
        )
        for item in data.get("web", {}).get("results", [])
    ]

    return WebSearchOutput(results=results)
