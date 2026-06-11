"""Lightweight web search + fetch for client-side tool use.

Used by the OpenAI path of BasicMemoryAgent to give the model web access
without an Anthropic-style server tool. Search is provider-pluggable
(Brave or Tavily — both have free tiers and need only an HTTP call, no
SDK); fetch is self-contained (httpx + BeautifulSoup), so it costs
nothing and depends on no external service.

All functions are defensive: any failure returns a structured error
string/dict rather than raising, so a flaky network never breaks chat.
"""

from __future__ import annotations

from typing import Any, Dict, List

import httpx
from bs4 import BeautifulSoup
from loguru import logger

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


async def web_search(
    query: str,
    *,
    provider: str = "brave",
    api_key: str = "",
    max_results: int = 5,
) -> List[Dict[str, str]]:
    """Search the web. Returns a list of {title, url, snippet}.

    On error returns a single-element list whose dict has an "error" key,
    so the caller can surface it to the model without special-casing.
    """
    query = (query or "").strip()
    if not query:
        return [{"error": "empty query"}]
    if not api_key:
        return [{"error": f"no API key configured for provider '{provider}'"}]

    try:
        if provider == "brave":
            return await _brave_search(query, api_key, max_results)
        elif provider == "tavily":
            return await _tavily_search(query, api_key, max_results)
        else:
            return [{"error": f"unknown search provider '{provider}'"}]
    except httpx.HTTPStatusError as e:
        logger.warning(f"[web_search] {provider} HTTP {e.response.status_code}")
        return [{"error": f"search failed: HTTP {e.response.status_code}"}]
    except Exception as e:
        logger.warning(f"[web_search] {provider} failed: {type(e).__name__}: {e}")
        return [{"error": f"search failed: {type(e).__name__}"}]


async def _brave_search(
    query: str, api_key: str, max_results: int
) -> List[Dict[str, str]]:
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {"X-Subscription-Token": api_key, "Accept": "application/json"}
    params = {"q": query, "count": max(1, min(max_results, 20))}
    async with httpx.AsyncClient(timeout=15.0) as client:
        resp = await client.get(url, headers=headers, params=params)
        resp.raise_for_status()
        data = resp.json()
    out: List[Dict[str, str]] = []
    for item in (data.get("web", {}) or {}).get("results", [])[:max_results]:
        out.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("description", ""),
            }
        )
    return out or [{"error": "no results"}]


async def _tavily_search(
    query: str, api_key: str, max_results: int
) -> List[Dict[str, str]]:
    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max(1, min(max_results, 20)),
        "search_depth": "basic",
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    out: List[Dict[str, str]] = []
    for item in data.get("results", [])[:max_results]:
        out.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                # Tavily returns cleaned content already; use it as snippet.
                "snippet": (item.get("content", "") or "")[:500],
            }
        )
    return out or [{"error": "no results"}]


async def web_fetch(url: str, *, max_chars: int = 20000) -> Dict[str, Any]:
    """Fetch a URL and return {url, title, text} with cleaned article text.

    Self-contained (httpx + BeautifulSoup). On error returns {url, error}.
    Only handles HTML/text — PDFs and JS-rendered SPAs are not supported.
    """
    url = (url or "").strip()
    if not url.startswith(("http://", "https://")):
        return {"url": url, "error": "invalid url (must start with http)"}

    try:
        async with httpx.AsyncClient(
            timeout=20.0, follow_redirects=True, headers={"User-Agent": _UA}
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            ctype = resp.headers.get("content-type", "")
            if "html" not in ctype and "text" not in ctype:
                return {
                    "url": url,
                    "error": f"unsupported content type: {ctype or 'unknown'}",
                }
            html = resp.text
    except httpx.HTTPStatusError as e:
        return {"url": url, "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        logger.warning(f"[web_fetch] {url} failed: {type(e).__name__}: {e}")
        return {"url": url, "error": f"fetch failed: {type(e).__name__}"}

    title, text = _extract_main_text(html)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…(truncated)"
    return {"url": url, "title": title, "text": text}


def _extract_main_text(html: str) -> tuple[str, str]:
    """Strip boilerplate and return (title, main_text) from raw HTML."""
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string.strip() if soup.title and soup.title.string else "")

    # Drop non-content elements outright.
    for tag in soup(
        ["script", "style", "noscript", "nav", "header", "footer", "aside", "form"]
    ):
        tag.decompose()

    # Prefer <article> / <main> if present, else fall back to <body>.
    container = soup.find("article") or soup.find("main") or soup.body or soup
    parts: List[str] = []
    for el in container.find_all(
        ["h1", "h2", "h3", "h4", "p", "li", "blockquote", "pre"]
    ):
        txt = el.get_text(" ", strip=True)
        if txt:
            parts.append(txt)
    text = "\n".join(parts)
    # Fallback: if structured extraction found little, use the whole text.
    if len(text) < 200:
        text = container.get_text("\n", strip=True)
    return title, text
