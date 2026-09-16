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

import re
from typing import Any, Dict, List, Optional

import httpx
from bs4 import BeautifulSoup, NavigableString, Tag
from loguru import logger

_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Appended to every SUCCESSFUL web result (あさひ 08-31). Both think-only
# incidents that day were web rounds: the model digested the results in a
# long thinking block and ended the turn without ever speaking. The music
# MCP has carried the same result-adjacent instruction since 07-27 with zero
# silent turns on record — this is that mechanism, not a "use tools more"
# prompt nudge (those are proven useless). Honest wording: web results DO
# persist in context (truncated per round), so no "this disappears" claim.
#
# ⚠ WORDING CONSTRAINT (09-01, first classifier hit on record): the original
# clause 「頭の中で整理するだけで終わらせず…自分の言葉にして」 tripped
# Anthropic's safety classifier as CoT DISTILLATION — any instruction that
# references the model's internal thinking and asks it to externalise it
# reads as a thinking-extraction attempt. Phrase result-notes as a positive
# "tell the user in your reply" only; never contrast thinking vs speaking.
_RESULT_NOTE = (
    "\n\n（メモ：この結果はあなたの参照用の資料。調べて分かった内容は、"
    "このターンの返信本文の中で必ず自分の言葉でユーザーに伝えること。）"
)


def format_search_results(query: str, results: Any) -> Any:
    """Successful searches become readable text (uber-style) + the note.

    Error shapes pass through UNCHANGED — the agent's is_error detection
    branches on the list/dict shape (08-09 lesson), so errors must keep it.
    """
    if (
        not isinstance(results, list)
        or not results
        or any(isinstance(r, dict) and r.get("error") for r in results)
    ):
        return results
    lines = [f"「{query}」の検索結果:"]
    for i, r in enumerate(results, 1):
        if not isinstance(r, dict):
            continue
        lines.append(f"{i}. {(r.get('title') or '').strip() or '(無題)'}")
        if r.get("url"):
            lines.append(f"   {r['url']}")
        snippet = (r.get("snippet") or "").strip()
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines) + _RESULT_NOTE


def format_fetch_result(result: Any) -> Any:
    """Successful fetches become readable text + the note; errors unchanged."""
    if not isinstance(result, dict) or result.get("error"):
        return result
    header = f"取得: {result.get('url', '')}"
    title = (result.get("title") or "").strip()
    if title:
        header += f"\nタイトル: {title}"
    return f"{header}\n\n{result.get('text', '')}{_RESULT_NOTE}"


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
            # Raw BYTES, not resp.text: httpx decodes .text from the HTTP
            # header charset (falling back to UTF-8), but many older JA sites
            # declare Shift_JIS/EUC-JP only in <meta charset> — that combo
            # produced mojibake ("couldn't read the page"). BeautifulSoup's
            # own detector reads the meta declaration, so give it the bytes.
            raw = resp.content
    except httpx.HTTPStatusError as e:
        return {"url": url, "error": f"HTTP {e.response.status_code}"}
    except Exception as e:
        logger.warning(f"[web_fetch] {url} failed: {type(e).__name__}: {e}")
        return {"url": url, "error": f"fetch failed: {type(e).__name__}"}

    title, text = _extract_main_text(raw)
    if not text.strip():
        # Say so instead of returning silent emptiness — the model can tell
        # the user the page was unreadable instead of hallucinating content.
        return {
            "url": url,
            "title": title,
            "error": "no extractable text (page may be JS-rendered)",
        }
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…(truncated)"
    return {"url": url, "title": title, "text": text}


def _extract_main_text(html: str | bytes) -> tuple[str, str]:
    """Strip boilerplate and return (title, main_text) from raw HTML.

    Accepts bytes so BeautifulSoup can sniff the charset from the page's own
    <meta> declaration (see web_fetch).

    Structure-preserving since 09-16 (あさひ: the tenki.jp forecast came out
    as one flattened cell per line, tables unreadable, every cell doubled).
    The old extractor listed the text of every h*/p/li/td/dt element in
    document order, so a <td> holding two <p> emitted three lines (cell, p,
    p) and a table lost its rows. Now the DOM is walked ONCE, each text node
    emitted exactly once, and structure is kept in a light markdown dialect:
    headings prefixed with #, lists as "- " items (nested indented), table
    rows as cells joined by " | ", <dl> pairs joined as "term: value" when
    the values are short, <pre> verbatim. Noise removed on the way: nav /
    header / footer / aside / form and friends, hidden elements, lists whose
    items are all bare links (site navigation), readability-style
    boilerplate classes (only when the element holds a minor share of the
    page — a wrapper class like "with_header" must never take the page with
    it; if pruning still empties the page it is retried without), exact
    repeats of substantial lines (responsive mobile/desktop duplicates) and
    headings left with nothing under them."""
    title, text = _render_page(html, prune=True)
    if len(text) < 200:
        title, text = _render_page(html, prune=False)
    return title, text


_DROP_TAGS = {
    "script",
    "style",
    "noscript",
    "nav",
    "header",
    "footer",
    "aside",
    "form",
    "button",
    "select",
    "option",
    "svg",
    "iframe",
    "template",
    "canvas",
    "video",
    "audio",
    "map",
    "input",
    "textarea",
    "label",
}
_DROP_ROLES = {
    "navigation",
    "banner",
    "contentinfo",
    "search",
    "menu",
    "menubar",
    "toolbar",
    "dialog",
    "alertdialog",
    "tooltip",
    "complementary",
}
# Readability-style boilerplate class/id words; see _prune_unlikely for the
# size guard that keeps this from eating a page wrapper.
_UNLIKELY_RE = re.compile(
    r"breadcrumb|combx|comment|community|disqus|footer|header|menu|remark|rss|"
    r"shoutbox|sidebar|sponsor|ad-break|agegate|pagination|pager|popup|cookie|"
    r"consent|banner|share|social|related|recommend|promo|advert|adsense|"
    r"newsletter|subscribe|editsection|skyscraper|\bad\b|\bads\b|-ad-|_ad_",
    re.I,
)
_MAYBE_CONTENT_RE = re.compile(r"article|body|content|main|story|entry|text", re.I)
_HEADING_MARK = {
    "h1": "#",
    "h2": "##",
    "h3": "###",
    "h4": "####",
    "h5": "#####",
    "h6": "######",
}
_BLOCK_TAGS = {
    "p",
    "div",
    "section",
    "article",
    "main",
    "blockquote",
    "li",
    "dt",
    "dd",
    "thead",
    "tbody",
    "tfoot",
    "tr",
    "figure",
    "figcaption",
    "details",
    "summary",
    "hr",
    "address",
    "center",
    "body",
    "html",
} | set(_HEADING_MARK)
_MIN_TEXT = 200


def _class_id(el: Tag) -> str:
    return " ".join(
        [str(el.get("id") or "")] + [str(c) for c in (el.get("class") or [])]
    )


def _is_hidden(el: Tag) -> bool:
    if el.get("hidden") is not None or el.get("aria-hidden") == "true":
        return True
    style = (el.get("style") or "").replace(" ", "").lower()
    return "display:none" in style or "visibility:hidden" in style


def _is_nav_list(el: Tag) -> bool:
    """A list whose items are (almost) all bare links is site navigation."""
    items = el.find_all("li")
    if len(items) < 4:
        return False
    linky = 0
    for li in items:
        txt = li.get_text(" ", strip=True)
        a = li.find("a")
        if a is not None and txt and a.get_text(" ", strip=True) == txt:
            linky += 1
    return linky >= 0.8 * len(items)


def _inline_text(el: Tag) -> str:
    return " ".join(el.get_text(" ", strip=True).split())


class _TextRenderer:
    """One DOM walk; every text node lands in exactly one output line."""

    def __init__(self) -> None:
        self.lines: List[str] = []
        self._buf: List[str] = []  # inline pieces of the block being built

    def render(self, node) -> List[str]:
        self._walk(node)
        self._flush()
        return self.lines

    def _flush(self) -> None:
        txt = " ".join(" ".join(self._buf).split())
        self._buf = []
        if txt:
            self.lines.append(txt)

    def _emit(self, line: str) -> None:
        self._flush()
        if line:
            self.lines.append(line)

    def _sub(self, node: Tag, depth: int = 0, skip=()) -> List[str]:
        r = _TextRenderer()
        for c in node.children:
            if isinstance(c, Tag) and c.name in skip:
                continue
            r._walk(c, depth)
        r._flush()
        return r.lines

    def _walk(self, node, depth: int = 0) -> None:
        if isinstance(node, NavigableString):
            # Comment / Doctype / CData / ProcessingInstruction are subclasses
            # and must not leak (tenki.jp's template comments did).
            if type(node) is NavigableString and node.strip():
                self._buf.append(str(node))
            return
        if not isinstance(node, Tag):
            return
        name = node.name.lower()
        if name in _DROP_TAGS or _is_hidden(node):
            return
        if (node.get("role") or "").lower() in _DROP_ROLES:
            return
        if name in _HEADING_MARK:
            self._emit(f"{_HEADING_MARK[name]} {_inline_text(node)}")
        elif name == "table":
            self._flush()
            self._table(node)
        elif name in ("ul", "ol"):
            self._flush()
            if not _is_nav_list(node):
                self._list(node, depth)
        elif name == "dl":
            self._flush()
            self._dl(node)
        elif name == "pre":
            self._flush()
            txt = node.get_text().strip("\n")
            if txt.strip():
                self.lines.append(txt)
        elif name == "br":
            self._flush()
        elif name == "hr":
            self._emit("---")
        elif name in _BLOCK_TAGS:
            self._flush()
            for c in node.children:
                self._walk(c, depth)
            self._flush()
        else:  # inline element: keep accumulating into the current block
            for c in node.children:
                self._walk(c, depth)

    def _list(self, node: Tag, depth: int) -> None:
        indent = "  " * depth
        for li in node.find_all("li", recursive=False):
            lines = self._sub(li, depth + 1, skip=("ul", "ol"))
            if lines:
                self.lines.append(indent + "- " + lines[0])
                for extra in lines[1:]:
                    # a <pre> inside the item keeps its own indentation
                    self.lines.append(extra if "\n" in extra else indent + "  " + extra)
            for c in li.find_all(["ul", "ol"], recursive=False):
                if not _is_nav_list(c):
                    self._list(c, depth + 1)

    def _dl(self, node: Tag) -> None:
        """dt/dd pairs: short single-line values join their term on one line
        ("最高: 21 ℃ [-7]"); long bodies (API docs) stay as their own lines."""
        terms: List[str] = []
        values: List[List[str]] = []

        def emit() -> None:
            if not terms and not values:
                return
            flat = [v for vs in values for v in vs]
            head = " / ".join(terms)
            if flat and all(len(vs) == 1 and len(vs[0]) < 200 for vs in values):
                joined = " ".join(flat)
                self.lines.append(f"{head}: {joined}" if head else joined)
            else:
                if head:
                    self.lines.append(head)
                self.lines.extend(flat)
            terms.clear()
            values.clear()

        for c in node.children:
            if not isinstance(c, Tag):
                continue
            if c.name == "dt":
                if values:
                    emit()
                terms.extend(self._sub(c))
            elif c.name == "dd":
                values.append(self._sub(c))
            elif c.name == "div":  # HTML5 allows <div> grouping inside <dl>
                emit()
                self._dl(c)
        emit()

    def _table(self, node: Tag) -> None:
        cap = node.find("caption")
        if cap is not None:
            txt = _inline_text(cap)
            if txt:
                self.lines.append(f"[表] {txt}")
        rows: List[List[str]] = []
        for tr in node.find_all("tr"):
            if tr.find_parent("table") is not node:
                continue  # nested table: rendered when its own <table> is walked
            cells: List[str] = []
            for c in tr.find_all(["th", "td"], recursive=False):
                if c.find("table"):
                    cells.append(" ".join(self._sub(c)))
                else:
                    cells.append(_inline_text(c))
            if any(cells):
                rows.append(cells)
        if not rows:
            return
        if len(rows) == 1 and len(rows[0]) == 1:
            self.lines.append(rows[0][0])  # layout table with a single cell
            return
        for r in rows:
            self.lines.append(" | ".join(r))


def _heading_level(line: str) -> Optional[int]:
    m = re.match(r"^(#{1,6}) ", line)
    return len(m.group(1)) if m else None


def _cleanup_lines(lines: List[str]) -> List[str]:
    """Drop blanks, consecutive repeats, exact repeats of substantial lines
    anywhere (responsive duplicates), then headings with no content under
    them (their section was pruned as navigation)."""
    out: List[str] = []
    seen = set()
    for ln in lines:
        key = ln.rstrip()
        core = key.strip()
        if not core:
            continue
        if out and out[-1] == key:
            continue
        if (
            len(core) >= 12
            and "\n" not in core
            and not core.startswith(("|", "-", "#"))
        ):
            if core in seen:
                continue
            seen.add(core)
        out.append(key)
    changed = True
    while changed:
        changed = False
        kept: List[str] = []
        for i, ln in enumerate(out):
            lvl = _heading_level(ln)
            if lvl is not None:
                nxt = _heading_level(out[i + 1]) if i + 1 < len(out) else 0
                if nxt is not None and nxt <= lvl:
                    changed = True
                    continue
            kept.append(ln)
        out = kept
    return out


def _prune_unlikely(container: Tag) -> None:
    total = len(container.get_text(" ", strip=True)) or 1
    for el in list(container.find_all(True)):
        if el is container or el.parent is None or el.name in ("body", "html"):
            continue
        ci = _class_id(el)
        if not ci or not _UNLIKELY_RE.search(ci) or _MAYBE_CONTENT_RE.search(ci):
            continue
        if len(el.get_text(" ", strip=True)) > 0.3 * total:
            continue  # holds most of the page: cannot be boilerplate
        el.decompose()


def _render_page(html: str | bytes, prune: bool) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    for tag in soup(list(_DROP_TAGS)):
        tag.decompose()
    body = soup.body or soup
    # Prefer <article> / <main>; a sparse one (a news index whose first
    # <article> is a teaser) escalates to <body> — still structured — and
    # only then to a raw text dump.
    containers: List[Tag] = []
    for c in (soup.find("article"), soup.find("main"), body):
        if c is not None and all(c is not x for x in containers):
            containers.append(c)
    text = ""
    for container in containers:
        if prune:
            _prune_unlikely(container)
        text = "\n".join(_cleanup_lines(_TextRenderer().render(container)))
        if len(text) >= _MIN_TEXT:
            return title, text
    text = max([text, body.get_text("\n", strip=True)], key=len)
    return title, text
