# utils/web_search.py
from __future__ import annotations
import re
import html as htmlmod
import time
import math
from urllib.parse import urlsplit
from utils.llm_runtime import get_setting
from html.parser import HTMLParser
from typing import List, Dict, Iterable

import requests

# ---------- Détection lib officielle DuckDuckGo ----------
_HAS_DDG = True
try:
    from duckduckgo_search import DDGS  # pip install duckduckgo-search
    from duckduckgo_search.exceptions import RatelimitException, TimeoutException
except Exception:
    _HAS_DDG = False

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Streamlit/WebSearch"}
DEFAULT_TIMEOUT = 10.0

class WebSearchError(RuntimeError):
    """Safe connector error; never include response bodies or credentials."""


class WebSearchRateLimitError(WebSearchError):
    """The provider refused automated requests."""


class WebSearchResults(list):
    """List compatible with agents, with safe provider metadata."""
    def __init__(self, hits, provider):
        super().__init__(hits)
        self.provider = provider


def _search_web_tavily(q, max_results, timeout, domains, api_key):
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": "Bearer " + api_key},
            json=dict(query=q, max_results=max_results, search_depth="basic",
                      include_domains=domains, include_answer=False, include_raw_content=False),
            timeout=timeout, allow_redirects=False,
        )
        if response.status_code in (401, 403):
            raise WebSearchError("Tavily indisponible : authentification ou droits refuses.")
        if response.status_code in (429, 432, 433):
            raise WebSearchError("Tavily indisponible : quota ou limite de requetes atteint.")
        if not 200 <= response.status_code < 300:
            raise WebSearchError("Tavily indisponible : erreur HTTP.")
        payload = response.json()
        if not isinstance(payload, dict) or not isinstance(payload.get('results'), list):
            raise WebSearchError("Tavily indisponible : reponse invalide.")
        hits = []
        for item in payload['results']:
            if not isinstance(item, dict) or not all(isinstance(item.get(k), str) for k in ('title', 'url', 'content')):
                raise WebSearchError("Tavily indisponible : resultat invalide.")
            hits.append(dict(title=item['title'], href=item['url'], body=item['content']))
        return _dedup_hits(hits)
    except WebSearchError:
        raise
    except requests.Timeout:
        raise WebSearchError("Tavily indisponible : delai depasse.") from None
    except Exception:
        raise WebSearchError("Tavily indisponible : reseau ou reponse invalide.") from None


def normalize_domains(domains):
    if not isinstance(domains, list):
        raise ValueError("Liste de domaines invalide.")
    normalized = []
    for value in domains:
        if not isinstance(value, str):
            raise ValueError("Domaine invalide.")
        value = value.strip().lower().rstrip(".")
        try:
            value = value.encode("idna").decode("ascii")
        except UnicodeError:
            raise ValueError("Domaine invalide.") from None
        if not value or len(value) > 253 or not all(re.fullmatch(r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?", label) for label in value.split(".")):
            raise ValueError("Saisir des noms de domaine sans URL, port ni chemin.")
        if value not in normalized:
            normalized.append(value)
    return normalized


def web_configuration():
    from utils.settings import get_settings
    cfg = get_settings()
    enabled = get_setting("WEB_SEARCH_ENABLED", str(cfg.enable_web_search)).strip().lower()
    if enabled not in {"true", "false", "1", "0"}:
        raise ValueError("WEB_SEARCH_ENABLED doit valoir true ou false.")
    env_domains = get_setting("WEB_ALLOWED_DOMAINS", "")
    domains = normalize_domains([d.strip() for d in env_domains.split(",") if d.strip()] if env_domains else cfg.allowed_domains)
    try:
        timeout = float(get_setting("WEB_TIMEOUT", "10"))
        limit = int(cfg.web_search_max)
        if not math.isfinite(timeout) or not 0 < timeout <= 60 or not 1 <= limit <= 10:
            raise ValueError()
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Configuration Web invalide : timeout (0,60], maximum de resultats [1,10].") from None
    return dict(enabled=enabled in {"true", "1"}, domains=domains, timeout=timeout, max_results=limit)


def _allowed_url(url, domains=()):
    try:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower().rstrip(".").encode("idna").decode("ascii")
        return bool(parsed.scheme in {"http", "https"} and host and not parsed.username and not parsed.password
                    and (not domains or any(host == domain or host.endswith("." + domain) for domain in domains)))
    except (ValueError, TypeError, UnicodeError):
        return False


# ----------------- Utilitaires -----------------
def _clean_text(s: str) -> str:
    if not s:
        return ""
    s = htmlmod.unescape(s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _norm_href(h: str) -> str:
    h = (h or "").strip()
    if not h:
        return h
    # Retirer éventuels redirecteurs "http(s)://r.duckduckgo.com/l/?uddg=..."
    if "uddg=" in h:
        try:
            from urllib.parse import parse_qs, urlparse, unquote
            qs = parse_qs(urlparse(h).query)
            if "uddg" in qs and qs["uddg"]:
                return unquote(qs["uddg"][0])
        except Exception:
            pass
    return h

def _dedup_hits(hits: Iterable[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    out: List[Dict[str, str]] = []
    for h in hits:
        if not isinstance(h, dict) or not all(isinstance(h.get(key, ""), str) for key in ("href", "title", "body")):
            continue
        href = _norm_href(h.get("href", ""))
        key = href.lower()
        if _allowed_url(href) and key not in seen:
            seen.add(key)
            out.append(dict(title=_clean_text(h.get("title", "")), href=href, body=_clean_text(h.get("body", ""))))
    return out

# ----------------- Moteur via la lib DDGS -----------------
def _search_web_ddg_lib(q: str, max_results: int, region: str, timeout: float = DEFAULT_TIMEOUT) -> List[Dict[str, str]]:
    # Quelques DDG régionales utiles : "fr-fr", "en-us", "uk-en", "ng-en"...
    # safesearch: 'off'|'moderate'|'strict'
    try:
        with DDGS(timeout=timeout) as ddgs:
            results = ddgs.text(
                q,
                region=region,
                safesearch="moderate",
                timelimit="y",            # 1 an ; réduit un peu le bruit
                max_results=max_results,
            )
            out: List[Dict[str, str]] = []
            for r in results or []:
                out.append({
                    "title": _clean_text((r.get("title") or "")),
                    "href": _norm_href(r.get("href") or ""),
                    "body": _clean_text((r.get("body") or "")),
                })
            return _dedup_hits(out)[:max_results]
    except RatelimitException:
        raise WebSearchRateLimitError("Recherche DuckDuckGo limitee (rate limit). Reessayer plus tard.") from None
    except TimeoutException:
        raise WebSearchError("Recherche DuckDuckGo indisponible (delai depasse).") from None
    except Exception as exc:
        if "10013" in str(exc):
            raise WebSearchError("Connexion DuckDuckGo bloquee par les permissions reseau (WinError 10013).") from None
    raise WebSearchError("Recherche DuckDuckGo indisponible (reseau, delai ou reponse invalide).") from None

# ----------------- Fallback HTML (sans dépendance) -----------------
def _search_web_ddg_html(q: str, max_results: int, timeout: float = DEFAULT_TIMEOUT) -> List[Dict[str, str]]:
    # Interface HTML simple et gratuite
    url = "https://html.duckduckgo.com/html/"
    try:
        r = requests.post(url, data={"q": q}, headers=UA, timeout=timeout)
        if r.status_code in (202, 403, 429):
            raise WebSearchRateLimitError("Recherche DuckDuckGo limitee ou bloquee (rate limit / anti-bot). Reessayer plus tard.")
        r.raise_for_status()
        html_text = r.text
        if any(marker in html_text.lower() for marker in ('anomaly.js', 'anomaly-modal', 'challenge-form')):
            raise WebSearchRateLimitError("Recherche DuckDuckGo bloquee par une verification anti-bot. Reessayer plus tard.")
    except WebSearchError:
        raise
    except requests.Timeout:
        raise WebSearchError("Recherche Web indisponible (delai depasse).") from None
    except Exception as exc:
        if "10013" in str(exc):
            raise WebSearchError("Connexion Web bloquee par les permissions reseau (WinError 10013).") from None
        raise WebSearchError("Recherche Web indisponible (HTTP, reseau ou delai depasse).") from None

    class ResultsParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.hits = []
            self.capture = None
            self.depth = 0

        def handle_starttag(self, tag, attrs):
            if tag in {"br", "img", "input", "hr", "meta", "link"}:
                return
            attrs = dict(attrs)
            classes = attrs.get('class', '').split()
            if self.capture:
                self.depth += 1
            elif tag == 'a' and 'result__a' in classes:
                self.hits.append(dict(title='', href=_norm_href(attrs.get('href', '')), body=''))
                self.capture, self.depth = 'title', 1
            elif 'result__snippet' in classes and self.hits:
                self.capture, self.depth = 'body', 1

        def handle_endtag(self, tag):
            if self.capture:
                self.depth -= 1
                if self.depth <= 0:
                    self.capture = None

        def handle_data(self, data):
            if self.capture:
                self.hits[-1][self.capture] += data

    parser = ResultsParser()
    try:
        parser.feed(html_text)
    except Exception:
        raise WebSearchError("Reponse Web invalide.") from None
    if not parser.hits and not re.search(r'no-results|no results', html_text, re.IGNORECASE):
        raise WebSearchError("Reponse DuckDuckGo non reconnue ; recherche non confirmee.")
    return _dedup_hits(parser.hits)[:max_results]

# ----------------- API publique -----------------
def search_web(query: str, max_results: int = 3, region: str = "fr-fr") -> List[Dict[str, str]]:
    """
    Recherche Tavily puis DuckDuckGo en cas d indisponibilite.
    Retour: liste de dicts {title, href, body}.
    Tente d'abord la lib officielle (si installée), sinon fallback HTML.
    """
    config = web_configuration()
    if not config['enabled']:
        return []
    q = (query or "").strip()
    if not q:
        return []
    if isinstance(max_results, bool) or not isinstance(max_results, int) or not 1 <= max_results <= 10:
        raise ValueError("Nombre de resultats invalide (1-10).")
    limit = min(max_results, config['max_results'])
    tavily_error = None
    api_key = get_setting('TAVILY_API_KEY', '').strip()
    if api_key:
        try:
            hits = _search_web_tavily(q, limit, config['timeout'], config['domains'], api_key)
            return WebSearchResults([hit for hit in hits if _allowed_url(hit['href'], config['domains'])][:limit], 'Tavily')
        except WebSearchError as exc:
            tavily_error = exc
    out = []
    library_error = None
    if _HAS_DDG:
        try:
            out = _search_web_ddg_lib(q, max_results=limit, region=region, timeout=config['timeout'])
        except WebSearchError as exc:
            library_error = exc
    if not out:
        try:
            out = _search_web_ddg_html(q, max_results=limit, timeout=config['timeout'])
        except WebSearchError as exc:
            prefix = f"{tavily_error} Repli DuckDuckGo : " if tavily_error else ""
            if library_error:
                raise WebSearchError(f"{prefix}{library_error} Repli HTML : {exc}") from None
            raise WebSearchError(f"{prefix}{exc}") from None
    return WebSearchResults([hit for hit in _dedup_hits(out) if _allowed_url(hit['href'], config['domains'])][:limit], 'DuckDuckGo')


def fetch_and_clean(url: str, max_chars: int = 2800, timeout: float = DEFAULT_TIMEOUT) -> str:
    """Compatibility stub: source-page downloads are disabled, including direct calls."""
    return ""


def _text_response(response) -> bool:
    """Inspect final response bytes before any text decoding or HTML parsing."""
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if content_type and content_type not in {"text/html", "application/xhtml+xml", "text/plain"}:
        return False
    raw = response.content
    return not (b"%PDF-" in raw[:1024] or any(byte < 32 and byte not in (9, 10, 13) for byte in raw))


def _safe_excerpt(text: str) -> bool:
    """Validate both evidence sources before truncation, including provider-only mode."""
    if not text or re.search(
        r"%PDF|\bendobj\b|ViewerPreferences|\b(?:xref|endstream)\b|"
        r"captcha|access denied|verify you are human|just a moment|"
        r"anomaly\.js|anomaly-modal|challenge-form|"
        r"<\s*[!/?a-z][^>]*>|&(?:lt|gt|#0*60|#x0*3c);|"
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd]",
        text, re.IGNORECASE,
    ):
        return False
    navigation = re.findall(
        r"\b(?:skip to (?:main )?content|menu|home|accueil|navigation|contact|"
        r"connexion|login|subscribe|newsletter|privacy|cookies|all rights reserved)\b",
        text, re.IGNORECASE,
    )
    return not navigation or sum(len(word.split()) for word in navigation) / max(1, len(text.split())) < 0.35


class _PageTextParser(HTMLParser):
    """Skip structural boilerplate; avoid broad substring filters on content."""
    VOID = {"br", "hr", "img", "input", "meta", "link", "source", "wbr", "area", "base", "embed", "param", "track", "col"}
    SKIP = {"script", "style", "nav", "footer", "header", "aside", "form", "noscript", "svg", "template"}

    def __init__(self):
        super().__init__()
        self.stack, self.parts = [], []
        self.main_parts, self.article_parts, self.role_parts = [], [], []

    def preferred_parts(self):
        for parts in (self.main_parts, self.article_parts, self.role_parts, self.parts):
            if any(part.strip() for part in parts):
                return parts
        return []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        tokens = re.split(r"[\s_-]+", (attrs.get("class", "") + " " + attrs.get("id", "")).lower())
        skip = (tag in self.SKIP or attrs.get("role") in {"navigation", "contentinfo", "banner"}
                or bool(set(tokens) & {"newsletter", "cookie", "cookies", "sidebar", "advertisement", "social", "subscribe", "menu", "navbar", "navigation", "footer", "skip"})
                or (tag == "a" and attrs.get("href", "").lower() in {"#main", "#content", "#main-content", "#maincontent"})
                or "hidden" in attrs or attrs.get("aria-hidden") == "true")
        if tag not in self.VOID:
            self.stack.append((tag, skip or bool(self.stack and self.stack[-1][1]), attrs.get("role") == "main"))

    def handle_startendtag(self, tag, attrs):
        pass

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break

    def handle_data(self, data):
        if (not self.stack or not self.stack[-1][1]) and not re.search(r"skip to (?:main )?content", data, re.IGNORECASE):
            self.parts.append(data)
            tags = {item[0] for item in self.stack}
            if "main" in tags:
                self.main_parts.append(data)
            if "article" in tags:
                self.article_parts.append(data)
            if any(item[2] for item in self.stack):
                self.role_parts.append(data)


def build_web_context(
    hits: List[Dict[str, str]],
    max_chars_total: int = 2200,
    fetch_full: bool = False
) -> str:
    """
    Construit un bloc CONTEXTE à donner au LLM.
    Format par item :
      [W1] Titre
      URL
      Extrait
    """
    # fetch_full is retained for callers; only provider snippets are used.
    if not hits:
        return ""
    headers = [f"[W{i}] {(h.get('title') or '').strip()}\n{(h.get('href') or '').strip()}" for i, h in enumerate(hits, 1)]
    # References remain intact even when their metadata exceeds the soft budget.
    budget = max(0, max_chars_total - sum(map(len, headers)) - 3 * len(headers))
    parts = []
    for index, (h, header) in enumerate(zip(hits, headers)):
        snippet = (h.get("body") or "").strip()
        body = _clean_text(snippet) if _safe_excerpt(snippet) and _safe_excerpt(_clean_text(snippet)) else ""
        if not body:
            body = "Aucun extrait exploitable disponible."
        allowance = budget // (len(headers) - index)
        excerpt = body[:allowance]
        budget -= len(excerpt)
        parts.append(header + ("\n" + excerpt if excerpt else ""))
    return "\n\n".join(parts)
