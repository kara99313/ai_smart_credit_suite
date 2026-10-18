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
except Exception:
    _HAS_DDG = False

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Streamlit/WebSearch"}
DEFAULT_TIMEOUT = 10.0

class WebSearchError(RuntimeError):
    """Safe connector error; never include response bodies or credentials."""


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
    tries = 2
    for _ in range(tries):
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
        except Exception:
            time.sleep(0.2)
    raise WebSearchError("Recherche DuckDuckGo indisponible (reseau, delai ou reponse invalide).") from None

# ----------------- Fallback HTML (sans dépendance) -----------------
def _search_web_ddg_html(q: str, max_results: int, timeout: float = DEFAULT_TIMEOUT) -> List[Dict[str, str]]:
    # Interface HTML simple et gratuite
    url = "https://html.duckduckgo.com/html/"
    try:
        r = requests.post(url, data={"q": q}, headers=UA, timeout=timeout)
        r.raise_for_status()
        html_text = r.text
    except Exception:
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
    return _dedup_hits(parser.hits)[:max_results]

# ----------------- API publique -----------------
def search_web(query: str, max_results: int = 3, region: str = "fr-fr") -> List[Dict[str, str]]:
    """
    Recherche DuckDuckGo.
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
    out = []
    if _HAS_DDG:
        try:
            out = _search_web_ddg_lib(q, max_results=limit, region=region, timeout=config['timeout'])
        except WebSearchError:
            pass
    if not out:
        out = _search_web_ddg_html(q, max_results=limit, timeout=config['timeout'])
    return [hit for hit in _dedup_hits(out) if _allowed_url(hit['href'], config['domains'])][:limit]


def fetch_and_clean(url: str, max_chars: int = 2800, timeout: float = DEFAULT_TIMEOUT) -> str:
    """
    Télécharge une page et renvoie un texte brut court (sans dépendances externes).
    """
    config = web_configuration()
    if not config['enabled'] or not _allowed_url(url, config['domains']):
        return ""
    timeout = min(float(timeout), config['timeout'])
    try:
        # Check every redirect against the same domain policy before fetching it.
        for _ in range(5):
            r = requests.get(url, headers=UA, timeout=timeout, allow_redirects=False)
            if not 300 <= r.status_code < 400:
                break
            from urllib.parse import urljoin
            url = urljoin(url, r.headers.get("Location", ""))
            if not _allowed_url(url, config['domains']):
                return ""
        else:
            return ""
        r.raise_for_status()
        txt = r.text
    except Exception:
        return ""
    # Retire scripts/styles et balises
    txt = re.sub(r"(?is)<script.*?>.*?</script>", " ", txt)
    txt = re.sub(r"(?is)<style.*?>.*?</style>", " ", txt)
    txt = re.sub(r"(?is)<[^>]+>", " ", txt)
    txt = _clean_text(txt)
    if len(txt) > max_chars:
        txt = txt[:max_chars] + "…"
    return txt

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
    if not hits:
        return ""
    parts: List[str] = []
    budget = max_chars_total
    for i, h in enumerate(hits, 1):
        title = (h.get("title") or "").strip()
        href  = (h.get("href")  or "").strip()
        body  = (h.get("body")  or "").strip()

        if fetch_full and href:
            full = fetch_and_clean(href, max_chars=min(900, budget))
            if full:
                body = full

        block = f"[W{i}] {title}\n{href}\n{body}".strip()
        if len(block) > budget:
            block = block[:budget] + "…"
        parts.append(block)
        budget -= len(block)
        if budget <= 0:
            break

        time.sleep(0.2)  # politesse pour éviter de spammer DDG
    return "\n\n".join(parts)
