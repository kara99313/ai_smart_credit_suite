# utils/web_search.py
from __future__ import annotations
import re
import html as htmlmod
import time
from dataclasses import dataclass
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

@dataclass
class WebHit:
    title: str
    href: str
    body: str

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
        key = (_norm_href(h.get("href", "")) or "").lower()
        if key and key not in seen:
            seen.add(key)
            h["href"] = _norm_href(h.get("href", ""))
            out.append(h)
    return out

# ----------------- Moteur via la lib DDGS -----------------
def _search_web_ddg_lib(q: str, max_results: int, region: str) -> List[Dict[str, str]]:
    # Quelques DDG régionales utiles : "fr-fr", "en-us", "uk-en", "ng-en"...
    # safesearch: 'off'|'moderate'|'strict'
    tries = 2
    for _ in range(tries):
        try:
            with DDGS() as ddgs:
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
    return []

# ----------------- Fallback HTML (sans dépendance) -----------------
def _search_web_ddg_html(q: str, max_results: int) -> List[Dict[str, str]]:
    # Interface HTML simple et gratuite
    url = "https://html.duckduckgo.com/html/"
    try:
        r = requests.post(url, data={"q": q}, headers=UA, timeout=DEFAULT_TIMEOUT)
        r.raise_for_status()
        html_text = r.text
    except Exception:
        return []

    # Selectors HTML basés sur la version "html" (léger mais sujet à variations)
    titles = re.findall(r'<a[^>]+class="result__a"[^>]*>(.*?)</a>', html_text, flags=re.I | re.S)
    hrefs  = re.findall(r'<a[^>]+class="result__a"[^>]*href="([^"]+)"', html_text, flags=re.I | re.S)
    bodies = re.findall(r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>', html_text, flags=re.I | re.S)

    hits: List[WebHit] = []
    n = min(max_results, len(hrefs))
    for i in range(n):
        t = re.sub("<.*?>", "", titles[i] if i < len(titles) else "")
        b = re.sub("<.*?>", "", bodies[i] if i < len(bodies) else "")
        hit = WebHit(
            title=_clean_text(t) or _clean_text(hrefs[i]),
            href=_norm_href(hrefs[i]),
            body=_clean_text(b),
        )
        hits.append(hit)

    return [h.__dict__ for h in _dedup_hits([h.__dict__ for h in hits])][:max_results]

# ----------------- API publique -----------------
def search_web(query: str, max_results: int = 3, region: str = "fr-fr") -> List[Dict[str, str]]:
    """
    Recherche DuckDuckGo.
    Retour: liste de dicts {title, href, body}.
    Tente d'abord la lib officielle (si installée), sinon fallback HTML.
    """
    q = (query or "").strip()
    if not q:
        return []
    if _HAS_DDG:
        try:
            out = _search_web_ddg_lib(q, max_results=max_results, region=region)
            if out:
                return out
        except Exception:
            pass
    # Fallback sans dépendances
    return _search_web_ddg_html(q, max_results=max_results)

def fetch_and_clean(url: str, max_chars: int = 2800, timeout: float = DEFAULT_TIMEOUT) -> str:
    """
    Télécharge une page et renvoie un texte brut court (sans dépendances externes).
    """
    try:
        r = requests.get(url, headers=UA, timeout=timeout)
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
