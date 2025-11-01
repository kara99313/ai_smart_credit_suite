# utils/health.py
from __future__ import annotations
import socket
import requests
from typing import Tuple, Optional

def check_internet(timeout: float = 2.0) -> bool:
    try:
        socket.setdefaulttimeout(timeout)
        socket.create_connection(("1.1.1.1", 53), timeout=timeout)
        return True
    except Exception:
        return False

def ping_url(url: str, timeout: float = 2.5) -> Tuple[bool, str]:
    try:
        r = requests.get(url, timeout=timeout)
        return (r.status_code < 400), f"{r.status_code} {r.reason}"
    except Exception as e:
        return False, str(e)

def ping_ollama(base_url: str) -> Tuple[bool, str]:
    if not base_url:
        return False, "base_url manquant"
    url = base_url.rstrip("/") + "/api/tags"
    return ping_url(url)

def ping_openai(api_key: Optional[str]) -> Tuple[bool, str]:
    """Ping léger du endpoint models (si clé fournie)."""
    if not api_key:
        return False, "clé absente"
    try:
        r = requests.get(
            "https://api.openai.com/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=4.0,
        )
        if r.status_code == 200:
            return True, "200 OK"
        return False, f"{r.status_code} {r.text[:80]}"
    except Exception as e:
        return False, str(e)
