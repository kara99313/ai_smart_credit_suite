"""Execution settings shared by the API, its client and the local launcher."""
from __future__ import annotations
import os
from pathlib import Path
from urllib.parse import urlsplit
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"
load_dotenv(ENV_PATH, override=False, encoding="utf-8")

def _config() -> dict:
    path = ROOT / "config.yaml"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as source:
        settings = yaml.safe_load(source) or {}
    if not isinstance(settings, dict):
        raise ValueError("config.yaml doit contenir un objet de configuration.")
    return settings

def _port(value) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError(f"Port invalide : {port}")
    return port

def api_bind() -> tuple[str, int]:
    backend = _config().get("backend", {})
    return (os.getenv("API_HOST") or backend.get("host", "127.0.0.1"),
            _port(os.getenv("API_PORT") or backend.get("port", 18000)))

def loopback_url() -> str:
    return f"http://127.0.0.1:{_port(os.getenv("PORT") or os.getenv("API_PORT") or 8000)}"


def web_port() -> int:
    return _port(os.getenv("WEB_PORT") or _config().get("ui", {}).get("web_port", 8501))

def backend_url(base: str | None = None) -> str:
    url = base or os.getenv("BACKEND_URL") or os.getenv("API_BASE")
    if not url:
        url = _config().get("backend", {}).get("base_url")
    if not url:
        host, port = api_bind()
        host = "127.0.0.1" if host == "0.0.0.0" else host
        host = "::1" if host == "::" else host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        url = f"http://{host}:{port}"
    url = str(url).strip().rstrip("/")
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("L'URL API doit etre une URL http(s) absolue.")
    if parsed.query or parsed.fragment:
        raise ValueError("L'URL API ne doit pas contenir de query ni de fragment.")
    return url


if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-dotenv", action="store_true")
    args = parser.parse_args()
    from dotenv import dotenv_values
    host, port = api_bind()
    print(json.dumps({
        "api_host": host, "api_port": port, "web_port": web_port(),
        "backend_url": backend_url(),
        "dotenv_environment": {name: ("[REDACTED]" if any(part in name.upper() for part in ("SECRET", "TOKEN", "PASSWORD", "KEY")) else os.environ[name]) for name in dotenv_values(ENV_PATH, encoding="utf-8") if name in os.environ} if args.include_dotenv else {},
    }))
