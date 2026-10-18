"""Shell-free service entry points shared by Docker and Render."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
from urllib.request import urlopen
from utils.runtime_config import ROOT, api_bind, backend_url, web_port, _port


def command(service):
    if service == "api":
        host, port = api_bind()
        port = _port(os.getenv("PORT") or port)
        model = Path(os.getenv("MODEL_PATH") or "model/logistic_pipeline_best.pkl")
        if not (model if model.is_absolute() else ROOT / model).is_file():
            raise RuntimeError("MODEL_PATH absent ou inaccessible ; demarrage API annule.")
        return [sys.executable, "-m", "uvicorn", "server:app", "--host", host,
                "--port", str(port)]
    if os.getenv("INTERNAL_API_HOSTPORT"):
        os.environ["BACKEND_URL"] = "http://" + os.environ["INTERNAL_API_HOSTPORT"]
    backend_url()
    port = _port(os.getenv("PORT") or web_port())
    return [sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
            "--server.address", os.getenv("WEB_HOST", "127.0.0.1"),
            "--server.port", str(port), "--server.headless=true",
            "--browser.gatherUsageStats=false"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("service", choices=("api", "web", "check-api", "check-web"))
    service = parser.parse_args().service
    os.chdir(ROOT)
    if service.startswith("check-"):
        api = service == "check-api"
        port = _port(os.getenv("PORT") or (api_bind()[1] if api else web_port()))
        path = "/health" if api else "/_stcore/health"
        with urlopen(f"http://127.0.0.1:{port}{path}", timeout=3) as response:
            if response.status != 200:
                raise RuntimeError("Health check failed")
        return
    args = command(service)
    if os.name == "nt":
        # Windows execv does not quote interpreter paths containing spaces.
        import subprocess
        raise SystemExit(subprocess.call(args, creationflags=subprocess.CREATE_NO_WINDOW))
    os.execv(sys.executable, args)


if __name__ == "__main__":
    main()
