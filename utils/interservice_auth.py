"""Interservice credentials, scoped outbound routing and fail-closed ASGI guard."""
import hmac
import os
from contextvars import ContextVar
from starlette.responses import JSONResponse

HEADER = "X-Interservice-Secret"
backend_loopback = ContextVar("backend_loopback", default=False)


def secret(value=None):
    if value is None:
        value = os.environ.get("INTERSERVICE_SECRET", "")
    if not value or value != value.strip() or not value.isascii() or any(ord(c) < 33 or ord(c) > 126 for c in value):
        raise ValueError("Authentification interservice indisponible.")
    return value


def headers():
    if backend_loopback.get():
        return {HEADER: secret()}
    from utils.llm_runtime import get_setting
    return {HEADER: secret(get_setting("INTERSERVICE_SECRET", ""))}


class InterserviceAuthMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        protected = path.rstrip("/") in {"/api/predict", "/api/predict_ping", "/api/predict_echo"} or path == "/tools" or path.startswith("/tools/")
        if path.rstrip("/") in {"/docs", "/docs/oauth2-redirect", "/redoc", "/openapi.json"}:
            local = scope.get("client") and scope["client"][0] in {"127.0.0.1", "::1"}
            if not (os.getenv("ENABLE_LOCAL_API_DOCS") == "1" and local):
                return await JSONResponse({"detail": "Not found"}, status_code=404)(scope, receive, send)
        if protected:
            try:
                expected = secret()
            except ValueError:
                return await JSONResponse({"detail": "Service indisponible."}, status_code=503)(scope, receive, send)
            supplied = [v for k, v in scope.get("headers", []) if k.lower() == b"x-interservice-secret"]
            if len(supplied) != 1 or not hmac.compare_digest(supplied[0], expected.encode("ascii")):
                return await JSONResponse({"detail": "Authentification requise."}, status_code=401)(scope, receive, send)
        return await self.app(scope, receive, send)
