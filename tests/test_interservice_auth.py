import os
from unittest.mock import patch
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
import server
from test_prediction import PAYLOAD
from utils import api_client, runtime_config
from utils.interservice_auth import HEADER, backend_loopback

SECRET = "test-interservice-credential"

@pytest.mark.parametrize("path", ["/api/predict", "/api/predict_ping", "/api/predict_echo", "/tools/explain", "/tools/now", "/tools/rag_search", "/tools/web_search", "/tools/generate_report_html", "/tools/predict_credit", "/tools/future"])
@pytest.mark.parametrize("credential", [None, "wrong"])
def test_refuse_before_business(path, credential, caplog):
    client = TestClient(server.app)
    with patch.object(server, "_compute_pd") as score, patch.object(server, "_import_agent_tools") as tools:
        response = client.post(path, json=PAYLOAD, headers={} if credential is None else {HEADER: credential})
    assert response.status_code == 401
    score.assert_not_called()
    tools.assert_not_called()
    assert SECRET not in response.text + caplog.text

@pytest.mark.parametrize("value", ["", " ", "bad\nvalue", "\u00e9"])
def test_invalid_backend_secret(value, monkeypatch):
    monkeypatch.setenv("INTERSERVICE_SECRET", value)
    client = TestClient(server.app)
    assert client.post("/api/predict", json=PAYLOAD).status_code == 503
    assert client.get("/health").status_code == 200


def test_absent_backend_secret(monkeypatch):
    monkeypatch.delenv("INTERSERVICE_SECRET", raising=False)
    assert TestClient(server.app).get("/api/predict_ping").status_code == 503
    assert TestClient(server.app).get("/health").status_code == 200


def test_authenticated_business_unchanged():
    client = TestClient(server.app, headers={HEADER: SECRET})
    result = client.post("/api/predict", json=PAYLOAD)
    assert result.status_code == 200
    assert result.json() == server.api_predict(server.PredictRequest(**PAYLOAD)).model_dump()
    explanation = client.post("/tools/explain", json=PAYLOAD)
    assert explanation.status_code == 200
    assert explanation.json()["predicted_pd"] == result.json()["pd"]
    assert client.get("/api/predict_ping").status_code == 200
    assert client.post("/api/predict_echo", json=PAYLOAD).status_code == 200
    assert SECRET not in result.text + explanation.text


@pytest.mark.parametrize("env,port", [({},8000), ({"API_PORT":"8000"},8000), ({"API_PORT":"8000","PORT":"9100"},9100)])
def test_loopback_port(env, port):
    with patch.dict(os.environ, env, clear=True):
        assert runtime_config.loopback_url() == f"http://127.0.0.1:{port}"


def test_tool_loopback_and_origin(monkeypatch):
    monkeypatch.setenv("PORT", "8000")
    monkeypatch.setenv("BACKEND_URL", "http://private-backend:9999")
    client = TestClient(server.app, headers={HEADER: SECRET})
    seen = []
    def transport(url, **kwargs):
        seen.append((url, kwargs))
        assert kwargs["allow_redirects"] is False
        return client.post("/api/predict", json=kwargs["json"], headers=kwargs["headers"])
    def tool(text):
        import json
        return str(api_client.predict_credit_api(json.loads(text), synthetic=True)["pd"])
    with patch.object(server, "_import_agent_tools", return_value=(tool,)), patch("utils.api_client.requests.post", side_effect=transport):
        response = client.post("/tools/predict_credit", json={"data":PAYLOAD})
    assert response.status_code == 200
    assert seen[0][0] == "http://127.0.0.1:8000/api/predict"
    assert seen[0][1]["headers"] == {HEADER: SECRET, "X-Prediction-Origin":"synthetic"}
    assert not backend_loopback.get()


def test_outbound_redirect_and_external_destination(monkeypatch):
    monkeypatch.setenv("BACKEND_URL", "http://backend:8000")
    with patch("utils.api_client.requests.post", return_value=SimpleNamespace(status_code=302)) as post:
        with pytest.raises(api_client.ApiError):
            api_client.predict_credit_api(PAYLOAD)
        assert post.call_args.kwargs["allow_redirects"] is False
        assert post.call_args.kwargs["headers"][HEADER] == SECRET
        post.reset_mock()
        with pytest.raises(api_client.ApiError):
            api_client.predict_credit_api(PAYLOAD, base="https://external.example")
        assert api_client.explain_credit_api(PAYLOAD, base="https://external.example") is None
        post.assert_not_called()


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_documentation_local_explicit(path, monkeypatch):
    monkeypatch.delenv("ENABLE_LOCAL_API_DOCS", raising=False)
    local = TestClient(server.app, client=("127.0.0.1",1234))
    assert local.get(path).status_code == 404
    monkeypatch.setenv("ENABLE_LOCAL_API_DOCS", "1")
    assert local.get(path).status_code == 200
    assert TestClient(server.app, client=("198.51.100.1",1234)).get(path).status_code == 404


def test_configuration_redacts_secret(monkeypatch, capsys):
    import runpy, sys
    monkeypatch.setattr(sys, "argv", ["runtime_config", "--include-dotenv"])
    with patch("dotenv.dotenv_values", return_value={"INTERSERVICE_SECRET": SECRET}):
        runpy.run_module("utils.runtime_config", run_name="__main__")
    output = capsys.readouterr().out
    assert SECRET not in output
    assert "[REDACTED]" in output


def test_duplicate_header_refused():
    response = TestClient(server.app).post("/api/predict", json=PAYLOAD,
        headers=[(HEADER, SECRET), (HEADER, SECRET)])
    assert response.status_code == 401


def test_business_error_does_not_disclose_credential(caplog, capsys):
    with patch.object(server, "_compute_pd", side_effect=RuntimeError(SECRET)):
        response = TestClient(server.app).post("/api/predict", json=PAYLOAD, headers={HEADER: SECRET})
    assert response.status_code == 500
    captured = capsys.readouterr()
    assert SECRET not in response.text + caplog.text + captured.out + captured.err


def test_client_missing_secret_never_sends(monkeypatch):
    monkeypatch.delenv("INTERSERVICE_SECRET", raising=False)
    with patch("utils.llm_runtime.get_setting", return_value=""), patch("utils.api_client.requests.post") as post:
        with pytest.raises(api_client.ApiError):
            api_client.predict_credit_api(PAYLOAD)
        assert api_client.explain_credit_api(PAYLOAD) is None
        post.assert_not_called()
