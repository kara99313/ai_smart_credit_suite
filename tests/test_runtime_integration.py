from synthetic_history import LOG, SyntheticHistoryTestCase
"""Opt-in live smoke test: RUN_RUNTIME_SMOKE=1 python -B -m unittest discover -s tests."""
import hashlib
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import requests
from test_prediction import PAYLOAD
from utils.api_client import predict_credit_api

ROOT = Path(__file__).resolve().parents[1]


def available_port(preferred):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", preferred))
        except OSError:
            sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@unittest.skipUnless(os.getenv("RUN_RUNTIME_SMOKE") == "1", "Live servers require RUN_RUNTIME_SMOKE=1")
class RuntimeIntegrationTests(SyntheticHistoryTestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="credit-runtime-")
        cls.children = []
        cls.handles = []
        cls.log_before = LOG.read_bytes()
        cls.model_before = hashlib.sha256((ROOT / "model/logistic_pipeline_best.pkl").read_bytes()).hexdigest()
        cls.api_port = available_port(18000)
        cls.ui_port = available_port(8501)
        cls.base = f"http://127.0.0.1:{cls.api_port}"
        cls.env = dict(os.environ, BACKEND_URL=cls.base, API_BASE=cls.base,
            API_PORT=str(cls.api_port), WEB_PORT=str(cls.ui_port),
            PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1")
        try:
            cls.start("api", ["-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(cls.api_port)])
            cls.wait_for(cls.base + "/health", cls.children[-1])
            print(f"LIVE FastAPI ready: {cls.base}", flush=True)
            cls.start("streamlit", ["-m", "streamlit", "run", "streamlit_app.py",
                "--server.address=127.0.0.1", f"--server.port={cls.ui_port}",
                "--server.headless=true", "--browser.gatherUsageStats=false"])
            cls.wait_for(f"http://127.0.0.1:{cls.ui_port}/_stcore/health", cls.children[-1])
            print(f"LIVE Streamlit ready: http://127.0.0.1:{cls.ui_port}", flush=True)
        except BaseException:
            cls.cleanup()
            raise

    @classmethod
    def start(cls, name, arguments):
        handle = open(Path(cls.temp.name) / (name + ".log"), "w+", encoding="utf-8")
        cls.handles.append(handle)
        cls.children.append(subprocess.Popen([sys.executable, "-B", "-X", "utf8", *arguments],
            cwd=ROOT, env=cls.env, stdout=handle, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)))

    @classmethod
    def wait_for(cls, url, process):
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if process.poll() is not None:
                cls.handles[-1].seek(0)
                raise AssertionError(cls.handles[-1].read()[-6000:])
            try:
                if requests.get(url, timeout=1).ok:
                    return
            except requests.RequestException:
                pass
            time.sleep(.25)
        raise AssertionError("Startup timeout: " + url)

    @classmethod
    def cleanup(cls):
        for child in reversed(cls.children):
            if child.poll() is None:
                if os.name == "nt":
                    # The Windows venv launcher owns the actual interpreter child.
                    subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    child.wait(timeout=10)
                else:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait(timeout=10)
        for handle in cls.handles:
            handle.close()
        temp_dir = Path(cls.temp.name).resolve()
        if temp_dir.parent != Path(tempfile.gettempdir()).resolve() or not temp_dir.name.startswith("credit-runtime-"):
            raise RuntimeError("Unexpected temporary cleanup path")
        for attempt in range(20):
            try:
                cls.temp.cleanup()
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(.1)

    @classmethod
    def tearDownClass(cls):
        cls.cleanup()
        assert LOG.read_bytes() == cls.log_before, "Production log was modified"
        assert hashlib.sha256((ROOT / "model/logistic_pipeline_best.pkl").read_bytes()).hexdigest() == cls.model_before

    def test_live_http_prediction_and_category_echo(self):
        results = []
        for flag in (False, True):
            payload = dict(PAYLOAD, HasMortgage=flag)
            result = predict_credit_api(payload, base=self.base)
            self.assertTrue(0 <= result["pd"] <= 1)
            echo = requests.post(self.base + "/api/predict_echo", json=payload, timeout=10)
            self.assertEqual(echo.status_code, 200)
            self.assertEqual(echo.json()["row0"]["HasMortgage"], "Yes" if flag else "No")
            results.append(result["pd"])
        self.assertNotEqual(*results)
        print("LIVE PD HasMortgage No/Yes:", results, flush=True)

    def prediction_page(self):
        from streamlit.testing.v1 import AppTest
        script = ("import runpy\n"
            + f"ns = runpy.run_path({str(ROOT / 'app/prediction.py')!r}, run_name='prediction_smoke')\n"
            + f"ns['main'].__globals__['LOG_PATH'] = {str(Path(self.temp.name) / 'predictions.csv')!r}\n"
            + f"ns['main'].__globals__['REPORTS_DIR'] = {self.temp.name!r}\n"
            + "ns['main']()\n")
        return AppTest.from_string(script, default_timeout=90)

    def test_streamlit_prediction_page_calls_live_api(self):
        with patch.dict(os.environ, {"BACKEND_URL": self.base, "API_BASE": self.base}):
            app = self.prediction_page().run()
            self.assertEqual(len(app.exception), 0, str(app.exception))
            app.toggle[0].set_value(False)
            for value in ("Non", "Oui"):
                app.selectbox[1].set_value(value)
                with patch("utils.api_client.requests.post", wraps=requests.post) as post:
                    app.button[0].click().run()
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertEqual(len(app.error), 0, str(app.error))
                calls = [call for call in post.call_args_list if call.args[0].endswith("/api/predict")]
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0].kwargs["json"]["HasMortgage"], value == "Oui")
            print("STREAMLIT AppTest: two real widget predictions reached FastAPI", flush=True)

    def test_streamlit_with_global_explanation_enabled(self):
        from utils.reporting_pdf import build_pdf_bytes
        with patch.dict(os.environ, {"BACKEND_URL": self.base, "API_BASE": self.base}):
            app = self.prediction_page().run()
            self.assertEqual(len(app.exception), 0, str(app.exception))
            self.assertTrue(app.toggle[0].value)
            with patch("utils.api_client.requests.post", wraps=requests.post) as post, \
                 patch("utils.reporting_pdf.build_pdf_bytes", wraps=build_pdf_bytes) as pdf:
                app.button[0].click().run()
            self.assertEqual(len(app.exception), 0, str(app.exception))
            self.assertEqual(len(app.error), 0, str(app.error))
            explanation_calls = [c for c in post.call_args_list if "/tools/explain?" in c.args[0]]
            self.assertEqual(len(explanation_calls), 1)
            body = explanation_calls[0].kwargs["json"]
            self.assertTrue(body["expected_model_id"])
            self.assertTrue(body["expected_input_id"])
            self.assertTrue(0 <= body["expected_pd"] <= 1)
            frame = app.dataframe[0].value
            self.assertEqual(len(frame), 15)
            self.assertIn("HasMortgage_Yes", set(frame["feature"]))
            self.assertNotIn("contribution", frame.columns)
            self.assertTrue(pdf.called)
            self.assertTrue(all(call.kwargs["df_exp"] is None for call in pdf.call_args_list))
            labels = " ".join(element.value for element in app.subheader)
            self.assertIn("Importance globale", labels)
            self.assertNotIn("Waterfall", labels)
            self.assertFalse(any("augmentent la pd" in element.value.lower() for element in app.markdown))
            print("STREAMLIT: explanation enabled, 15 global importances, no individual attribution", flush=True)

    def test_streamlit_invalid_explanation_is_explicitly_unavailable(self):
        from types import SimpleNamespace
        real_post = requests.post
        def invalid_explanation(url, **kwargs):
            if "/tools/explain?" in url:
                return SimpleNamespace(ok=True, json=lambda: {"ok": True, "method": "shap", "scope": "individual",
                    "items": [{"feature": "HasMortgage", "contribution": .5}]})
            return real_post(url, **kwargs)
        with patch.dict(os.environ, {"BACKEND_URL": self.base, "API_BASE": self.base}):
            app = self.prediction_page().run()
            with patch("utils.api_client.requests.post", side_effect=invalid_explanation):
                app.button[0].click().run()
            self.assertEqual(len(app.exception), 0, str(app.exception))
            self.assertEqual(len(app.error), 0, str(app.error))
            self.assertTrue(any("Aucune explication fiable" in element.value for element in app.warning))
            self.assertEqual(len(app.dataframe), 0)
            self.assertFalse(any("augmentent la pd" in element.value.lower() for element in app.markdown))

    def test_live_prediction_html_pdf_history_and_reports_are_consistent(self):
        from test_reports import reports_page, pdf_text
        from utils.report_generator import render_html_report, report_display_values
        from utils.reporting_pdf import build_pdf_bytes
        from utils.csv_utils import read_logs_robust
        contexts, htmls, pdfs = [], [], []
        def capture_html(ctx):
            contexts.append(dict(ctx))
            rendered = render_html_report(ctx)
            htmls.append(rendered)
            return rendered
        def capture_pdf(*args, **kwargs):
            rendered = build_pdf_bytes(*args, **kwargs)
            pdfs.append(rendered)
            return rendered
        with patch.dict(os.environ, {"BACKEND_URL": self.base, "API_BASE": self.base}):
            page = self.prediction_page().run()
            page.text_input[0].set_value("PHASE6-REPORT-CLIENT")
            page.toggle[0].set_value(False)
            with patch("utils.report_generator.render_html_report", side_effect=capture_html), \
                 patch("utils.reporting_pdf.build_pdf_bytes", side_effect=capture_pdf), \
                 patch("utils.api_client.requests.post", wraps=requests.post) as post:
                page.button[0].click().run()
            self.assertEqual(len(page.exception), 0, str(page.exception))
            self.assertEqual(len(page.error), 0, str(page.error))
            call = next(call for call in post.call_args_list if call.args[0].endswith("/api/predict"))
            api = requests.post(self.base + "/api/predict", json=call.kwargs["json"], timeout=10).json()
            ctx = contexts[0]
            for key in ("score_1000", "rating", "decision", "threshold", "model_id", "input_id"):
                self.assertEqual(ctx[key], api[key])
            self.assertEqual(ctx["pd_value"], api["pd"])
            self.assertEqual(ctx["client_id"], "PHASE6-REPORT-CLIENT")
            text = pdf_text(pdfs[0])
            for value in report_display_values(ctx).values():
                self.assertIn(value, htmls[0])
                self.assertIn(value, text)
            cards = " ".join(element.value for element in page.markdown)
            for key in ("pd_value", "score_1000", "rating", "decision", "risk_level"):
                self.assertIn(report_display_values(ctx)[key], cards)
            history = read_logs_robust(Path(self.temp.name) / "predictions.csv")
            archived = history[history["client_id"] == ctx["client_id"]].iloc[-1]
            self.assertEqual(archived["score_1000"], ctx["score_1000"])
            self.assertEqual(archived["risk_level"], ctx["risk_level"])
            report = reports_page(Path(self.temp.name) / "predictions.csv", self.temp.name).run()
            report.selectbox[0].set_value(ctx["client_id"]).run()
            self.assertEqual(len(report.exception), 0, str(report.exception))
            self.assertEqual(len(report.error), 0, str(report.error))
            expected = report_display_values(ctx)
            for metric, key in zip(report.metric, ("pd_value", "score_1000", "rating", "risk_level", "decision", "threshold")):
                self.assertEqual(metric.value, expected[key])
            print("LIVE Reports: API = prediction screen = HTML = PDF = reopened history", flush=True)

    def test_home_health_button_checks_real_fastapi_without_predicting(self):
        from streamlit.testing.v1 import AppTest
        from test_home import button
        with patch.dict(os.environ, {"BACKEND_URL": self.base, "API_BASE": self.base}):
            page = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=90).run()
            with patch("utils.api_client.requests.get", wraps=requests.get) as get, \
                 patch("utils.api_client.requests.post", wraps=requests.post) as post:
                button(page, "V\u00e9rifier FastAPI").click().run()
            self.assertEqual(len(page.exception), 0, str(page.exception))
            self.assertEqual(len(page.error), 0, str(page.error))
            get.assert_called_once_with(self.base + "/health", timeout=3)
            post.assert_not_called()
            text = " ".join(str(item.value) for item in page.markdown)
            self.assertIn("Joignable (/health)", text)
            self.assertIn("scoring : Indisponible", text)
            self.assertEqual(page.metric[0].value, "4")
            print("LIVE Home: GET /health verified; no prediction, model/RAG/LLM not certified", flush=True)

    def test_health_check_live_api_and_inference_leave_history_unchanged(self):
        from streamlit.testing.v1 import AppTest
        from test_health_check import button
        from utils.prediction_logs import snapshot_path
        path = LOG
        before = path.read_bytes()
        sidecar = snapshot_path(path)
        sidecar_before = sidecar.read_bytes() if sidecar.exists() else None
        with patch.dict(os.environ, {"BACKEND_URL": self.base, "API_BASE": self.base}):
            page = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=90).run()
            page.switch_page("app/health_check.py").run()
            self.assertEqual(len(page.exception), 0, str(page.exception))
            with patch("utils.health_diagnostics.requests.get", wraps=requests.get) as get, \
                 patch("utils.health_diagnostics.requests.post", wraps=requests.post) as post:
                button(page, "Tester GET /health").click().run()
                self.assertEqual(len(page.exception), 0, str(page.exception))
                get.assert_called_once_with(self.base + "/health", timeout=4)
                post.assert_not_called()
                button(page, "Tester une inference synthetique").click().run()
                self.assertEqual(len(page.exception), 0, str(page.exception))
                self.assertEqual(len(page.error), 0, str(page.error))
                self.assertEqual(post.call_count, 1)
                self.assertEqual(post.call_args.args[0], self.base + "/api/predict")
            texts = " ".join(str(item.value) for item in page.markdown)
            self.assertIn("Inference du modele : Fonctionnel", texts)
            self.assertIn("FastAPI : Disponible", texts)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(sidecar.read_bytes() if sidecar.exists() else None, sidecar_before)
        print("LIVE Health Check: health then explicit synthetic inference; CSV and JSONL unchanged", flush=True)
