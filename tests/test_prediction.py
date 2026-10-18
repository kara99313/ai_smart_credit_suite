import copy
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from fastapi.testclient import TestClient
import server
from utils import api_client, runtime_config

PAYLOAD = dict(DTIRatio=0.35, TrustScorePsychometric=0.62, HouseholdSize=4,
    NumCreditLines=2, Income=300000., CommunityGroupMember=False,
    HasMortgage=False, MonthsEmployed=36, HasSocialAid=False,
    MobileMoneyTransactions=120, Age=32, InterestRate=12., LoanTerm=24,
    LoanAmount=800000., InformalIncome=50000., threshold=0.10)


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(server.app)

    def test_category_aliases_and_rejection(self):
        for value in (True, 1, "Oui", "Yes", "true"):
            self.assertEqual(server._normalize_prediction_category(value), "Yes")
        for value in (False, 0, "Non", "No", "false"):
            self.assertEqual(server._normalize_prediction_category(value), "No")
        for value in (None, "unknown", "", 2, [], float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                server._normalize_prediction_category(value)

    def test_actual_pipeline_receives_different_categories(self):
        pipe = server._pipe
        for name in server.CATEGORICAL:
            with self.subTest(category=name):
                first = copy.deepcopy(PAYLOAD)
                second = dict(first, **{name: True})
                with patch.object(pipe, "predict_proba", wraps=pipe.predict_proba) as predict:
                    responses = [self.client.post("/api/predict", json=p) for p in (first, second)]
                self.assertEqual([r.status_code for r in responses], [200, 200])
                frames = [call.args[0] for call in predict.call_args_list]
                self.assertEqual(len(frames), 2)
                self.assertEqual(list(frames[0]), list(pipe.feature_names_in_))
                self.assertEqual(frames[0][name].iloc[0], "No")
                self.assertEqual(frames[1][name].iloc[0], "Yes")
                transformed = [pipe.named_steps["preprocessing"].transform(frame) for frame in frames]
                self.assertFalse(np.array_equal(*transformed))
                for response, frame in zip(responses, frames):
                    data = response.json()
                    expected = pipe.predict_proba(frame)[0, list(pipe.classes_).index(1)]
                    self.assertEqual(data["pd"], expected)
                    self.assertTrue(0 <= data["pd"] <= 1)
                    self.assertEqual(data["score_1000"], round((1 - expected) * 1000))
                    self.assertEqual(data["threshold"], 0.10)
                if name == "HasMortgage":
                    self.assertNotEqual(responses[0].json()["pd"], responses[1].json()["pd"])

    def test_unknown_category_and_missing_input_are_422(self):
        for name in server.CATEGORICAL:
            response = self.client.post("/api/predict", json=dict(PAYLOAD, **{name: "unknown"}))
            self.assertEqual(response.status_code, 422)
        missing = dict(PAYLOAD)
        del missing["Income"]
        self.assertEqual(self.client.post("/api/predict", json=missing).status_code, 422)
        response = self.client.post("/api/predict", json=dict(PAYLOAD, Income="NaN"))
        self.assertEqual(response.status_code, 422)

    def test_echo_reports_actual_yes_no_inputs(self):
        response = self.client.post("/api/predict_echo", json=dict(PAYLOAD, HasMortgage="Oui"))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["row0"]["HasMortgage"], "Yes")

    def test_default_class_is_located_not_assumed(self):
        fake = SimpleNamespace(classes_=np.array([1, 0]), predict_proba=lambda frame: np.array([[.7, .3]]))
        with patch.object(server, "_pipe", fake):
            self.assertEqual(server._compute_pd(server.pd.DataFrame([{}])), .7)

    def test_invalid_model_output_never_fabricates_pd(self):
        invalid = [None, np.array([.2, .8]), np.array([[np.nan, .8]]),
                   np.array([[-.1, 1.1]]), np.array([[.2, .2]])]
        for output in invalid:
            fake = SimpleNamespace(classes_=np.array([0, 1]), predict_proba=lambda frame, out=output: out)
            with self.subTest(output=output), patch.object(server, "_pipe", fake), self.assertRaises(RuntimeError):
                server._compute_pd(server.pd.DataFrame([{}]))
        for fake in (SimpleNamespace(predict=lambda frame: [1]),
                     SimpleNamespace(classes_=np.array([0, 2]), predict_proba=lambda frame: [[.2, .8]])):
            with patch.object(server, "_pipe", fake), self.assertRaises(RuntimeError):
                server._compute_pd(server.pd.DataFrame([{}]))

    def test_probability_failure_is_explicit_http_error(self):
        with patch.object(server, "_compute_pd", side_effect=RuntimeError("No valid probability")):
            response = self.client.post("/api/predict", json=PAYLOAD)
        self.assertEqual(response.status_code, 500)
        self.assertNotIn("pd", response.json())

    def test_unknown_normalized_label_is_rejected_before_encoder(self):
        encoder = server._pipe.named_steps["preprocessing"].named_transformers_["cat"]
        with patch.object(encoder, "categories_", [np.array(["False", "True"])] * 3):
            with self.assertRaises(RuntimeError):
                server._prepare_prediction_df(PAYLOAD)

    def test_api_client_rejects_incomplete_or_invalid_pd(self):
        good = self.client.post("/api/predict", json=PAYLOAD).json()
        for value in (None, float("nan"), float("inf"), -.1, 1.1, True, "0.5"):
            with self.subTest(value=value), self.assertRaises(api_client.ApiError):
                api_client._validate_prediction_response(dict(good, pd=value))
        incomplete = dict(good)
        del incomplete["pd"]
        with self.assertRaises(api_client.ApiError):
            api_client._validate_prediction_response(incomplete)

    def test_invalid_api_configuration_is_explicit_client_error(self):
        with self.assertRaises(api_client.ApiError):
            api_client.predict_credit_api(PAYLOAD, base="not-an-http-url")

    def test_launcher_uses_environment_over_dotenv_without_printing_secrets(self):
        import json, os, subprocess, sys
        env = dict(os.environ, API_PORT="19001", WEB_PORT="18501", BACKEND_URL="http://127.0.0.1:19001")
        result = subprocess.run([sys.executable, "-B", "-m", "utils.runtime_config"],
            env=env, text=True, capture_output=True, check=True, timeout=30)
        settings = json.loads(result.stdout)
        self.assertEqual(settings["api_port"], 19001)
        self.assertEqual(settings["web_port"], 18501)
        self.assertEqual(settings["backend_url"], "http://127.0.0.1:19001")
        self.assertEqual(settings["dotenv_environment"], {})

    def test_api_url_priority_and_bind_fallback(self):
        with patch.dict(runtime_config.os.environ, {"BACKEND_URL": "http://backend:9000/", "API_BASE": "http://alias:9001"}, clear=True):
            self.assertEqual(api_client._resolve_base(None), "http://backend:9000")
            self.assertEqual(api_client._resolve_base("http://explicit:9002/"), "http://explicit:9002")
            del runtime_config.os.environ["BACKEND_URL"]
            self.assertEqual(api_client._resolve_base(None), "http://alias:9001")
        with patch.dict(runtime_config.os.environ, {"API_HOST": "0.0.0.0", "API_PORT": "19000"}, clear=True):
            self.assertEqual(api_client._resolve_base(None), "http://127.0.0.1:19000")
        with patch.dict(runtime_config.os.environ, {}, clear=True):
            self.assertEqual(api_client._resolve_base(None), "http://127.0.0.1:18000")


if __name__ == "__main__":
    unittest.main()
