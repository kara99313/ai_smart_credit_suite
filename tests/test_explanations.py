import copy
import csv
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from interservice_support import TestClient
import server
from test_prediction import PAYLOAD
from utils import api_client
from utils.scoring_explanations import global_coefficient_importance

NUMERIC = ["DTIRatio", "TrustScorePsychometric", "HouseholdSize", "NumCreditLines",
           "Income", "MonthsEmployed", "MobileMoneyTransactions", "Age", "InterestRate",
           "LoanTerm", "LoanAmount", "InformalIncome"]
TRANSFORMED = NUMERIC + ["HasMortgage_Yes", "HasSocialAid_Yes", "CommunityGroupMember_Yes"]


class ExplanationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.page = runpy.run_path(str(Path("app/prediction.py")), run_name="explanation_helpers")

    def setUp(self):
        self.client = TestClient(server.app)

    def explanation(self, payload=None):
        return self.client.post("/tools/explain?top_k=30", json=payload or PAYLOAD).json()

    def test_exact_15_coefficient_feature_mapping(self):
        df = server._prepare_prediction_df(PAYLOAD)
        result = global_coefficient_importance(server._pipe, df)
        self.assertEqual([item["feature"] for item in result["items"]], TRANSFORMED)
        historical = dict(zip(TRANSFORMED, server._pipe.steps[-1][1].coef_[0]))
        # Historical local exports are optional; a clean checkout still tests the tracked model.
        export = Path("model/outputs/feature_importance_weighted.csv")
        if export.exists():
            with export.open(encoding="utf-8", newline="") as stream:
                self.assertEqual(historical, {row["feature"]: float(row["coefficient"]) for row in csv.DictReader(stream)})
        self.assertEqual(len(result["items"]), 15)
        for item in result["items"]:
            self.assertEqual(item["coefficient"], historical[item["feature"]])
            self.assertEqual(item["importance"], abs(historical[item["feature"]]))
        values = {item["feature"]: item["coefficient"] for item in result["items"]}
        self.assertEqual(values["DTIRatio"], historical["DTIRatio"])
        self.assertNotEqual(values["DTIRatio"], historical["HasMortgage_Yes"])
        self.assertNotIn("HasMortgage", values)

    def test_api_global_fallback_has_no_individual_contributions(self):
        result = self.explanation()
        self.assertTrue(result["ok"])
        self.assertEqual(result["method"], "global_importance")
        self.assertEqual(result["scope"], "global")
        self.assertEqual(result["space"], "transformed_features")
        self.assertEqual(result["coefficient_space"], "log_odds")
        self.assertEqual(result["class_label"], 1)
        self.assertEqual(result["fallback_reason"], "individual_shap_not_validated")
        self.assertEqual(len(result["items"]), 15)
        self.assertTrue(all("contribution" not in item for item in result["items"]))
        self.assertNotIn("base_value", result)

    def test_prediction_and_explanation_share_normalized_inputs_and_probability(self):
        pipe = server._pipe
        payload = dict(PAYLOAD, HasMortgage="Oui", HasSocialAid="Non")
        with patch.object(pipe, "predict_proba", wraps=pipe.predict_proba) as predict:
            prediction = self.client.post("/api/predict", json=payload).json()
            result = self.explanation(dict(payload, expected_model_id=prediction["model_id"],
                expected_input_id=prediction["input_id"], expected_pd=prediction["pd"]))
        frames = [call.args[0] for call in predict.call_args_list]
        self.assertEqual(len(frames), 2)
        self.assertTrue(frames[0].equals(frames[1]))
        self.assertEqual(frames[1]["HasMortgage"].iloc[0], "Yes")
        self.assertEqual(result["predicted_pd"], prediction["pd"])
        for key in ("model_id", "input_id"):
            self.assertEqual(result[key], prediction[key])

    def test_boolean_and_yes_no_aliases_have_same_fingerprint(self):
        a = self.explanation(dict(PAYLOAD, HasMortgage=True))
        b = self.explanation(dict(PAYLOAD, HasMortgage="Yes"))
        self.assertEqual(a["input_id"], b["input_id"])
        self.assertEqual(a["predicted_pd"], b["predicted_pd"])

    def test_stale_prediction_or_changed_input_is_rejected(self):
        prediction = self.client.post("/api/predict", json=PAYLOAD).json()
        for extra in ({"expected_model_id": "other-model"}, {"expected_input_id": "other-input"},
                      {"expected_pd": .123}):
            response = self.client.post("/tools/explain", json=dict(PAYLOAD, **extra))
            self.assertEqual(response.status_code, 409)
        response = self.client.post("/tools/explain", json=dict(PAYLOAD, HasMortgage=True,
            expected_input_id=prediction["input_id"]))
        self.assertEqual(response.status_code, 409)

    def test_unknown_explanation_category_is_rejected(self):
        self.assertEqual(self.client.post("/tools/explain", json=dict(PAYLOAD, HasMortgage="unknown")).status_code, 422)

    def test_bad_coefficient_dimensions_return_explicit_unavailable(self):
        copied_pipeline = copy.deepcopy(server._pipe)
        copied_pipeline.steps[-1][1].coef_ = copied_pipeline.steps[-1][1].coef_[:, :-1]
        with self.assertRaises(ValueError):
            global_coefficient_importance(copied_pipeline, server._prepare_prediction_df(PAYLOAD))
        with patch("utils.scoring_explanations.global_coefficient_importance", side_effect=ValueError("Invalid coefficient dimensions")):
            result = self.explanation()
        self.assertFalse(result["ok"])
        self.assertEqual(result["method"], "unavailable")
        self.assertEqual(result["reason"], "invalid_global_explanation")
        self.assertEqual(result["items"], [])

    def test_bad_feature_names_never_fabricate_importance(self):
        preprocessor = server._pipe.named_steps["preprocessing"]
        with patch.object(preprocessor, "get_feature_names_out", return_value=np.array(["same"] * 15)):
            result = self.explanation()
        self.assertFalse(result["ok"])
        self.assertEqual(result["items"], [])

    def test_global_class_orientation_follows_numeric_default_class(self):
        copied_pipeline = copy.deepcopy(server._pipe)
        estimator = copied_pipeline.steps[-1][1]
        original = estimator.coef_.copy()
        estimator.classes_ = np.array([1, 0])
        result = global_coefficient_importance(copied_pipeline, server._prepare_prediction_df(PAYLOAD))
        np.testing.assert_array_equal([item["coefficient"] for item in result["items"]], -original[0])

    def test_page_parser_keeps_importance_separate_and_does_not_add_raw_categories(self):
        frame, method = self.page["_parse_explain"](self.explanation(), PAYLOAD)
        self.assertEqual(method, "global_importance")
        self.assertEqual(set(frame["feature"]), set(TRANSFORMED))
        self.assertNotIn("contribution", frame.columns)
        self.assertNotIn("sign", frame.columns)
        self.assertNotIn("_waterfall_plot", self.page)

    def test_page_parser_rejects_mixed_importance_and_contribution(self):
        result = self.explanation()
        result["items"][0]["contribution"] = .5
        with self.assertRaises(ValueError):
            self.page["_parse_explain"](result, PAYLOAD)

    def test_legacy_report_consumers_do_not_receive_global_importance(self):
        result = self.explanation()
        response = SimpleNamespace(ok=True, status_code=200, json=lambda: result)
        with patch.object(api_client.requests, "post", return_value=response):
            self.assertIsNone(api_client.explain_credit_api(PAYLOAD))
            self.assertEqual(api_client.explain_credit_api(PAYLOAD, top_k=30, allow_global=True), result)

    def test_client_rejects_invalid_or_unrelated_explanations(self):
        prediction = self.client.post("/api/predict", json=PAYLOAD).json()
        good = self.explanation()
        candidates = []
        for key,value in (("model_id", "other-model"), ("input_id", "other-input"),
                          ("predicted_pd", .123), ("method", "shap"), ("scope", "individual"),
                          ("coefficient_space", "probability"), ("feature_count", 30)):
            candidates.append(dict(good, **{key:value}))
        mixed = copy.deepcopy(good)
        mixed["items"][0]["contribution"] = .5
        candidates.append(mixed)
        for result in candidates:
            response = SimpleNamespace(ok=True, status_code=200, json=lambda r=result: r)
            with self.subTest(result=result), patch.object(api_client.requests, "post", return_value=response):
                self.assertIsNone(api_client.explain_credit_api(PAYLOAD, top_k=30, allow_global=True, prediction=prediction))
