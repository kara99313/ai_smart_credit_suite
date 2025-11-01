# utils/api_specs.py
from __future__ import annotations
from typing import Dict, List

# 15 features du modèle
FEATURES = [
    "DTIRatio",
    "TrustScorePsychometric",
    "HouseholdSize",
    "NumCreditLines",
    "Income",
    "CommunityGroupMember",
    "HasMortgage",
    "MonthsEmployed",
    "HasSocialAid",
    "MobileMoneyTransactions",
    "Age",
    "InterestRate",
    "LoanTerm",
    "LoanAmount",
    "InformalIncome",
]

# Endpoint standard de prédiction
PREDICT_PATH = "/api/predict"

def openapi_spec_dict(base_url: str = "http://localhost:8000") -> Dict:
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Credit Risk Scoring API",
            "version": "0.1.0",
            "description": "Endpoints REST pour scorer un client, récupérer les logs et vérifier la santé.",
        },
        "servers": [{"url": base_url}],
        "paths": {
            "/health": {
                "get": {"summary": "Healthcheck", "responses": {"200": {"description": "OK"}}}
            },
            "/score": {
                "post": {
                    "summary": "Calcule PD / score / rating",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {f: {"type": ["number", "boolean", "integer", "string"]} for f in FEATURES},
                                    "required": FEATURES,
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Résultat de scoring",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "pd": {"type": "number"},
                                            "score_1000": {"type": "integer"},
                                            "rating": {"type": "string"},
                                            "risk_level": {"type": "string"},
                                            "decision": {"type": "string"},
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
        },
    }

EXAMPLE_PAYLOAD = {
    "DTIRatio": 0.35,
    "TrustScorePsychometric": 0.62,
    "HouseholdSize": 4,
    "NumCreditLines": 2,
    "Income": 300000,
    "CommunityGroupMember": True,
    "HasMortgage": False,
    "MonthsEmployed": 36,
    "HasSocialAid": False,
    "MobileMoneyTransactions": 120,
    "Age": 32,
    "InterestRate": 12.0,
    "LoanTerm": 24,
    "LoanAmount": 800000,
    "InformalIncome": 50000,
}
