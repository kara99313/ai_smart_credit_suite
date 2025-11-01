# utils/monitoring_tools.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats  # KS test

LOG_PATH = Path("data/predictions_log.csv")


# =============================
# Helpers (timestamps, rolling)
# =============================
def coerce_timestamp_utc_naive(ts_series: pd.Series) -> pd.Series:
    """Convertit une série en datetime UTC puis supprime le timezone (naïf).
    Évite les comparaisons invalides (aware vs naive).
    """
    return pd.to_datetime(ts_series, errors="coerce", utc=True).dt.tz_convert(None)


def compute_daily_rates(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Agrège par jour : volume total et taux de défaut (>= threshold)."""
    if df.empty or "timestamp" not in df or "prob_default" not in df:
        return pd.DataFrame(columns=["date", "n", "rate"])
    tmp = df.copy()
    tmp["date"] = tmp["timestamp"].dt.date
    out = (
        tmp.groupby("date", as_index=False)
           .apply(lambda g: pd.Series({
               "n": len(g),
               "rate": (g["prob_default"] >= threshold).mean() if len(g) else 0.0
           }))
           .reset_index(drop=True)
           .sort_values("date")
    )
    return out


def rolling_mean(y: pd.Series, window: int = 7) -> pd.Series:
    """Moyenne mobile simple (par défaut 7)."""
    return y.rolling(window=window, min_periods=1).mean()


# =============================
# Chargement & prétraitements
# =============================
def load_logs(path: Union[str, Path] = LOG_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=[
            "timestamp","client_id","prob_default","pred_label","threshold","inputs_json","model_tag"
        ])
    df = pd.read_csv(path)

    # Timestamps harmonisés (UTC -> naïf)
    if "timestamp" in df.columns:
        df["timestamp"] = coerce_timestamp_utc_naive(df["timestamp"])

    # Types numériques
    for c in ["prob_default", "threshold"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "pred_label" in df.columns:
        df["pred_label"] = pd.to_numeric(df["pred_label"], errors="coerce").astype("Int64")

    return df.dropna(subset=["timestamp"]).copy()


# =============================
# KPI simples
# =============================
def compute_basic_kpis(df: pd.DataFrame, threshold: float = 0.50) -> Dict[str, Any]:
    n = len(df)
    if n == 0:
        return {
            "n_rows": 0,
            "accept_rate": None,
            "default_rate_at_threshold": None,
            "unique_clients": 0,
            "mean_prob": None,
        }
    accept = (df["prob_default"] < threshold).mean() if "prob_default" in df else None
    default_rate = (df["prob_default"] >= threshold).mean() if "prob_default" in df else None
    uniq = df["client_id"].nunique() if "client_id" in df else None
    mean_prob = df["prob_default"].mean() if "prob_default" in df else None
    return {
        "n_rows": n,
        "accept_rate": accept,
        "default_rate_at_threshold": default_rate,
        "unique_clients": uniq,
        "mean_prob": mean_prob
    }


# =============================
# PSI (Population Stability Index)
# =============================
def calc_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI ~ stabilité de distribution.
    ~0   : stable
    0.10–0.25 : léger drift
    >0.25 : drift marqué
    """
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual, dtype=float)
    if expected.size == 0 or actual.size == 0:
        return float("nan")

    # Bornes par quantiles sur la base
    quantiles = np.linspace(0, 1, bins + 1)
    cuts = np.quantile(expected, quantiles)
    cuts = np.unique(cuts)
    if cuts.size < 3:
        low, high = min(expected.min(), actual.min()), max(expected.max(), actual.max())
        if low == high:
            return 0.0
        cuts = np.linspace(low, high, bins + 1)

    exp_counts, _ = np.histogram(expected, bins=cuts)
    act_counts, _ = np.histogram(actual,   bins=cuts)

    exp_props = np.maximum(exp_counts / max(1, exp_counts.sum()), 1e-6)
    act_props = np.maximum(act_counts / max(1, act_counts.sum()), 1e-6)

    psi = np.sum((act_props - exp_props) * np.log(act_props / exp_props))
    return float(psi)


# =============================
# KS test (sur les scores)
# =============================
def ks_statistic(expected: np.ndarray, actual: np.ndarray) -> float:
    """KS entre deux distributions (scores). 0 → identiques, plus grand → différentes."""
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual, dtype=float)
    if expected.size == 0 or actual.size == 0:
        return float("nan")
    d, _ = stats.ks_2samp(expected, actual, alternative="two-sided", mode="auto")
    return float(d)


# =============================
# Rapport de drift
# =============================
@dataclass
class DriftReport:
    psi: Optional[float]
    ks: Optional[float]
    base_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]]
    current_period: Optional[Tuple[pd.Timestamp, pd.Timestamp]]
    n_base: int
    n_current: int
    flags: List[str]


def drift_report(
    df_base: pd.DataFrame,
    df_current: pd.DataFrame,
    score_col: str = "prob_default",
    psi_bins: int = 10,
    psi_warn: float = 0.10,
    psi_alert: float = 0.25,
    ks_warn: float = 0.12,
    ks_alert: float = 0.20
) -> DriftReport:
    flags: List[str] = []
    psi_val: Optional[float] = None
    ks_val: Optional[float] = None

    n_base, n_cur = len(df_base), len(df_current)
    base_period = (df_base["timestamp"].min(), df_base["timestamp"].max()) if n_base else None
    cur_period  = (df_current["timestamp"].min(), df_current["timestamp"].max()) if n_cur else None

    if n_base >= 50 and n_cur >= 50 and score_col in df_base and score_col in df_current:
        psi_val = calc_psi(df_base[score_col].values, df_current[score_col].values, bins=psi_bins)
        ks_val  = ks_statistic(df_base[score_col].values, df_current[score_col].values)

        # Règles simples
        if isinstance(psi_val, float) and not np.isnan(psi_val):
            if psi_val >= psi_alert:
                flags.append(f"PSI={psi_val:.3f} ≥ {psi_alert} (ALERTE)")
            elif psi_val >= psi_warn:
                flags.append(f"PSI={psi_val:.3f} ≥ {psi_warn} (AVERTISSEMENT)")
        if isinstance(ks_val, float) and not np.isnan(ks_val):
            if ks_val >= ks_alert:
                flags.append(f"KS={ks_val:.3f} ≥ {ks_alert} (ALERTE)")
            elif ks_val >= ks_warn:
                flags.append(f"KS={ks_val:.3f} ≥ {ks_warn} (AVERTISSEMENT)")
    else:
        if n_base < 50 or n_cur < 50:
            flags.append("Trop peu d’observations pour mesurer le drift (min 50 / période).")

    return DriftReport(
        psi=psi_val,
        ks=ks_val,
        base_period=base_period,
        current_period=cur_period,
        n_base=n_base,
        n_current=n_cur,
        flags=flags
    )


# =============================
# Règles d’alerte additionnelles
# =============================
def additional_alerts(
    df: pd.DataFrame,
    threshold: float = 0.50,
    min_volume: int = 100,
    max_drop_rate: float = 0.5
) -> List[str]:
    """Quelques règles “sanity check” :
    - Volume < min_volume
    - Chute > max_drop_rate vs semaine précédente (sur le volume)
    - Default rate anormalement haut (>80%)
    """
    alerts: List[str] = []
    n = len(df)
    if n < min_volume:
        alerts.append(f"Volume faible: {n} < {min_volume}")

    if "timestamp" in df.columns and n:
        df_w = df.copy()
        df_w["d"] = df_w["timestamp"].dt.floor("D")
        # semaine courante (7 derniers jours glissants)
        last_7 = df_w[df_w["d"] >= (df_w["d"].max() - pd.Timedelta(days=6))]
        # semaine précédente
        prev_7 = df_w[(df_w["d"] < (df_w["d"].max() - pd.Timedelta(days=6))) &
                      (df_w["d"] >= (df_w["d"].max() - pd.Timedelta(days=13)))]
        if len(prev_7) > 0:
            drop = 1 - (len(last_7) / len(prev_7))
            if drop > max_drop_rate:
                alerts.append(f"Grosse baisse de volume: -{drop:.0%} vs semaine précédente")

    if "prob_default" in df.columns:
        high_def = (df["prob_default"] >= threshold).mean()
        if high_def > 0.80:
            alerts.append(f"Taux défaut (≥ seuil) très élevé: {high_def:.0%}")

    return alerts
