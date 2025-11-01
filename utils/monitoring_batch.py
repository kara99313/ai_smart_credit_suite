# utils/monitoring_batch.py
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Outils déjà fournis
from utils.monitoring_tools import (
    load_logs,
    compute_basic_kpis,
    drift_report,
    calc_psi,
    ks_statistic,
)

# -------- Params par défaut --------
DEFAULT_LOG = Path("data/predictions_log.csv")
DEFAULT_OUT = Path("monitoring_outputs")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _to_naive_utc(ts: pd.Series) -> pd.Series:
    """Normalise les timestamps en datetime *naïf* UTC-like pour comparaisons robustes."""
    s = pd.to_datetime(ts, errors="coerce", utc=True)
    return s.dt.tz_convert(None)


def _make_windows(
    df: pd.DataFrame,
    end_dt: Optional[datetime],
    base_days: int,
    current_days: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Tuple[datetime, datetime], Tuple[datetime, datetime]]:
    """
    Construit 2 fenêtres non chevauchantes :
      - Base:    [end - (current_days + base_days) + 1  →  end - current_days]
      - Current: [end - current_days + 1               →  end]
    """
    if df.empty:
        now = datetime.utcnow()
        return df, df, (now, now), (now, now)

    # borne supérieure = min(end_dt, max log)
    df = df.copy()
    df["timestamp"] = _to_naive_utc(df["timestamp"])
    max_ts = df["timestamp"].max()
    end_dt = min(_to_naive_utc(pd.Series([end_dt or max_ts]))[0], max_ts)

    # bornes
    cur_start = (end_dt - timedelta(days=current_days - 1)).replace(microsecond=0)
    cur_end = end_dt.replace(microsecond=0)

    base_end = (cur_start - timedelta(seconds=1)).replace(microsecond=0)
    base_start = (base_end - timedelta(days=base_days - 1)).replace(microsecond=0)

    base_mask = (df["timestamp"] >= base_start) & (df["timestamp"] <= base_end)
    cur_mask = (df["timestamp"] >= cur_start) & (df["timestamp"] <= cur_end)
    return (
        df[base_mask].copy(),
        df[cur_mask].copy(),
        (base_start, base_end),
        (cur_start, cur_end),
    )


def _dist_plot(
    base: pd.Series, cur: pd.Series, title: str = "Distribution des scores (base vs courant)"
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=base, name="Base", opacity=0.65, nbinsx=40))
    fig.add_trace(go.Histogram(x=cur, name="Courant", opacity=0.65, nbinsx=40))
    fig.update_layout(
        barmode="overlay",
        title=title,
        xaxis_title="Score / prob_default",
        yaxis_title="Comptes",
        height=420,
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _cdf_plot(base: pd.Series, cur: pd.Series, title: str = "CDF des scores") -> go.Figure:
    """Courbes cumulées pour visualiser un écart de distributions (KS visuel)."""
    b = pd.Series(sorted(base.dropna().values))
    c = pd.Series(sorted(cur.dropna().values))
    b_cdf = pd.Series(range(1, len(b) + 1)) / max(1, len(b))
    c_cdf = pd.Series(range(1, len(c) + 1)) / max(1, len(c))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=b, y=b_cdf, mode="lines", name="Base CDF"))
    fig.add_trace(go.Scatter(x=c, y=c_cdf, mode="lines", name="Courant CDF"))
    fig.update_layout(
        title=title,
        xaxis_title="Score / prob_default",
        yaxis_title="F(x)",
        height=420,
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def run_batch(
    log_path: Path = DEFAULT_LOG,
    out_dir: Path = DEFAULT_OUT,
    score_col: str = "prob_default",
    base_days: int = 30,
    current_days: int = 7,
    threshold: float = 0.50,
    psi_bins: int = 10,
    psi_warn: float = 0.10,
    psi_alert: float = 0.20,
    ks_warn: float = 0.12,
    ks_alert: float = 0.20,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Pipeline batch complet. Retourne un dict récapitulatif et écrit des fichiers dans out_dir."""
    _ensure_dir(out_dir)
    df = load_logs(log_path)
    if df.empty:
        summary = {
            "status": "empty",
            "message": f"Aucune donnée dans {log_path.as_posix()}",
            "outputs": {},
        }
        (out_dir / "last_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return summary

    # Normalisation timestamps
    df["timestamp"] = _to_naive_utc(df["timestamp"])

    # end_date optionnelle (YYYY-MM-DD)
    end_dt = None
    if end_date:
        try:
            end_dt = pd.to_datetime(end_date, utc=True).tz_convert(None).to_pydatetime()
        except Exception:
            end_dt = None

    base_df, cur_df, base_period, cur_period = _make_windows(df, end_dt, base_days, current_days)

    # KPIs
    base_kpi = compute_basic_kpis(base_df, threshold=threshold)
    cur_kpi = compute_basic_kpis(cur_df, threshold=threshold)

    # Drift (PSI / KS + flags)
    rep = drift_report(
        base_df, cur_df,
        score_col=score_col,
        psi_bins=psi_bins,
        psi_warn=psi_warn,
        psi_alert=psi_alert,
        ks_warn=ks_warn,
        ks_alert=ks_alert
    )

    # Graphiques
    figs: Dict[str, go.Figure] = {}
    if not base_df.empty and not cur_df.empty and score_col in base_df and score_col in cur_df:
        figs["Distribution base vs courant"] = _dist_plot(base_df[score_col], cur_df[score_col])
        figs["CDF base vs courant"] = _cdf_plot(base_df[score_col], cur_df[score_col])

    # Sauvegardes
    ts_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outputs = {}

    # 1) Résumé JSON
    summary = {
        "status": "ok",
        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
        "log_path": str(log_path),
        "out_dir": str(out_dir),
        "score_col": score_col,
        "windows": {
            "base": {"start": base_period[0].isoformat(), "end": base_period[1].isoformat(), "rows": int(len(base_df))},
            "current": {"start": cur_period[0].isoformat(), "end": cur_period[1].isoformat(), "rows": int(len(cur_df))},
        },
        "kpis": {"base": base_kpi, "current": cur_kpi},
        "drift": {
            "psi": rep.psi,
            "ks": rep.ks,
            "flags": rep.flags,
            "n_base": rep.n_base,
            "n_current": rep.n_current,
            "base_period": [base_period[0].isoformat(), base_period[1].isoformat()],
            "current_period": [cur_period[0].isoformat(), cur_period[1].isoformat()],
        },
    }
    summary_path = out_dir / f"monitoring_summary_{ts_tag}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    outputs["summary_json"] = summary_path.as_posix()

    # 2) CSV des fenêtres (utile pour audit)
    base_csv = out_dir / f"window_base_{ts_tag}.csv"
    cur_csv = out_dir / f"window_current_{ts_tag}.csv"
    if not base_df.empty:
        base_df.to_csv(base_csv, index=False)
        outputs["base_csv"] = base_csv.as_posix()
    if not cur_df.empty:
        cur_df.to_csv(cur_csv, index=False)
        outputs["current_csv"] = cur_csv.as_posix()

    # 3) Graphs en HTML + PNG (si possible)
    try:
        for name, fig in figs.items():
            safe = name.lower().replace(" ", "_").replace("/", "_")
            html_path = out_dir / f"{safe}_{ts_tag}.html"
            fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
            outputs.setdefault("figures_html", []).append(html_path.as_posix())
            # Essai PNG (si kaleido installé)
            try:
                import plotly.io as pio  # noqa
                png_path = out_dir / f"{safe}_{ts_tag}.png"
                fig.write_image(str(png_path), format="png", scale=2)  # nécessite kaleido
                outputs.setdefault("figures_png", []).append(png_path.as_posix())
            except Exception:
                # Pas bloquant
                pass
    except Exception as e:
        outputs["figures_error"] = f"Erreur export figures: {e}"

    # Écrire un alias “last_summary.json”
    (out_dir / "last_summary.json").write_text(json.dumps({**summary, "outputs": outputs}, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"summary": summary, "outputs": outputs}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Batch de monitoring PSI/KS sur predictions_log.csv (drift & KPIs)."
    )
    p.add_argument("--log_path", type=str, default=str(DEFAULT_LOG), help="Chemin du CSV de logs (predictions_log.csv)")
    p.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT), help="Dossier de sortie pour les rapports/graphs")
    p.add_argument("--score_col", type=str, default="prob_default", help="Nom de la colonne score/proba")
    p.add_argument("--base_days", type=int, default=30, help="Taille de la fenêtre base (jours)")
    p.add_argument("--current_days", type=int, default=7, help="Taille de la fenêtre courante (jours)")
    p.add_argument("--threshold", type=float, default=0.50, help="Seuil d’acceptation (pour KPIs)")
    p.add_argument("--psi_bins", type=int, default=10, help="Nombre de bins pour PSI")
    p.add_argument("--psi_warn", type=float, default=0.10, help="Seuil d’avertissement PSI")
    p.add_argument("--psi_alert", type=float, default=0.20, help="Seuil d’alerte PSI")
    p.add_argument("--ks_warn", type=float, default=0.12, help="Seuil d’avertissement KS")
    p.add_argument("--ks_alert", type=float, default=0.20, help="Seuil d’alerte KS")
    p.add_argument("--end_date", type=str, default=None, help="Date fin (YYYY-MM-DD), sinon max(timestamp)")
    return p


def main():
    args = build_argparser().parse_args()
    ret = run_batch(
        log_path=Path(args.log_path),
        out_dir=Path(args.out_dir),
        score_col=args.score_col,
        base_days=args.base_days,
        current_days=args.current_days,
        threshold=args.threshold,
        psi_bins=args.psi_bins,
        psi_warn=args.psi_warn,
        psi_alert=args.psi_alert,
        ks_warn=args.ks_warn,
        ks_alert=args.ks_alert,
        end_date=args.end_date,
    )
    # Console friendly
    print(json.dumps(ret.get("summary", {}), indent=2, ensure_ascii=False))
    print("\nOutputs:")
    print(json.dumps(ret.get("outputs", {}), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
