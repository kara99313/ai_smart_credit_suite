"""Client dashboard calculations: recorded results and explicit threshold simulation."""
from __future__ import annotations
import math
import pandas as pd
import plotly.graph_objects as go
from utils.csv_utils import read_logs_robust
from utils.report_generator import normalize_report_context, report_display_values, _optional_text


def valid_number(value, maximum=1):
    try:
        number = float(value)
        if isinstance(value, bool) or not math.isfinite(number) or not 0 <= number <= maximum:
            return None
        return number
    except (ValueError, TypeError, OverflowError):
        return None


def load_history(path):
    df = read_logs_robust(path).copy()
    df['_timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True, format='mixed')
    df['_client'] = df['client_id'].map(lambda value: _optional_text(value) or '')
    df['_pd'] = df['prob_default'].map(valid_number)
    df['_threshold'] = df['threshold'].map(valid_number)
    return df.sort_values(['_timestamp', '_record_number'], ascending=[False, False],
                          na_position='last', kind='stable') if not df.empty else df


def filter_history(df, client=None, start=None, end=None, include_undated=True):
    filtered = df.copy()
    if client is not None:
        filtered = filtered[filtered['_client'] == client]
    if start is not None or end is not None:
        mask = filtered['_timestamp'].notna()
        if start is not None:
            mask &= filtered['_timestamp'] >= pd.Timestamp(start, tz='UTC')
        if end is not None:
            mask &= filtered['_timestamp'] < pd.Timestamp(end, tz='UTC') + pd.Timedelta(days=1)
        if include_undated:
            mask |= filtered['_timestamp'].isna()
        filtered = filtered[mask]
    return filtered


def calculate_kpis(df, simulation_threshold=None):
    """Denominators contain only valid data for each specific indicator; no observed outcomes."""
    valid_pd = df['_pd'].notna()
    paired = valid_pd & df['_threshold'].notna()
    count = int(valid_pd.sum())
    paired_count = int(paired.sum())
    decisions = df['decision'].map(_optional_text)
    exceeded = int((df.loc[paired, '_pd'] >= df.loc[paired, '_threshold']).sum())
    result = dict(total=len(df), valid_pd=count, invalid_pd=len(df)-count,
                  mean_pd=float(df.loc[valid_pd, '_pd'].mean()) if count else None,
                  historical_comparable=paired_count, historical_above=exceeded,
                  historical_above_rate=exceeded/paired_count if paired_count else None,
                  recorded_decisions=int(decisions.notna().sum()),
                  decision_counts=decisions.dropna().value_counts().to_dict(),
                  unique_clients=int(df.loc[df['_client'] != '', '_client'].nunique()),
                  undated=int(df['_timestamp'].isna().sum()),
                  missing_client=int((df['_client'] == '').sum()))
    if simulation_threshold is not None:
        threshold = valid_number(simulation_threshold)
        if threshold is None:
            raise ValueError('Seuil de simulation invalide.')
        above = int((df.loc[valid_pd, '_pd'] >= threshold).sum())
        result.update(simulation_threshold=threshold, simulation_above=above,
                      simulation_rate=above/count if count else None)
    return result


def record_context(row):
    names = ('client_id', 'timestamp', 'threshold', 'inputs_json', 'model_tag',
             'rating', 'score_1000', 'decision', 'risk_level', 'model_id', 'input_id')
    ctx = normalize_report_context({**{name: row.get(name) for name in names},
                                    'pd_value': row.get('prob_default')})
    if row.get('_history_warning'):
        ctx['data_notes'].append(str(row['_history_warning']))
    return ctx


def record_display(row):
    """An invalid PD blocks report generation, but does not hide other recorded fields."""
    try:
        ctx = record_context(row)
        return report_display_values(ctx), ctx, None
    except (ValueError, TypeError, OverflowError) as error:
        text = lambda value: _optional_text(value) or 'Non renseigne'
        probability = valid_number(row.get('prob_default'))
        threshold = valid_number(row.get('threshold'))
        score = valid_number(row.get('score_1000'), 1000)
        rating = _optional_text(row.get('rating'))
        if rating and rating.upper() not in {'AAA','AA','A','BBB','BB','B','CCC','CC','C','D'}:
            rating = None
        return dict(pd_value=f'{probability:.2%}' if probability is not None else 'Indisponible',
                    score_1000=str(int(score)) if score is not None and score.is_integer() else 'Non renseigne',
                    rating=text(rating), risk_level=text(row.get('risk_level')),
                    decision=text(row.get('decision')),
                    threshold=f'{threshold:.2%}' if threshold is not None else 'Non renseigne'), None, str(error)


def timeline_figure(df, simulation_threshold=None):
    fig = go.Figure()
    dated = df[df['_timestamp'].notna()].sort_values(['_timestamp','_record_number'], kind='stable')
    for client, group in dated.groupby('_client', sort=False):
        label = client or 'Client non renseigne'
        for column, suffix, dash in (('_pd','PD predite','solid'), ('_threshold','Seuil historique','dot')):
            fig.add_trace(go.Scatter(x=group['_timestamp'].tolist(),
                y=[None if pd.isna(value) else float(value) for value in group[column]],
                mode='lines+markers', name=f'{label} : {suffix}', connectgaps=False,
                line=dict(dash=dash), hovertemplate='%{x}<br>%{y:.2%}<extra>%{fullData.name}</extra>'))
    if simulation_threshold is not None:
        fig.add_hline(y=simulation_threshold, line_dash='dash', line_color='orange',
                      annotation_text='Simulation uniquement')
    fig.update_layout(title='PD predites et seuils historiques par client',
        xaxis_title='Date du dossier (UTC)', yaxis_title='PD / seuil',
        yaxis=dict(tickformat='.0%', range=[0,1]), height=360)
    return fig


def distribution_figure(df, simulation_threshold=None):
    fig = go.Figure(go.Histogram(x=df.loc[df['_pd'].notna(), '_pd'].tolist(),
        xbins=dict(start=0,end=1,size=.05), name='PD predites valides'))
    if simulation_threshold is not None:
        fig.add_vline(x=simulation_threshold, line_dash='dash', line_color='orange',
                      annotation_text='Seuil de simulation uniquement')
    fig.update_layout(title='Distribution des PD predites',
        xaxis=dict(title='PD predite', tickformat='.0%', range=[0,1]),
        yaxis=dict(title='Nombre de predictions', dtick=1), height=360)
    return fig


def export_history_csv(df):
    columns = [column for column in df if not column.startswith('_')]
    return df[columns].to_csv(index=False).encode('utf-8')
