"""Portfolio aggregation of the shared operational history; no observed outcomes."""
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from utils.client_dashboard_data import calculate_kpis, filter_history, valid_number
from utils.report_generator import _optional_text


def dimension_values(df, column):
    if column not in df:
        return pd.Series(None, index=df.index, dtype=object)
    return df[column].map(_optional_text)


def filter_portfolio(df, *, clients=None, ratings=None, decisions=None, models=None,
                     start=None, end=None, include_undated=True, last_30_days=False, now=None):
    result = filter_history(df, start=start, end=end, include_undated=include_undated)
    for column, selected in (('_client', clients), ('rating', ratings),
                             ('decision', decisions), ('model_tag', models)):
        if selected:
            values = result['_client'].map(lambda value: value or None) if column == '_client' else dimension_values(result, column)
            result = result[values.isin(selected)]
    if last_30_days:
        now = pd.Timestamp.now(tz='UTC') if now is None else pd.Timestamp(now)
        now = now.tz_localize('UTC') if now.tzinfo is None else now.tz_convert('UTC')
        dated = result['_timestamp']
        mask = (dated >= now-pd.Timedelta(days=30)) & (dated <= now)
        if include_undated:
            mask |= dated.isna()
        result = result[mask]
    return result


def portfolio_kpis(df, simulation_threshold=None):
    result = calculate_kpis(df, simulation_threshold)
    decisions = dimension_values(df, 'decision')
    # Only the two documented API outcomes have unambiguous operational meaning.
    recognized = decisions.isin(['ACCEPT', 'REVIEW/REJECT'])
    accepted = int((decisions == 'ACCEPT').sum())
    count = int(recognized.sum())
    result.update(accepted=accepted, recognized_decisions=count,
                  unrecognized_decisions=int(decisions.notna().sum())-count,
                  missing_decisions=int(decisions.isna().sum()),
                  acceptance_rate=accepted/count if count else None)
    return result


def daily_summary(df):
    dated = df[df['_timestamp'].notna()].copy()
    if dated.empty:
        return pd.DataFrame(columns=['date','volume','valid_pd','mean_pd'])
    dated['date'] = dated['_timestamp'].dt.floor('D')
    return dated.groupby('date', as_index=False, sort=True).agg(
        volume=('_record_number', 'size'), valid_pd=('_pd', 'count'), mean_pd=('_pd', 'mean'))


def daily_figures(df):
    daily = daily_summary(df)
    mean = go.Figure(go.Scatter(x=daily['date'].tolist(),
        y=[None if pd.isna(value) else float(value) for value in daily['mean_pd']],
        mode='lines+markers', connectgaps=False, name='PD predite moyenne',
        customdata=daily[['valid_pd','volume']].values.tolist(),
        hovertemplate='%{x}<br>PD moyenne : %{y:.2%}<br>PD valides : %{customdata[0]} / %{customdata[1]} dossiers<extra></extra>'))
    mean.update_layout(title='PD predite moyenne par jour (PD valides uniquement)',
        xaxis_title='Jour (UTC)', yaxis=dict(title='PD predite moyenne',tickformat='.0%',range=[0,1]),height=340)
    volume = go.Figure(go.Bar(x=daily['date'].tolist(),y=daily['volume'].tolist(),name='Dossiers dates'))
    volume.update_layout(title='Volume de dossiers par jour',xaxis_title='Jour (UTC)',
                         yaxis=dict(title='Nombre de dossiers',dtick=1),height=340)
    return mean,volume


def recorded_scores(df):
    values = df['score_1000'] if 'score_1000' in df else pd.Series(None,index=df.index,dtype=object)
    def score(value):
        number = valid_number(value,1000)
        return number if number is not None and number.is_integer() else None
    return values.map(score)


def score_figure(df):
    values = recorded_scores(df).dropna().tolist()
    fig = go.Figure(go.Histogram(x=values,xbins=dict(start=0,end=1000,size=50),name='Scores archives valides'))
    fig.update_layout(title=f'Scores archives valides (n={len(values)} / {len(df)})',
        xaxis=dict(title='Score archive (0-1000)',range=[0,1000]),yaxis=dict(title='Nombre de dossiers',dtick=1),height=340)
    return fig


def category_figure(df, column, title):
    values = dimension_values(df,column) if column != '_client' else df['_client'].map(lambda value:value or None)
    known = int(values.notna().sum())
    labels = values.fillna('Non renseigne')
    counts = labels.value_counts(sort=False).sort_index()
    fig = go.Figure(go.Bar(x=counts.index.tolist(),y=counts.tolist(),name='Valeurs archivees'))
    fig.update_layout(title=f'{title} ({known} renseignes / {len(df)} dossiers)',
        xaxis_title='Valeur archivee ; manquants inclus',yaxis=dict(title='Nombre de dossiers',dtick=1),height=340)
    return fig


def threshold_figure(df):
    values = df['_threshold'].dropna().tolist()
    fig = go.Figure(go.Histogram(x=values,xbins=dict(start=0,end=1,size=.05),name='Seuils historiques valides'))
    fig.update_layout(title=f'Seuils historiques valides (n={len(values)} / {len(df)})',
        xaxis=dict(title='Seuil historique',tickformat='.0%',range=[0,1]),
        yaxis=dict(title='Nombre de dossiers',dtick=1),height=340)
    return fig
