"""Admin-only technical telemetry dashboard."""
from pathlib import Path
import pandas as pd
import streamlit as st
from utils.admin_auth import render_gate
from utils.observability_telemetry import prediction_snapshot, agent_snapshot

ROOT = Path(__file__).resolve().parents[1]
PRED_LOG = ROOT / 'logs/prediction_telemetry.sqlite3'
AGENT_LOG = ROOT / 'logs/agent_telemetry.sqlite3'


def show_source(snapshot):
    state = snapshot['source']['state']
    if state == 'missing':
        st.info('Source absente ; aucune observation disponible.')
    elif state == 'unavailable':
        st.error('Source indisponible ou incompatible ; aucune métrique calculée.')
    elif state == 'empty':
        st.caption('Aucune observation dans la période sélectionnée.')
    else:
        st.caption('Source lisible ; ce statut ne certifie pas un service.')
    return state in {'available', 'empty'}


def metrics(k, fields):
    columns = st.columns(3)
    for i, (key, label) in enumerate(fields):
        value = k[key]
        if value is None:
            value = 'Indisponible'
        elif key == 'success_rate':
            value = f'{value:.1%}'
        elif key in ('mean_ms', 'median_ms'):
            value = f'{value:.1f} ms'
        else:
            value = str(value)
        columns[i % 3].metric(label, value)


def daily_chart(rows):
    if rows:
        frame = pd.DataFrame(rows).pivot(index='date', columns='status', values='calls').fillna(0)
        st.bar_chart(frame)


def main():
    if not render_gate():
        return
    st.title('Observabilité technique')
    period = st.selectbox('Période', ['24 heures', '7 jours', '30 jours'], index=2)
    st.button('Actualiser les journaux')
    hours = {'24 heures': 24, '7 jours': 168, '30 jours': 720}[period]
    origin = st.selectbox('Origine API / préchauffages', ['Tous', 'business', 'synthetic'])
    pred = prediction_snapshot(PRED_LOG, hours=hours, origin=None if origin == 'Tous' else origin)
    agent = agent_snapshot(AGENT_LOG, hours=hours)
    st.subheader('État des sources')
    st.caption('Prediction — SQLite technique 4B')
    pred_ok = show_source(pred)
    st.caption('Agent — SQLite technique 4C')
    agent_ok = show_source(agent)
    st.subheader('Prédictions / API')
    st.caption('business est une origine déclarative, sans preuve de client réel. Les KPI principaux couvrent uniquement api_predict.')
    if pred_ok and pred['kpis'] is not None:
        metrics(pred['kpis'], [('total', 'Tentatives API'), ('success', 'Succès API'), ('validation_error', 'validation_error'),
                ('internal_error', 'internal_error'), ('success_rate', 'Taux de succès API'), ('mean_ms', 'Latence moyenne API'), ('median_ms', 'Latence médiane API')])
        st.caption('Taux : succès / tentatives API. Latences : toutes les tentatives API terminées, en millisecondes.')
        st.metric('Préchauffages (hors KPI API)', str(pred['warmups']['total']))
        daily_chart(pred['daily'])
        if pred['models']:
            st.dataframe(pd.DataFrame(pred['models']), hide_index=True)
    st.subheader('Agent')
    if agent_ok and agent['kpis'] is not None:
        metrics(agent['kpis'], [('total', 'Exécutions'), ('success', 'Succès Agent'), ('error', 'Erreurs Agent'),
                ('running', 'Non terminées'), ('success_rate', 'Taux de succès Agent'), ('mean_ms', 'Latence moyenne Agent'),
                ('median_ms', 'Latence médiane Agent'), ('attempts', 'Tentatives Agent'),
                ('retry_decided', 'Tentatives au statut retry'), ('retries_started', 'Retries effectivement démarrés')])
        st.caption('Taux : succès / (succès + erreurs). Latences : exécutions terminales seulement. Retry démarré : attempt > 1 ; retry décidé : statut retry.')
        daily_chart(agent['daily'])
        if agent['models']:
            st.dataframe(pd.DataFrame(agent['models']), hide_index=True)
    st.subheader('Outils')
    if agent_ok and agent['tools'] is not None:
        metrics(agent['tools'], [('total', 'Outils démarrés'), ('terminal', 'Outils terminés'), ('error', 'Outils en erreur'),
                                ('interrupted', 'interrupted'), ('running', 'Outils non terminés')])
        st.caption('Terminés : success, error, no_result, unknown. Durées moyennes et médianes : ces mêmes appels terminés uniquement.')
        if agent['tool_groups']:
            frame = pd.DataFrame(agent['tool_groups'])
            st.dataframe(frame, hide_index=True)
            st.bar_chart(frame.set_index('tool_name')[['success', 'error', 'no_result', 'unknown', 'interrupted', 'running']])
    st.subheader('Limites de mesure')
    st.caption('Dates et regroupements en UTC ; fenêtre appliquée au début des appels, tentatives et outils indépendamment. running signifie sans terminaison observée, y compris après un arrêt brutal. Écritures best effort : couverture non exhaustive. Rétention de 30 jours côté producteurs ; cette page lit sans écrire ni purger. Aucun statut global des services ne peut être déduit.')
    st.caption('rag_retrieve, search_web et build_report sont des appels techniques. build_report ne confirme aucun téléchargement. Aucun contenu métier ou LLM n’est affiché.')
    if st.button('Ouvrir les diagnostics explicites des services'):
        st.switch_page('app/health_check.py')
    if st.button('Ouvrir Administration'):
        st.switch_page('app/admin.py')


if __name__ == '__main__':
    main()
