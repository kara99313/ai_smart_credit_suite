"""Read-only operational journals; availability and coverage are explicit."""
from __future__ import annotations
from pathlib import Path
import streamlit as st
from utils.runtime_config import ROOT
from utils import audit
from utils.observability_data import prediction_snapshot, csv_snapshot, agent_statistics, safe_frame

PRED_LOG = ROOT / 'data/predictions_log.csv'
AGENT_LOG = ROOT / 'data/agent_usage.csv'


def show_source(snapshot):
    source = snapshot['source']
    labels = {'missing': 'Absent', 'unavailable': 'Inaccessible', 'invalid': 'Invalide',
              'empty': 'Vide', 'available': 'Lisible', 'present': 'Present'}
    st.caption('Etat du fichier : ' + labels[source['state']] + '. Ce statut ne certifie pas un service.')
    if source['size'] is not None:
        st.caption(f"Taille observee : {source['size']} octets. Modification du fichier (UTC) : {source['modified']}.")
    for warning in snapshot['warnings']:
        st.warning(warning)
    if source['state'] in {'invalid', 'unavailable'}:
        st.error('Journal indisponible ; aucune metrique calculee pour cette source.')
    elif source['state'] == 'missing':
        st.info('Journal absent ; aucune observation disponible.')
    return source['state'] in {'available', 'empty'}


def percentage(value):
    return f'{value:.1%}' if value is not None else 'Indisponible'


def main():
    st.title('Observabilite et journaux')
    st.caption('Lecture seule, sans correction des fichiers ni appel externe. Actualiser relit les sources ; aucune absence de mesure ne vaut zero ou succes.')
    st.button('Actualiser les journaux')  # Streamlit reruns and reloads all snapshots.
    st.subheader('Journal des predictions archivees')
    pred = prediction_snapshot(PRED_LOG)
    if show_source(pred):
        k = pred['kpis']
        if k:
            cols = st.columns(3)
            cols[0].metric('Predictions archivees', str(k['total']))
            cols[1].metric('PD predite moyenne', percentage(k['mean_pd']))
            cols[2].metric('Decisions ACCEPT archivees', percentage(k['acceptance_rate']))
            st.caption(f"PD valides : {k['valid_pd']} / {k['total']}. Decisions reconnues : {k['recognized_decisions']} / {k['total']}. Dates absentes ou invalides : {k['undated']}. Aucun resultat reel de remboursement n est observe ici.")
            daily = pred['daily']
            if not daily.empty:
                st.line_chart(daily.set_index('date')[['volume', 'mean_pd']])
            with st.expander('Derniers enregistrements de predictions'):
                st.dataframe(safe_frame(pred['frame'].drop(columns=[c for c in pred['frame'] if c.startswith('_')]),200), hide_index=True)

    st.caption('Les graphiques regroupent les dates en UTC ; une date historique sans fuseau est interpretee UTC, sans preuve de son fuseau d origine.')
    st.subheader('Journal d usage des outils agents')
    st.caption('Fichier optionnel agent_usage.csv. Les parcours agents actuels ne produisent pas ce journal ; sa presence ne prouve pas une instrumentation active ni une couverture exhaustive.')
    agents = csv_snapshot(AGENT_LOG)
    if show_source(agents):
        k = agent_statistics(agents['frame'])
        cols = st.columns(3)
        cols[0].metric('Lignes d usage archivees', str(k['total']))
        cols[1].metric('Succes declares dans le journal', percentage(k['success_rate']))
        cols[2].metric('Outil le plus archive', k['most_used'] or 'Indisponible')
        st.caption(f"Succes renseignes (true/false/1/0) : {k['known_success']} / {k['total']}. Inconnus : {k['unknown_success']}. Dates absentes ou invalides : {k['undated']}.")
        if not k['daily'].empty:
            st.bar_chart(k['daily'])
        with st.expander('Enregistrements d usage'):
            st.caption('Apercu des 300 dernieres lignes du fichier, ordre inverse ; valeurs sensibles masquees.')
            st.dataframe(safe_frame(agents['frame'].tail(300).iloc[::-1]), hide_index=True)

    st.subheader('Journal d audit de configuration')
    st.caption('Evenements locaux des actions Administration. L acteur session_non_authentifiee ne constitue pas une identite verifiee. Journal non signe ; aucune garantie d exhaustivite ou d inviolabilite.')
    events = csv_snapshot(audit.AUDIT_PATH, required_columns=audit.COLUMNS)
    if show_source(events):
        st.metric('Lignes d audit archivees', str(len(events['frame'])))
        if not events['frame'].empty:
            st.dataframe(safe_frame(events['frame'].tail(300).iloc[::-1]), hide_index=True)

    st.subheader('Journaux applicatifs et etat des services')
    st.info('Aucun collecteur de logs du serveur ou de Streamlit n est raccorde a cette page. Aucun statut de service, taux d erreur global, latence, cout ou consommation de tokens ne peut etre deduit de ces fichiers.')
    if st.button('Ouvrir les diagnostics explicites des services'):
        st.switch_page('app/health_check.py')
    if st.button('Ouvrir Administration'):
        st.switch_page('app/admin.py')


if __name__ == '__main__':
    main()
