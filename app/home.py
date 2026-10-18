"""Home uses the same archived operational indicators as the global dashboard."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import streamlit as st
from utils.client_dashboard_data import load_history, distribution_figure
from utils.global_dashboard_data import portfolio_kpis, daily_figures
from utils.runtime_config import backend_url
from utils.api_client import ping_backend

LOG_PATH = Path('data/predictions_log.csv')
SHORTCUTS = (
    ('Pr\u00e9diction', 'app/prediction.py'),
    ('Rapports', 'app/report.py'),
    ('Dashboard Client', 'app/client_dashboard.py'),
    ('Dashboard Global', 'app/global_dashboard.py'),
)


def load_logs(path=None):
    # Re-read CSV and JSONL on each rerun; never cache by pathname alone.
    return load_history(LOG_PATH if path is None else path)


def kpis_from_df(df):
    """Compatibility entry point delegating every aggregate to the common helper."""
    if df.empty:
        required = ['_pd','_threshold','decision','_client','_timestamp']
        df = df.reindex(columns=list(dict.fromkeys([*df.columns,*required])))
    return portfolio_kpis(df)


def rate(value):
    return 'Indisponible' if value is None else f'{value:.2%}'


def check_api(base):
    """A health request checks the HTTP service, not model inference or RAG."""
    ok, response = ping_backend(base=base,timeout=3)
    valid = ok and isinstance(response,dict) and response.get('status') == 'ok'
    declared = response.get('model_loaded') if isinstance(response,dict) else None
    return dict(base=base,checked_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                api='Joignable (/health)' if valid else 'Indisponible',
                model_declared=declared if isinstance(declared,bool) else None,
                detail='R\u00e9ponse /health valide.' if valid else 'Service inaccessible ou r\u00e9ponse /health inattendue.')


def service_panel():
    st.subheader('\u00c9tat des services')
    try:
        base = backend_url()
    except (ValueError,TypeError,OSError):
        base = None
        st.warning('Configuration API invalide ; disponibilit\u00e9 non v\u00e9rifiable.')
    if st.button('V\u00e9rifier FastAPI',disabled=base is None,key='home_check_api'):
        st.session_state['home_api_check'] = check_api(base)
    snapshot = st.session_state.get('home_api_check')
    if snapshot and snapshot.get('base') != base:
        snapshot = None
    st.write('FastAPI :',snapshot['api'] if snapshot else 'Indisponible (non v\u00e9rifi\u00e9)')
    if snapshot:
        st.caption(f"Contr\u00f4le HTTP GET /health \u00e0 {snapshot['checked_at']} (UTC). {snapshot['detail']}")
        if snapshot['model_declared'] is not None:
            st.caption(f"D\u00e9claration du backend : model_loaded={snapshot['model_declared']}. Ce champ ne prouve pas qu'une inf\u00e9rence fonctionne.")
    st.write('Mod\u00e8le de scoring : Indisponible (contr\u00f4le fonctionnel non r\u00e9alis\u00e9 ici)')
    st.write('LLM : Indisponible (appel fonctionnel non r\u00e9alis\u00e9)')
    st.write('RAG : Indisponible (recherche documentaire non test\u00e9e)')
    st.caption("Ces statuts portent sur les contr\u00f4les r\u00e9alis\u00e9s dans cette page, pas sur l'existence de fichiers. "
               "Aucun appel LLM, chargement de mod\u00e8le ou construction d'index n'est lanc\u00e9 \u00e0 l'ouverture.")


def navigation_panel():
    st.subheader('Acc\u00e8s aux parcours valid\u00e9s')
    for column,(label,path) in zip(st.columns(4),SHORTCUTS):
        if column.button(f'Ouvrir {label}',key=f'home_nav_{Path(path).stem}'):
            try:
                st.switch_page(path)
            except st.errors.StreamlitAPIException:
                st.error('Ouvrez streamlit_app.py pour utiliser la navigation entre les onglets.')
    with st.expander('Autres composants : validation fonctionnelle restante'):
        st.write('Agent IA, Chatbot Assistant, Chatbot RAG, Observabilit\u00e9, Int\u00e9grations et Administration restent accessibles depuis le menu principal.')
        st.warning("Leur fonctionnement de bout en bout n'est pas valid\u00e9 par les parcours de scoring. "
                   "Les badges LLM indiquent la configuration uniquement ; "
                   "ils ne prouvent pas une r\u00e9ponse du service. Le parcours RAG reste \u00e0 auditer.")
        st.write("Le Health Check fournit un diagnostic explicite ; il ne certifie pas les parcours LLM/RAG de bout en bout.")
        st.caption("L'Accueil ne les annonce pas op\u00e9rationnels et ne d\u00e9clenche pas leurs traitements.")


def main():
    st.set_page_config(layout='wide',page_title='Accueil / Smart Credit Scoring')
    st.title('Accueil / Smart Credit Scoring')
    st.caption('Synth\u00e8se des dossiers archiv\u00e9s, navigation et contr\u00f4les explicites.')
    with st.sidebar:
        st.info('Utilisez les raccourcis de cette page ou le menu principal pour changer de parcours.')
    st.markdown("La pr\u00e9diction fournit une PD, un score et une d\u00e9cision. "
                "L'explicabilit\u00e9 disponible pr\u00e9sente des importances globales du mod\u00e8le ; "
                "elle ne constitue pas une contribution individuelle SHAP/LIME. "
                "Les rapports et dashboards conservent les r\u00e9sultats archiv\u00e9s.")
    navigation_panel()
    service_panel()
    st.subheader('Synth\u00e8se du portefeuille archiv\u00e9')
    try:
        df = load_logs()
    except (ValueError,OSError) as error:
        st.error(f'Lecture du journal impossible : {error}')
        return
    if '_history_warning' in df:
        for warning in df['_history_warning'].dropna().unique():
            st.warning(str(warning))
    k = kpis_from_df(df)
    st.caption(f"Historique disponible : {len(df)} dossiers. Source et formules identiques au Dashboard Global.")
    fields = [('Dossiers archiv\u00e9s',str(k['total'])),('Clients identifi\u00e9s',str(k['unique_clients'])),
              ('PD pr\u00e9dite moyenne',rate(k['mean_pd'])),('D\u00e9cisions enregistr\u00e9es',str(k['recorded_decisions'])),
              ('Acceptation historique',rate(k['acceptance_rate'])),('AUC op\u00e9rationnelle','Indisponible')]
    for column,(label,value) in zip(st.columns(6),fields):
        column.metric(label,value)
    st.caption(f"PD moyenne : {k['valid_pd']} PD valides / {k['total']} dossiers. "
               f"Acceptation historique : {k['accepted']} ACCEPT / {k['recognized_decisions']} d\u00e9cisions reconnues (ACCEPT ou REVIEW/REJECT). "
               f"{k['missing_decisions']} d\u00e9cisions manquantes et {k['unrecognized_decisions']} autres libell\u00e9s exclus de ce ratio.")
    st.caption(f"Donn\u00e9es indisponibles : {k['invalid_pd']} PD, {k['undated']} dates, {k['missing_client']} identifiants. "
               "Les lignes incompl\u00e8tes restent dans les comptages de dossiers.")
    st.info("AUC et taux de d\u00e9faut observ\u00e9 : Indisponible. Aucune vraie classe observ\u00e9e et appari\u00e9e aux pr\u00e9dictions n'est disponible ici. "
            "pred_label n'est pas une observation de d\u00e9faut. Aucune performance historique de validation n'est affich\u00e9e comme performance en temps r\u00e9el.")
    if df.empty:
        st.info('Aucun dossier enregistr\u00e9.')
        return
    st.subheader('Distributions et volumes')
    left,right = st.columns(2)
    left.plotly_chart(distribution_figure(df),use_container_width=True,config={'displayModeBar':False})
    _,volume = daily_figures(df)
    right.plotly_chart(volume,use_container_width=True,config={'displayModeBar':False})
    st.caption(f"Distribution : {k['valid_pd']} PD valides / {len(df)} dossiers, axe en pourcentage. "
               f"Volume quotidien : {len(df)-k['undated']} dossiers dat\u00e9s / {len(df)}, jours UTC, identifiants manquants inclus. "
               "Aucun seuil de d\u00e9cision ou de simulation n'est appliqu\u00e9 sur l'Accueil.")
    st.caption('Les filtres et simulations sont disponibles dans les dashboards ; les exports se trouvent dans Rapports et les dashboards.')


if __name__ == '__main__':
    main()
