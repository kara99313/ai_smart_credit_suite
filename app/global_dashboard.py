"""Global dashboard: recorded portfolio, explicitly scoped indicators and simulation."""
import pandas as pd
import streamlit as st
from utils.client_dashboard_data import load_history, distribution_figure, export_history_csv
from utils.global_dashboard_data import (dimension_values, filter_portfolio, portfolio_kpis,
    daily_figures, score_figure, category_figure, threshold_figure, recorded_scores)
from utils.settings import get_settings

LOG_PATH = 'data/predictions_log.csv'


def load_logs(path):
    # No path-only cache: appends to the CSV or JSONL must be visible on rerun.
    return load_history(path)


def rate(value):
    return 'Indisponible' if value is None else f'{value:.2%}'


def main():
    st.set_page_config(layout='wide',page_title='Dashboard Global')
    st.title('Dashboard Global / Portefeuille')
    st.caption('Dossiers de scoring archiv\u00e9s ; une pr\u00e9diction de d\u00e9faut ne constitue pas un d\u00e9faut observ\u00e9.')
    try:
        df = load_logs(LOG_PATH)
    except (ValueError,OSError) as error:
        st.error(f'Lecture du journal impossible : {error}')
        return
    st.caption(f'Historique disponible : {len(df)} dossiers')
    if df.empty:
        st.info('Aucun dossier enregistr\u00e9.')
        return
    if '_history_warning' in df:
        for warning in df['_history_warning'].dropna().unique():
            st.warning(str(warning))
    st.subheader('Filtres communs aux indicateurs, graphiques et export')
    selections = {}
    for column,label in (('_client','Clients'),('rating','Ratings'),('decision','D\u00e9cisions'),('model_tag','Mod\u00e8les')):
        values = df['_client'].map(lambda value:value or None) if column == '_client' else dimension_values(df,column)
        known = sorted(value for value in values.dropna().unique())
        options = known + ([None] if values.isna().any() else [])
        selections[column] = st.multiselect(label,options,format_func=lambda value: value if value is not None else 'Non renseign\u00e9')
    st.caption('Une s\u00e9lection vide conserve toutes les valeurs, y compris les donn\u00e9es manquantes.')
    start = end = None
    dates_enabled = st.checkbox('Filtrer par dates (UTC)',value=False)
    if dates_enabled:
        dated = df['_timestamp'].dropna()
        if dated.empty:
            st.info('Aucune date valide dans le journal.')
        else:
            left,right = st.columns(2)
            start = left.date_input('Du',dated.min().date())
            end = right.date_input('Au',dated.max().date())
            if start > end:
                st.warning('La date de d\u00e9but doit pr\u00e9c\u00e9der la date de fin.')
                return
    recent = st.checkbox('Derniers 30 jours (UTC)',value=False)
    include_undated = st.checkbox('Inclure les dossiers sans date',value=True) if dates_enabled or recent else True
    filtered = filter_portfolio(df,clients=selections['_client'],ratings=selections['rating'],
        decisions=selections['decision'],models=selections['model_tag'],
        start=start,end=end,include_undated=include_undated,last_30_days=recent)
    st.caption(f'P\u00e9rim\u00e8tre filtr\u00e9 : {len(filtered)} / {len(df)} dossiers')
    kpis = portfolio_kpis(filtered)
    st.subheader('Indicateurs du portefeuille filtr\u00e9')
    fields = [('Dossiers',str(kpis['total'])),('Clients identifi\u00e9s',str(kpis['unique_clients'])),
              ('PD pr\u00e9dite moyenne',rate(kpis['mean_pd'])),('D\u00e9cisions enregistr\u00e9es',str(kpis['recorded_decisions'])),
              ('Acceptation historique',rate(kpis['acceptance_rate'])),
              ('PD \u2265 seuil historique',rate(kpis['historical_above_rate']))]
    for column,(label,value) in zip(st.columns(6),fields):
        column.metric(label,value)
    st.caption(f"PD moyenne : {kpis['valid_pd']} PD valides / {kpis['total']} dossiers. "
               f"Acceptation historique : {kpis['accepted']} ACCEPT / {kpis['recognized_decisions']} d\u00e9cisions reconnues (ACCEPT, REVIEW/REJECT). "
               f"{kpis['missing_decisions']} d\u00e9cisions manquantes et {kpis['unrecognized_decisions']} autres libell\u00e9s exclus de ce ratio ; ces libell\u00e9s sont conserv\u00e9s dans le graphique et l'export.")
    st.caption(f"D\u00e9passement historique : {kpis['historical_above']} / {kpis['historical_comparable']} dossiers avec PD et seuil valides, chaque dossier \u00e0 son seuil archiv\u00e9. "
               "Ce ratio ne recalcule pas la d\u00e9cision historique et ne mesure pas un d\u00e9faut observ\u00e9.")
    st.caption(f"Donn\u00e9es indisponibles : {kpis['invalid_pd']} PD, {kpis['undated']} dates, {kpis['missing_client']} identifiants client. Aucune ligne n'est supprim\u00e9e hors filtres explicites.")
    st.info("AUC, accuracy, recall et autres performances op\u00e9rationnelles indisponibles : aucune vraie classe de d\u00e9faut observ\u00e9e et appari\u00e9e aux pr\u00e9dictions n'est fournie par ce journal. pred_label n'est pas une v\u00e9rit\u00e9 terrain.")
    simulation_threshold = None
    if st.checkbox('Activer une simulation de seuil',value=False):
        simulation_threshold = st.slider('Seuil simul\u00e9 (aucune modification historique)',
            0.0,1.0,float(get_settings().decision_threshold),.01)
        simulated = portfolio_kpis(filtered,simulation_threshold)
        st.metric('Simulation : PD \u2265 seuil simul\u00e9',rate(simulated['simulation_rate']))
        st.caption(f"Simulation uniquement : {simulated['simulation_above']} / {simulated['valid_pd']} PD valides. La valeur initiale vient de la configuration ; aucune acceptation n'est recalcul\u00e9e.")
    if filtered.empty:
        st.info('Aucun dossier pour ces filtres.')
    else:
        st.subheader('Graphiques du m\u00eame p\u00e9rim\u00e8tre filtr\u00e9')
        st.caption(f"Chronologie : {len(filtered)-kpis['undated']} dossiers dat\u00e9s / {len(filtered)} ; jours en UTC. "
                   "La moyenne journali\u00e8re exclut les PD invalides, avec son effectif dans l'infobulle. Le volume compte tous les dossiers dat\u00e9s, m\u00eame sans identifiant.")
        mean,volume = daily_figures(filtered)
        left,right = st.columns(2)
        left.plotly_chart(mean,use_container_width=True)
        right.plotly_chart(volume,use_container_width=True)
        st.caption(f"Distribution des PD : {kpis['valid_pd']} valeurs valides / {len(filtered)} dossiers. Orange = simulation uniquement.")
        left,right = st.columns(2)
        left.plotly_chart(distribution_figure(filtered,simulation_threshold),use_container_width=True)
        right.plotly_chart(threshold_figure(filtered),use_container_width=True)
        st.caption(f"Scores archiv\u00e9s valides : {int(recorded_scores(filtered).notna().sum())} / {len(filtered)} dossiers. Aucun score ni rating n'est reconstitu\u00e9 pour ces distributions.")
        left,right = st.columns(2)
        left.plotly_chart(score_figure(filtered),use_container_width=True)
        right.plotly_chart(category_figure(filtered,'rating','Ratings archives'),use_container_width=True)
        left,right = st.columns(2)
        left.plotly_chart(category_figure(filtered,'decision','Decisions archivees'),use_container_width=True)
        right.plotly_chart(category_figure(filtered,'_client','Dossiers par client'),use_container_width=True)
        st.caption('Les barres de ratings, d\u00e9cisions et clients comptent les dossiers ; les valeurs manquantes sont incluses. Les libell\u00e9s archiv\u00e9s ne sont pas des observations de d\u00e9faut.')
        columns = [name for name in filtered if not name.startswith('_')]
        st.subheader('Dossiers filtr\u00e9s : valeurs archiv\u00e9es')
        st.dataframe(filtered[columns].astype('string').fillna('Non renseign\u00e9'),hide_index=True,width='stretch')
    st.download_button('T\u00e9l\u00e9charger CSV du portefeuille filtr\u00e9',export_history_csv(filtered),
        file_name='global_dashboard_export.csv',mime='text/csv')


if __name__ == '__main__':
    main()
