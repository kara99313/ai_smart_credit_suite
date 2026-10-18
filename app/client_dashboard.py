"""Dashboard of recorded scoring dossiers; simulations never replace history."""
from pathlib import Path
import streamlit as st
import pandas as pd
from utils.client_dashboard_data import (load_history, filter_history, calculate_kpis,
    record_display, timeline_figure, distribution_figure, export_history_csv, valid_number)
from utils.report_generator import render_html_report
from utils.reporting_pdf import build_pdf_bytes
from utils.settings import get_settings

LOG_PATH = 'data/predictions_log.csv'


def load_logs():
    return load_history(LOG_PATH)


def _rate(value):
    return 'Indisponible' if value is None else f'{value:.1%}'


def main():
    st.title('Dashboard Client / Historique')
    st.caption('Résultats archivés et simulations distinctes. Aucun défaut observé n’est disponible dans ce journal.')
    try:
        df = load_logs()
    except (ValueError, OSError) as error:
        st.error(f'Lecture du journal impossible : {error}')
        return
    st.caption(f'Historique disponible : {len(df)} dossiers')
    if df.empty:
        st.info('Aucune prédiction enregistrée.')
        return
    for message in df.get('_history_warning', []).unique() if '_history_warning' in df else []:
        if isinstance(message, str):
            st.warning(message)
    clients = [None] + sorted(df['_client'].unique().tolist())
    client = st.selectbox('Client', clients,
        format_func=lambda value: 'Tous les clients' if value is None else value or 'Client non renseigné')
    start = end = None
    include_undated = True
    if st.checkbox('Filtrer par dates (UTC)', value=False):
        dated = df['_timestamp'].dropna()
        if dated.empty:
            st.info('Aucune date valide : les dossiers restent disponibles.')
        else:
            left,right = st.columns(2)
            start = left.date_input('Du', value=dated.min().date())
            end = right.date_input('Au', value=dated.max().date())
            include_undated = st.checkbox('Inclure les dossiers sans date', value=True)
            if start > end:
                st.warning('La date de début doit précéder la date de fin.')
                return
    filtered = filter_history(df, client, start, end, include_undated)
    if filtered.empty:
        st.info('Aucun dossier pour ces filtres.')
        return
    kpis = calculate_kpis(filtered)
    st.subheader('Indicateurs des dossiers filtrés')
    fields = [('Dossiers', str(kpis['total'])), ('PD valides', str(kpis['valid_pd'])),
              ('PD prédite moyenne', _rate(kpis['mean_pd'])),
              ('PD ≥ seuil historique', _rate(kpis['historical_above_rate'])),
              ('Décisions enregistrées', str(kpis['recorded_decisions'])),
              ('Clients identifiés', str(kpis['unique_clients']))]
    for column,(title,value) in zip(st.columns(6), fields):
        column.metric(title,value)
    st.caption(f"PD moyenne : moyenne des {kpis['valid_pd']} PD valides. "
               f"Dépassement historique : {kpis['historical_above']} / {kpis['historical_comparable']} dossiers avec PD et seuil valides, chaque dossier à son propre seuil. "
               'Ce ratio est distinct de la décision enregistrée et du défaut observé.')
    st.caption(f"Données indisponibles : {kpis['invalid_pd']} PD, {kpis['undated']} dates, {kpis['missing_client']} identifiants client. "
               'Ces lignes restent dans les dossiers et le CSV filtré.')
    if kpis['decision_counts']:
        st.write('Décisions historiques (libellés conservés) :', kpis['decision_counts'])
    st.subheader('Dossier sélectionné')
    st.caption('Sélection initiale : date connue la plus récente ; à date identique, dernière occurrence du journal. '
               'La chronologie des dossiers sans date est indéterminée.')
    choices = filtered.index.tolist()
    def label(index):
        row = filtered.loc[index]
        date = str(row['timestamp']) if row['_timestamp'] is not None and not pd.isna(row['_timestamp']) else 'Date indisponible'
        return f"{row['_client'] or 'Client non renseigne'} / {date} / dossier {row['_record_number']}"
    selected = st.selectbox('Dossier / occurrence', choices, format_func=label)
    values,ctx,error = record_display(filtered.loc[selected].to_dict())
    fields = [('pd_value','PD'), ('score_1000','Score (0-1000)'), ('rating','Notation'),
              ('risk_level','Niveau de risque'), ('decision','Décision historique'), ('threshold','Seuil historique')]
    for column,(key,title) in zip(st.columns(6), fields):
        column.metric(title,values[key])
    if error:
        st.warning(f'Dossier incomplet ou invalide : {error}')
    if ctx:
        for note in ctx['data_notes']:
            st.caption(note)
    simulation_threshold = None
    if st.checkbox('Activer une simulation de seuil', value=False):
        initial = valid_number(filtered.loc[selected].get('threshold'))
        if initial is None:
            initial = get_settings().decision_threshold
        simulation_threshold = st.slider('Seuil simulé (aucune modification des décisions)',
                                         0.0, 1.0, float(initial), .01)
        simulated = calculate_kpis(filtered, simulation_threshold)
        st.metric('Simulation : proportion de PD ≥ seuil simulé', _rate(simulated['simulation_rate']))
        st.caption(f"Simulation : {simulated['simulation_above']} / {simulated['valid_pd']} PD valides. Aucune décision de crédit n'est recalculée.")
    st.subheader('Graphiques des dossiers filtrés')
    st.caption('Une série par client, dates en UTC ; PD manquantes interrompues. Dates invalides exclues de la chronologie uniquement. '
               'La distribution utilise seulement les PD valides. Les seuils pointillés sont historiques ; orange = simulation.')
    left,right = st.columns(2)
    left.plotly_chart(timeline_figure(filtered,simulation_threshold),use_container_width=True)
    right.plotly_chart(distribution_figure(filtered,simulation_threshold),use_container_width=True)
    st.subheader('Historique filtré : valeurs archivées')
    columns = [name for name in ('timestamp','client_id','prob_default','score_1000','rating','risk_level','decision','threshold','model_tag') if name in filtered]
    st.dataframe(filtered[columns].astype('string').fillna('Non renseigné'),hide_index=True,width='stretch')
    st.download_button('Télécharger CSV historique filtré',export_history_csv(filtered),
                       file_name='client_dashboard_export.csv',mime='text/csv')
    st.subheader('Exporter le dossier sélectionné')
    st.caption('Mêmes valeurs que les cartes du dossier. La simulation et les agrégats ne remplacent pas le résultat historique dans le rapport.')
    if ctx is None:
        st.info('Export HTML/PDF indisponible pour ce dossier invalide ; sa ligne reste exportable en CSV.')
    else:
        cfg = get_settings()
        ctx.update(primary_color=cfg.report_primary_color,footer_text=cfg.report_footer_text)
        logo = cfg.logo_path if (cfg.logo_path and Path(cfg.logo_path).is_file()) else None
        try:
            html = render_html_report(ctx).encode('utf-8')
            pdf = build_pdf_bytes(ctx,df_exp=None,logo_path=logo)
        except (ValueError,TypeError,OSError) as error:
            st.error(f'Export impossible : {error}')
            return
        name = int(filtered.loc[selected,'_record_number'])
        st.download_button('Télécharger HTML du dossier',html,file_name=f'dossier_{name}.html',mime='text/html')
        st.download_button('Télécharger PDF du dossier',pdf,file_name=f'dossier_{name}.pdf',mime='application/pdf')


if __name__ == '__main__':
    main()
