from synthetic_history import SyntheticHistoryTestCase
"""Client dashboard data integrity, indicators, charts, selection and exports."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest
from utils.prediction_logs import COLUMNS, append_prediction_record
from utils.client_dashboard_data import (load_history,filter_history,calculate_kpis,
    record_context,record_display,timeline_figure,distribution_figure,export_history_csv)
from utils.report_generator import render_html_report,report_display_values
from utils.reporting_pdf import build_pdf_bytes
from test_reports import row,write_csv,pdf_text,BASELINE_HASH,LOG,ROOT


def fixtures(path):
    records = [
        dict(client_id='A',timestamp='2026-01-01',prob_default=.06,threshold=.1,decision='REFUS-HIST',score_1000=937),
        dict(client_id='A',timestamp='2026-02-01',prob_default=.9,threshold=.8,decision='ACCEPT',score_1000=456),
        dict(client_id='A',timestamp='',prob_default=None,threshold=.3,decision='REFUS-HIST',score_1000=777),
        dict(client_id='',timestamp='2026-01-03',prob_default=.2,threshold=None,decision='',score_1000=800),
        dict(client_id='B',timestamp='2026-01-04',prob_default=1.2,threshold=.5,decision='',score_1000=10),
        dict(client_id='B',timestamp='2026-03-01',prob_default=0,threshold=0,decision='REVIEW/REJECT',score_1000=1000),
    ]
    for values in records:
        record={**row(),**values}
        append_prediction_record(path,record,dict(score_1000=values['score_1000'],risk_level='SOURCE-RISK',model_id='MODEL-ID',input_id='INPUT-ID'))
    return load_history(path)


def page(path):
    script=('import runpy\n'+f"ns=runpy.run_path({str(ROOT/'app/client_dashboard.py')!r},run_name='client_test')\n"
        +f"ns['main'].__globals__['LOG_PATH']={str(path)!r}\nns['main']()\n")
    return AppTest.from_string(script,default_timeout=90)


class DashboardDataTests(SyntheticHistoryTestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.path=Path(self.temp.name)/'history.csv'
        self.df=fixtures(self.path)
    def tearDown(self):
        self.temp.cleanup()

    def test_all_4_records_and_original_file_are_preserved(self):
        before=LOG.read_bytes()
        self.assertEqual(len(load_history(LOG)),4)
        self.assertEqual(hashlib.sha256(before).hexdigest(),BASELINE_HASH)
        self.assertEqual(LOG.read_bytes(),before)

    def test_mixed_seven_nine_csv_and_missing_date_client_are_retained(self):
        path=Path(self.temp.name)/'mixed.csv'
        old=row(''); old['timestamp']=''
        new=row('A')
        write_csv(path,COLUMNS[:7],[[old[k] for k in COLUMNS[:7]],[new[k] for k in COLUMNS]])
        before=path.read_bytes();df=load_history(path)
        self.assertEqual(len(df),2)
        self.assertEqual(df['_timestamp'].isna().sum(),1)
        self.assertEqual((df['_client']=='').sum(),1)
        self.assertEqual(path.read_bytes(),before)

    def test_multiple_records_latest_and_undated_selection(self):
        selected=filter_history(self.df,'A')
        self.assertEqual(selected['_record_number'].tolist(),[2,1,3])
        self.assertEqual(selected.iloc[0]['decision'],'ACCEPT')
        self.assertEqual(len(filter_history(self.df,'')),1)
        self.assertTrue(filter_history(self.df,'unknown').empty)

    def test_kpi_formulas_and_specific_denominators(self):
        k=calculate_kpis(self.df)
        self.assertEqual(k['total'],6);self.assertEqual(k['valid_pd'],4)
        self.assertEqual(k['invalid_pd'],2);self.assertAlmostEqual(k['mean_pd'],.29)
        self.assertEqual(k['historical_comparable'],3);self.assertEqual(k['historical_above'],2)
        self.assertAlmostEqual(k['historical_above_rate'],2/3)
        self.assertEqual(k['recorded_decisions'],4);self.assertEqual(k['decision_counts'],{'REFUS-HIST':2,'ACCEPT':1,'REVIEW/REJECT':1})
        self.assertEqual(k['unique_clients'],2);self.assertEqual(k['missing_client'],1);self.assertEqual(k['undated'],1)

    def test_simulation_is_separate_and_does_not_mutate_history(self):
        before=self.df.copy(deep=True)
        low=calculate_kpis(self.df,.05);high=calculate_kpis(self.df,.95)
        self.assertEqual(low['simulation_above'],3);self.assertEqual(low['simulation_rate'],.75)
        self.assertEqual(high['simulation_above'],0);self.assertEqual(high['simulation_rate'],0)
        self.assertEqual(low['historical_above_rate'],high['historical_above_rate'])
        self.assertEqual(low['decision_counts'],high['decision_counts'])
        pd.testing.assert_frame_equal(before,self.df)
        with self.assertRaises(ValueError): calculate_kpis(self.df,1.1)

    def test_no_valid_pd_yields_unavailable_instead_of_zero_rate(self):
        invalid=self.df[self.df['_pd'].isna()]
        k=calculate_kpis(invalid,.5)
        self.assertIsNone(k['mean_pd']);self.assertIsNone(k['historical_above_rate']);self.assertIsNone(k['simulation_rate'])
        self.assertEqual(k['total'],2)

    def test_archived_values_win_over_recomputation_and_simulation(self):
        record=filter_history(self.df,'A').iloc[0]
        values,ctx,error=record_display(record)
        self.assertIsNone(error)
        self.assertEqual(values['pd_value'],'90.00%');self.assertEqual(values['score_1000'],'456')
        self.assertEqual(values['rating'],'BB');self.assertEqual(values['risk_level'],'SOURCE-RISK')
        self.assertEqual(values['decision'],'ACCEPT');self.assertEqual(values['threshold'],'80.00%')
        self.assertEqual(ctx['model_id'],'MODEL-ID');self.assertEqual(ctx['input_id'],'INPUT-ID')

    def test_invalid_pd_still_displays_recorded_other_fields_without_report(self):
        record=self.df[self.df['_record_number']==3].iloc[0]
        values,ctx,error=record_display(record)
        self.assertIsNone(ctx);self.assertTrue(error)
        self.assertEqual(values['pd_value'],'Indisponible')
        self.assertEqual(values['score_1000'],'777');self.assertEqual(values['rating'],'BB')
        self.assertEqual(values['decision'],'REFUS-HIST')
        for invalid in (None,float('nan'),float('inf'),-1,1.1,'unknown',True):
            test=dict(record,prob_default=invalid)
            self.assertEqual(record_display(test)[0]['pd_value'],'Indisponible')

    def test_legacy_missing_rating_decision_remains_missing(self):
        record=row();record.pop('rating');record.pop('decision')
        values,ctx,error=record_display(record)
        self.assertIsNone(error)
        self.assertEqual(values['rating'],'Non renseigne');self.assertEqual(values['decision'],'Non renseigne')
        self.assertEqual(ctx['score_source'],'derived_from_pd')
        self.assertTrue(any('Score non archive' in note for note in ctx['data_notes']))

    def test_date_filter_keeps_undated_explicitly_and_includes_full_last_day(self):
        selected=filter_history(self.df,'A',start='2026-02-01',end='2026-02-01')
        self.assertEqual(len(selected),2) # dated latest plus undated retained
        selected=filter_history(self.df,'A',start='2026-02-01',end='2026-02-01',include_undated=False)
        self.assertEqual(selected['_record_number'].tolist(),[2])
        path=Path(self.temp.name)/'dates.csv'
        for timestamp in ('2026-02-01T23:59:59.999999+00:00','2026-02-02T00:00:00+00:00'):
            append_prediction_record(path,dict(row(),timestamp=timestamp))
        self.assertEqual(len(filter_history(load_history(path),start='2026-02-01',end='2026-02-01')),1)

    def test_timeline_separates_clients_and_uses_historical_thresholds_and_gaps(self):
        fig=timeline_figure(self.df,.42)
        self.assertEqual(len(fig.data),6) # A, B and missing client; PD and stored threshold each
        for trace in fig.data:
            self.assertFalse(trace.connectgaps)
            self.assertEqual(len(trace.x),len(trace.y))
        a=next(trace for trace in fig.data if trace.name=='A : PD predite')
        self.assertEqual(list(a.y),[.06,.9])
        b=next(trace for trace in fig.data if trace.name=='B : PD predite')
        self.assertEqual(list(b.y),[None,0])
        threshold=next(trace for trace in fig.data if trace.name=='A : Seuil historique')
        self.assertEqual(list(threshold.y),[.1,.8])
        self.assertEqual(fig.layout.yaxis.tickformat,'.0%')
        self.assertEqual(fig.layout.shapes[0].y0,.42)

    def test_distribution_has_only_valid_pd_percent_axis_and_simulation_marker(self):
        fig=distribution_figure(self.df,.42)
        self.assertEqual(sorted(fig.data[0].x),[0,.06,.2,.9])
        self.assertEqual(fig.layout.xaxis.tickformat,'.0%')
        self.assertEqual(fig.layout.yaxis.title.text,'Nombre de predictions')
        self.assertEqual(fig.layout.shapes[0].x0,.42)
        self.assertFalse(distribution_figure(self.df).layout.shapes)

    def test_csv_export_contains_all_filtered_records_and_full_archived_result(self):
        exported=pd.read_csv(io.BytesIO(export_history_csv(self.df)))
        self.assertEqual(len(exported),6)
        record=exported[(exported.client_id=='A') & (exported.prob_default==.9)].iloc[0]
        self.assertEqual(record.score_1000,456);self.assertEqual(record.rating,'BB')
        self.assertEqual(record.risk_level,'SOURCE-RISK');self.assertEqual(record.decision,'ACCEPT')
        self.assertNotIn('_pd',exported.columns)

    def test_html_pdf_share_selected_dossier_contract(self):
        ctx=record_context(filter_history(self.df,'A').iloc[0])
        html=render_html_report(ctx);text=pdf_text(build_pdf_bytes(ctx,None,None))
        for value in report_display_values(ctx).values():
            self.assertIn(value,html);self.assertIn(value,text)
        self.assertNotIn('0.50',html+text)


class DashboardUITests(SyntheticHistoryTestCase):
    def test_main_navigation_opens_dashboard_and_all_4_rows(self):
        before=LOG.read_bytes()
        app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=90).run()
        app.switch_page('app/client_dashboard.py').run()
        self.assertEqual(len(app.exception),0,str(app.exception));self.assertEqual(len(app.error),0,str(app.error))
        self.assertEqual(app.metric[0].value,'4')
        self.assertEqual(app.title[0].value,'Dashboard Client / Historique')
        self.assertEqual(LOG.read_bytes(),before)

    def test_selection_simulation_and_real_html_pdf_download_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'history.csv';fixtures(path)
            app=page(path).run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            app.selectbox[0].set_value('A').run()
            self.assertEqual(app.metric[0].value,'3')
            self.assertEqual(app.metric[6].value,'90.00%');self.assertEqual(app.metric[7].value,'456')
            self.assertEqual(app.metric[10].value,'ACCEPT');self.assertEqual(app.metric[11].value,'80.00%')
            app.checkbox[1].set_value(True).run()
            app.slider[0].set_value(.95)
            captured_html=[];captured_pdf=[]
            def html(ctx):
                rendered=render_html_report(ctx);captured_html.append(rendered);return rendered
            def pdf(*args,**kwargs):
                rendered=build_pdf_bytes(*args,**kwargs);captured_pdf.append(rendered);return rendered
            with patch('utils.report_generator.render_html_report',side_effect=html),patch('utils.reporting_pdf.build_pdf_bytes',side_effect=pdf):
                app.run()
            self.assertEqual(len(app.exception),0,str(app.exception));self.assertEqual(len(app.error),0,str(app.error))
            self.assertEqual(app.metric[10].value,'ACCEPT');self.assertEqual(app.metric[11].value,'80.00%')
            self.assertEqual(app.metric[12].value,'0.0%')
            for rendered in (captured_html[0],pdf_text(captured_pdf[0])):
                for value in ('90.00%','456','BB','SOURCE-RISK','ACCEPT','80.00%'):
                    self.assertIn(value,rendered)
                self.assertNotIn('95.00%',rendered)
            app.selectbox[1].set_value(0).run()
            self.assertEqual(app.metric[6].value,'6.00%');self.assertEqual(app.metric[10].value,'REFUS-HIST')
            app.selectbox[1].set_value(2).run()
            self.assertEqual(app.metric[6].value,'Indisponible');self.assertEqual(app.metric[7].value,'777')
            self.assertTrue(any('Export HTML/PDF indisponible' in item.value for item in app.info))
            self.assertEqual(len(app.get('plotly_chart')),2)

    def test_missing_and_empty_logs_do_not_create_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'absent.csv'
            app=page(path).run()
            self.assertEqual(len(app.exception),0,str(app.exception));self.assertFalse(path.exists())
            self.assertTrue(any('Aucune' in item.value for item in app.info))
            path.write_text(','.join(COLUMNS)+'\n',encoding='utf-8')
            app=page(path).run()
            self.assertEqual(len(app.exception),0,str(app.exception));self.assertEqual(len(app.metric),0)

    def test_malformed_history_shows_error_instead_of_empty_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.csv';path.write_text(','.join(COLUMNS)+'\n"broken',encoding='utf-8')
            app=page(path).run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            self.assertTrue(any('Lecture du journal impossible' in item.value for item in app.error))
