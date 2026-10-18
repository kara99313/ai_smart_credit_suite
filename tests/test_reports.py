from synthetic_history import SyntheticHistoryTestCase
"""Reports preserve scoring results and never migrate or silently drop history."""
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from pypdf import PdfReader
from streamlit.testing.v1 import AppTest
from utils.csv_utils import read_logs_robust, sanitize_predictions_log
from utils.prediction_logs import COLUMNS, append_prediction_record, read_prediction_records, snapshot_path
from utils.report_generator import build_report_context, normalize_report_context, render_html_report, report_display_values
from utils.reporting_pdf import build_pdf_bytes

ROOT = Path(__file__).resolve().parents[1]
from synthetic_history import LOG, BASELINE_HASH


def row(client='CLIENT-1', pd_value=.06):
    return dict(timestamp='2026-10-18T12:00:00', client_id=client, prob_default=pd_value,
                pred_label=0, threshold=.1, inputs_json='{"HasMortgage": true}',
                model_tag='pipeline_v1', rating='BB', decision='ACCEPT')


def context(**updates):
    values=dict(client_id='CLIENT-1', timestamp='2026-10-18T12:00:00', pd_value=.06,
                threshold=.1, inputs_json='{"HasMortgage": true}', model_tag='pipeline_v1',
                rating='BB', score_1000=940, decision='ACCEPT', risk_level='RISK-SOURCE')
    values.update(updates)
    return build_report_context(**values)


def write_csv(path, header, rows):
    with Path(path).open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(header)
        for record in rows:
            writer.writerow(record)


def pdf_text(data):
    return '\n'.join(page.extract_text() for page in PdfReader(io.BytesIO(data)).pages)


def reports_page(path, folder):
    script=('import runpy\n'
        + f"ns=runpy.run_path({str(ROOT / 'app/report.py')!r},run_name='reports_test')\n"
        + f"ns['main'].__globals__['LOG_PATH']={str(path)!r}\n"
        + f"ns['main'].__globals__['REPORTS_DIR']={str(folder)!r}\n"
        + "ns['main']()\n")
    return AppTest.from_string(script,default_timeout=90)


class ReportContractTests(SyntheticHistoryTestCase):
    def test_preserves_scoring_result_including_zero_threshold(self):
        ctx=context(score_1000=937,threshold=0)
        self.assertEqual(ctx['rating'],'BB')  # old report grid incorrectly gave A
        self.assertEqual(ctx['score_1000'],937)  # source wins over recomputation
        self.assertEqual(ctx['risk_level'],'RISK-SOURCE')
        self.assertEqual(ctx['decision'],'ACCEPT')
        self.assertEqual(ctx['threshold'],0)
        self.assertEqual(ctx['pd_value'],.06)
        self.assertEqual(ctx,normalize_report_context(ctx))

    def test_missing_or_invalid_pd_never_becomes_zero(self):
        for value in (None,'',float('nan'),float('inf'),-.1,1.01,True,'unknown'):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,'PD absente ou invalide'):
                context(pd_value=value)
        self.assertEqual(context(pd_value=0)['pd_value'],0)

    def test_incomplete_history_does_not_choose_business_policy(self):
        ctx=context(rating=None,decision=None,risk_level=None,threshold=None,score_1000=None)
        self.assertIsNone(ctx['rating'])
        self.assertIsNone(ctx['decision'])
        self.assertIsNone(ctx['risk_level'])
        self.assertIsNone(ctx['threshold'])
        self.assertEqual(ctx['score_1000'],940)
        self.assertEqual(ctx['score_source'],'derived_from_pd')
        self.assertTrue(any('round(1000' in value for value in ctx['data_notes']))
        self.assertIn('Non renseigne',render_html_report(ctx))

    def test_html_and_actual_pdf_preserve_all_display_values(self):
        ctx=context(score_1000=937,threshold=.17,decision='SOURCE-DECISION')
        html=render_html_report(ctx)
        pdf=build_pdf_bytes(ctx,None,None)
        self.assertTrue(pdf.startswith(b'%PDF-'))
        text=pdf_text(pdf)
        for value in report_display_values(ctx).values():
            self.assertIn(value,html)
            self.assertIn(value,text)
        self.assertIn('data-field="rating">BB<',html)
        for value in ('PD a 12 mois','modele calibre','modele backteste','conforme IFRS'):
            self.assertNotIn(value,(html+text).lower())

    def test_global_importance_is_not_an_individual_contribution(self):
        frame=pd.DataFrame([dict(feature='FAKE_DIRECTION',importance=99,contribution=99)])
        text=pdf_text(build_pdf_bytes(context(),frame,None))
        html=render_html_report(context())
        for rendered in (html,text):
            self.assertIn('Aucune explication individuelle validee',rendered)
            self.assertIn('ne constitue pas une contribution individuelle',rendered)
            self.assertNotIn('FAKE_DIRECTION',rendered)
            self.assertNotIn('augmente la PD',rendered)
            self.assertNotIn('diminue la PD',rendered)

    def test_html_escapes_client_and_inputs(self):
        html=render_html_report(context(client_id='<script>bad()</script>',inputs_json='{"value":"<img>"}'))
        self.assertNotIn('<script>',html)
        self.assertIn('&lt;script&gt;',html)
        self.assertIn('&lt;img&gt;',html)


class HistoryTests(SyntheticHistoryTestCase):
    def test_mixed_seven_nine_columns_retains_all_rows_without_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'mixed.csv'
            first,second=row('OLD'),row('NEW')
            write_csv(path,COLUMNS[:7],[[first[k] for k in COLUMNS[:7]],[second[k] for k in COLUMNS]])
            before=path.read_bytes()
            df=read_logs_robust(path)
            self.assertEqual(len(df),2)
            self.assertEqual(df.iloc[1]['rating'],'BB')
            self.assertEqual(df.iloc[1]['decision'],'ACCEPT')
            self.assertIsNone(df.iloc[0]['rating'])
            sanitize_predictions_log(path)
            self.assertEqual(path.read_bytes(),before)

    def test_new_full_result_round_trip_with_legacy_and_current_header(self):
        for header in (COLUMNS[:7],COLUMNS):
            with self.subTest(width=len(header)),tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'history.csv'
                write_csv(path,header,[[row('OLD')[k] for k in header]])
                before=path.read_bytes()
                append_prediction_record(path,row(),dict(score_1000=937,risk_level='API-RISK',model_id='MODEL-ID',input_id='INPUT-ID'))
                self.assertTrue(path.read_bytes().startswith(before))
                with path.open(newline='',encoding='utf-8') as stream:
                    records=list(csv.reader(stream))
                self.assertEqual([len(record) for record in records],[len(header)]*3)
                df=read_logs_robust(path)
                self.assertEqual(len(df),2)
                new=df.iloc[1]
                self.assertEqual(new['score_1000'],937)
                self.assertEqual(new['rating'],'BB')
                self.assertEqual(new['decision'],'ACCEPT')
                self.assertEqual(new['risk_level'],'API-RISK')
                self.assertEqual(new['input_id'],'INPUT-ID')
                self.assertTrue(new['_result_snapshot'])

    def test_corrupt_snapshot_does_not_discard_csv_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'history.csv'
            write_csv(path,COLUMNS,[[row()[k] for k in COLUMNS]])
            snapshot_path(path).write_text('{bad json\n',encoding='utf-8')
            df=read_logs_robust(path)
            self.assertEqual(len(df),1)
            self.assertIn('unreadable',df.iloc[0]['_history_warning'])

    def test_unknown_csv_width_is_explicit_error_instead_of_dropped_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'history.csv'
            write_csv(path,COLUMNS,[list(range(10))])
            with self.assertRaisesRegex(ValueError,'no row was discarded'):
                read_logs_robust(path)

    def test_malformed_csv_raises_visible_error_without_rewrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'history.csv'
            path.write_text(','.join(COLUMNS)+'\n"unterminated',encoding='utf-8')
            before=path.read_bytes()
            with self.assertRaisesRegex(ValueError,'no row was discarded'):
                read_logs_robust(path)
            app=reports_page(path,tmp).run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            self.assertTrue(any('Lecture du journal impossible' in item.value for item in app.error))
            self.assertEqual(path.read_bytes(),before)

    def test_invalid_snapshot_is_rejected_before_csv_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'history.csv'
            write_csv(path,COLUMNS,[])
            before=path.read_bytes()
            with self.assertRaises(ValueError):
                append_prediction_record(path,row(),dict(score_1000=float('nan')))
            self.assertEqual(path.read_bytes(),before)

    def test_existing_4_records_are_byte_for_byte_unchanged(self):
        before=LOG.read_bytes()
        self.assertEqual(hashlib.sha256(before).hexdigest(),BASELINE_HASH)
        self.assertEqual(len(read_logs_robust(LOG)),4)
        sanitize_predictions_log(LOG)
        self.assertEqual(before,LOG.read_bytes())


class ReportsPageTests(SyntheticHistoryTestCase):
    def test_reports_opens_through_application_navigation(self):
        before=LOG.read_bytes()
        app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=90).run()
        self.assertEqual(len(app.exception),0,str(app.exception))
        app.switch_page('app/report.py').run()
        self.assertEqual(len(app.exception),0,str(app.exception))
        self.assertEqual(len(app.error),0,str(app.error))
        self.assertEqual(app.title[0].value,'Rapports HTML et PDF')
        self.assertEqual(len(app.selectbox),2)
        self.assertEqual(LOG.read_bytes(),before)

    def test_existing_history_opens_without_errors_or_writes(self):
        before=LOG.read_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            app=reports_page(LOG,tmp).run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            self.assertEqual(len(app.error),0,str(app.error))
            self.assertEqual(len(app.selectbox),2)
            self.assertTrue(any('4 dossiers' in element.value for element in app.caption))
        self.assertEqual(LOG.read_bytes(),before)

    def test_select_client_occurrence_and_save_both_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'history.csv'; folder=Path(tmp)/'reports'
            # Same timestamp/client, different occurrence: both remain selectable.
            for client,probability in (('CLIENT-A',.06),('CLIENT-B',.07),('CLIENT-A',.08)):
                record=row(client,probability)
                append_prediction_record(path,record,dict(score_1000=round(1000*(1-probability)),risk_level='API-RISK'))
            app=reports_page(path,folder).run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            app.selectbox[0].set_value('CLIENT-B').run()
            self.assertEqual(app.metric[0].value,'7.00%')
            app.selectbox[0].set_value('CLIENT-A').run()
            self.assertEqual(len(app.selectbox[1].options),2)
            app.selectbox[1].set_value(2).run()
            self.assertEqual(app.metric[0].value,'8.00%')
            app.button[0].click().run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            self.assertEqual(len(app.error),0,str(app.error))
            htmls=list(folder.glob('*.html')); pdfs=list(folder.glob('*.pdf'))
            self.assertEqual(len(htmls),1); self.assertEqual(len(pdfs),1)
            for rendered in (htmls[0].read_text(encoding='utf-8'),pdf_text(pdfs[0].read_bytes())):
                for expected in ('8.00%','920','BB','ACCEPT','API-RISK','CLIENT-A','10.00%'):
                    self.assertIn(expected,rendered)

    def test_legacy_incomplete_record_remains_usable_without_fake_rating(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'history.csv'
            write_csv(path,COLUMNS[:7],[[row()[key] for key in COLUMNS[:7]]])
            app=reports_page(path,tmp).run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            self.assertEqual(len(app.error),0,str(app.error))
            self.assertEqual(app.metric[2].value,'Non renseigne')
            self.assertEqual(app.metric[4].value,'Non renseigne')
            app.button[0].click().run()
            self.assertEqual(len(list(Path(tmp).glob('*.pdf'))),1)

    def test_missing_pd_warns_without_generating_zero_pd_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'history.csv'
            bad=row(pd_value='')
            write_csv(path,COLUMNS,[[bad[key] for key in COLUMNS]])
            app=reports_page(path,tmp).run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            self.assertTrue(any('PD absente ou invalide' in item.value for item in app.warning))
            self.assertEqual(len(app.metric),0)
            self.assertEqual(len(app.button),0)
