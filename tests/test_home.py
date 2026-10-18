from synthetic_history import SyntheticHistoryTestCase
"""Home delegates data/aggregation and offers truthful health and real navigation."""
import hashlib
import runpy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest
from utils.client_dashboard_data import load_history
from utils.global_dashboard_data import portfolio_kpis
from utils.prediction_logs import COLUMNS,append_prediction_record
from test_reports import ROOT,LOG,BASELINE_HASH,row,write_csv
from test_client_dashboard import fixtures


def module():
    return runpy.run_path(str(ROOT/'app/home.py'),run_name='home_test')


def page(path):
    script=('import runpy\n'+f"ns=runpy.run_path({str(ROOT/'app/home.py')!r},run_name='home_page_test')\n"
        +f"ns['main'].__globals__['LOG_PATH']={str(path)!r}\nns['main']()\n")
    return AppTest.from_string(script,default_timeout=90)


def button(app,label):
    return next(element for element in app.button if element.label==label)


class HomeDataTests(SyntheticHistoryTestCase):
    def test_4_rows_all_dates_and_original_bytes(self):
        before=LOG.read_bytes();df=module()['load_logs'](LOG)
        self.assertEqual(len(df),4);self.assertEqual(df['_timestamp'].isna().sum(),0)
        self.assertEqual(hashlib.sha256(before).hexdigest(),BASELINE_HASH)
        self.assertEqual(LOG.read_bytes(),before)

    def test_kpis_exactly_match_global_helper_not_a_new_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            df=fixtures(Path(tmp)/'history.csv');k=module()['kpis_from_df'](df)
            self.assertEqual(k,portfolio_kpis(df))
            self.assertEqual(k['total'],6);self.assertEqual(k['unique_clients'],2)
            self.assertAlmostEqual(k['mean_pd'],.29)
            self.assertEqual(k['acceptance_rate'],.5) # recorded recognized outcomes, not PD < .5
            self.assertEqual(k['recorded_decisions'],4)

    def test_invalid_pd_and_missing_decisions_are_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            df=fixtures(Path(tmp)/'history.csv')
            invalid=df[df['_pd'].isna()].copy();invalid['decision']=None
            k=module()['kpis_from_df'](invalid)
            self.assertIsNone(k['mean_pd']);self.assertIsNone(k['acceptance_rate'])
            k=module()['kpis_from_df'](pd.DataFrame())
            self.assertEqual(k['total'],0);self.assertIsNone(k['mean_pd']);self.assertIsNone(k['acceptance_rate'])

    def test_mixed_csv_and_jsonl_and_fresh_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'mixed.csv';old=row('OLD');new=row('NEW')
            write_csv(path,COLUMNS[:7],[[old[k] for k in COLUMNS[:7]],[new[k] for k in COLUMNS]])
            before=path.read_bytes();ns=module()
            df=ns['load_logs'](path)
            self.assertEqual(len(df),2);self.assertEqual(path.read_bytes(),before)
            self.assertIsNone(df[df['_client']=='OLD'].iloc[0]['decision'])
            append_prediction_record(path,row('API'),dict(score_1000=937,risk_level='SOURCE-RISK',model_id='MODEL-ID'))
            df=ns['load_logs'](path)
            self.assertEqual(len(df),3)
            record=df[df['_client']=='API'].iloc[0]
            self.assertEqual(record['score_1000'],937);self.assertEqual(record['risk_level'],'SOURCE-RISK')

    def test_api_check_requires_real_health_response_and_never_certifies_model(self):
        for response,expected in (((True,{'status':'ok','model_loaded':True}),'Joignable (/health)'),
                                  ((True,'OK'),'Indisponible'),
                                  ((True,{'status':'failed'}),'Indisponible'),
                                  ((False,'connection refused'),'Indisponible')):
            with self.subTest(response=response),patch('utils.api_client.ping_backend',return_value=response) as ping:
                result=module()['check_api']('http://127.0.0.1:18000')
                self.assertEqual(result['api'],expected)
                self.assertIn('checked_at',result)
                ping.assert_called_once_with(base='http://127.0.0.1:18000',timeout=3)
                self.assertNotIn('model_operational',result)

    def test_text_does_not_claim_shap_or_compliance_or_fixed_auc(self):
        source=(ROOT/'app/home.py').read_text(encoding='utf-8')
        self.assertNotIn('auc_proxy',source);self.assertNotIn('0.50',source)
        self.assertNotIn('conformit\u00e9 int\u00e9gr',source)
        self.assertNotIn('pred_label.mean',source)


class HomeUITests(SyntheticHistoryTestCase):
    def test_opens_full_history_with_correct_kpis_and_no_external_check(self):
        before=LOG.read_bytes()
        with patch('utils.api_client.ping_backend') as ping:
            app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=90).run()
        self.assertEqual(len(app.exception),0,str(app.exception));self.assertEqual(len(app.error),0,str(app.error))
        self.assertEqual([metric.value for metric in app.metric],['4','2','25.00%','4','50.00%','Indisponible'])
        self.assertEqual(len(app.get('plotly_chart')),2)
        self.assertEqual(len([element for element in app.button if element.label.startswith('Ouvrir ')]),4)
        ping.assert_not_called()
        self.assertEqual(LOG.read_bytes(),before)

    def test_four_shortcuts_really_open_the_registered_destinations(self):
        before=LOG.read_bytes()
        for label,title in (('Pr\u00e9diction',None),('Rapports','Rapports HTML et PDF'),
                            ('Dashboard Client','Dashboard Client / Historique'),
                            ('Dashboard Global','Dashboard Global / Portefeuille')):
            with self.subTest(label=label):
                app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=90).run()
                button(app,f'Ouvrir {label}').click().run()
                self.assertEqual(len(app.exception),0,str(app.exception));self.assertEqual(len(app.error),0,str(app.error))
                if title:
                    self.assertEqual(app.title[0].value,title)
                else:
                    self.assertTrue(any('Lancer la pr' in element.label for element in app.button))
                    self.assertEqual(len(app.number_input),12)
        self.assertEqual(LOG.read_bytes(),before)

    def test_health_button_calls_endpoint_only_when_clicked_and_shows_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'history.csv';fixtures(path)
            with patch('utils.api_client.ping_backend',return_value=(True,{'status':'ok','model_loaded':True})) as ping:
                app=page(path).run()
                ping.assert_not_called()
                button(app,'V\u00e9rifier FastAPI').click().run()
                self.assertEqual(len(app.exception),0,str(app.exception))
                ping.assert_called_once()
                texts=' '.join(str(item.value) for item in app.markdown)
                self.assertIn('Joignable (/health)',texts)
                self.assertIn('Mod\u00e8le de scoring : Indisponible',texts)
                self.assertIn('RAG : Indisponible',texts)
                self.assertIn('LLM : Indisponible',texts)
                self.assertTrue(any('ne prouve pas' in item.value for item in app.caption))

    def test_api_outage_leaves_home_functional(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'history.csv';fixtures(path)
            with patch('utils.api_client.ping_backend',return_value=(False,'connection refused')):
                app=page(path).run()
                button(app,'V\u00e9rifier FastAPI').click().run()
                self.assertEqual(len(app.exception),0,str(app.exception));self.assertEqual(len(app.error),0,str(app.error))
                self.assertEqual(app.metric[0].value,'6')
                self.assertTrue(any('inaccessible' in item.value for item in app.caption))

    def test_changed_api_configuration_does_not_reuse_previous_health_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'history.csv';fixtures(path)
            with patch('utils.runtime_config.backend_url',return_value='http://127.0.0.1:18000'),patch('utils.api_client.ping_backend',return_value=(True,{'status':'ok'})):
                app=page(path).run();button(app,'V\u00e9rifier FastAPI').click().run()
            with patch('utils.runtime_config.backend_url',return_value='http://127.0.0.1:18001'):
                app.run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            texts=' '.join(str(item.value) for item in app.markdown)
            self.assertNotIn('Joignable (/health)',texts)
            self.assertIn('non v\u00e9rifi\u00e9',texts)

    def test_invalid_api_configuration_and_bad_csv_are_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.csv';path.write_text(','.join(COLUMNS)+'\n"broken',encoding='utf-8')
            with patch('utils.runtime_config.backend_url',side_effect=ValueError('invalid')):
                app=page(path).run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            self.assertTrue(any('Configuration API invalide' in item.value for item in app.warning))
            self.assertTrue(button(app,'V\u00e9rifier FastAPI').disabled)
            self.assertTrue(any('Lecture du journal impossible' in item.value for item in app.error))

    def test_empty_log_and_missing_data_do_not_become_success_or_zero_pd(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'history.csv'
            app=page(path).run()
            self.assertEqual(len(app.exception),0,str(app.exception));self.assertFalse(path.exists())
            self.assertEqual(app.metric[2].value,'Indisponible');self.assertEqual(app.metric[4].value,'Indisponible')
            record=dict(row(),prob_default='',decision='',rating='')
            write_csv(path,COLUMNS,[[record[name] for name in COLUMNS]])
            app=page(path).run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            self.assertEqual(app.metric[0].value,'1');self.assertEqual(app.metric[2].value,'Indisponible')
            self.assertEqual(app.metric[4].value,'Indisponible')

    def test_unvalidated_components_are_identified_without_claiming_readiness(self):
        app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=90).run()
        self.assertTrue(any("pas valid" in item.value for item in app.warning))
        self.assertTrue(any('Health Check' in item.value and 'RAG' in item.value for item in app.markdown))
