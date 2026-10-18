from admin_auth_support import authenticated_admin_test
"""Phase 13: truthful Administration, local audit and read-only observability."""
import csv
import io
import json
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import pandas as pd
from streamlit.testing.v1 import AppTest
from utils import audit, observability_data as obs, admin_diagnostics as admin, integration_services
from utils.prediction_logs import snapshot_path
from test_client_dashboard import fixtures
from test_reports import ROOT, LOG, BASELINE_HASH, write_csv


def button(app,label):
    return next(b for b in app.button if b.label==label)


def observability_page(root):
    script=('import runpy\n'+f"ns=runpy.run_path({str(ROOT/'app/observability.py')!r},run_name='observability_test')\n"
        +f"ns['main'].__globals__['PRED_LOG']={str(root/'predictions.csv')!r}\n"
        +f"ns['main'].__globals__['AGENT_LOG']={str(root/'agents.csv')!r}\n"
        +"ns['main']()\n")
    return AppTest.from_string(script,default_timeout=30)


class ObservabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.patch=patch.object(audit,'AUDIT_PATH',self.root/'audit.csv');self.patch.start()
    def tearDown(self):
        self.patch.stop();self.tmp.cleanup()

    def test_predictions_use_recorded_decisions_and_valid_denominators(self):
        path=self.root/'predictions.csv';fixtures(path)
        before={p:p.read_bytes() for p in (path,snapshot_path(path))}
        result=obs.prediction_snapshot(path);k=result['kpis']
        self.assertEqual(result['source']['state'],'available')
        self.assertEqual(k['total'],6);self.assertEqual(k['valid_pd'],4)
        self.assertAlmostEqual(k['mean_pd'],.29);self.assertEqual(k['acceptance_rate'],.5)
        self.assertEqual(k['recognized_decisions'],2);self.assertEqual(k['undated'],1)
        for p,raw in before.items():self.assertEqual(p.read_bytes(),raw)

    def test_no_valid_probability_or_decision_stays_unavailable(self):
        path=self.root/'predictions.csv';fixtures(path)
        with patch.object(obs,'load_history',return_value=__import__('utils.client_dashboard_data',fromlist=['load_history']).load_history(path).assign(_pd=None,decision=None)):
            result=obs.prediction_snapshot(path)
        self.assertIsNone(result['kpis']['mean_pd']);self.assertIsNone(result['kpis']['acceptance_rate'])

    def test_missing_empty_invalid_and_unreadable_files_are_distinct(self):
        path=self.root/'usage.csv'
        self.assertEqual(obs.csv_snapshot(path)['source']['state'],'missing')
        path.write_text('');self.assertEqual(obs.csv_snapshot(path)['source']['state'],'empty')
        path.write_text('a,b\nx,y,z\n');result=obs.csv_snapshot(path)
        self.assertEqual(result['source']['state'],'invalid');self.assertTrue(result['frame'].empty)
        with patch.object(Path,'stat',side_effect=PermissionError('PRIVATE')):
            self.assertEqual(obs.csv_snapshot(path)['source']['state'],'unavailable')

    def test_bad_prediction_header_is_not_a_prediction_observation(self):
        path=self.root/'predictions.csv';path.write_text('unrelated\nvalue\n')
        result=obs.prediction_snapshot(path)
        self.assertEqual(result['source']['state'],'invalid');self.assertIsNone(result['kpis'])

    def test_malformed_prediction_rows_are_not_silently_skipped(self):
        path=self.root/'predictions.csv';fixtures(path)
        with path.open('a') as stream:stream.write('1,2,3,4,5,6,7,8,9,10,11\n')
        before=path.read_bytes();result=obs.prediction_snapshot(path)
        self.assertEqual(result['source']['state'],'invalid');self.assertEqual(path.read_bytes(),before)

    def test_sidecar_warning_is_shown_without_rewriting(self):
        path=self.root/'predictions.csv';fixtures(path)
        with snapshot_path(path).open('a') as stream:stream.write('broken\n')
        before=snapshot_path(path).read_bytes();result=obs.prediction_snapshot(path)
        self.assertEqual(result['kpis']['total'],6);self.assertTrue(result['warnings'])
        self.assertEqual(snapshot_path(path).read_bytes(),before)

    def test_agent_success_only_known_boolean_values(self):
        frame=pd.DataFrame(dict(success=['True','false','1','0','','unknown'],tool=['now']*6,timestamp=['2026-01-01','2026-01-01T12:00:00Z','','bad','2026-01-02','2026-01-02']))
        result=obs.agent_statistics(frame)
        self.assertEqual(result['known_success'],4);self.assertEqual(result['unknown_success'],2)
        self.assertEqual(result['success_rate'],.5);self.assertEqual(result['undated'],2)
        self.assertEqual(result['daily'].to_numpy().sum(),4)

    def test_tool_metric_and_chart_labels_are_redacted(self):
        frame=pd.DataFrame(dict(tool=['token=UNKNOWN'],timestamp=['2026-01-01'],success=['true']))
        result=obs.agent_statistics(frame)
        self.assertNotIn('UNKNOWN',result['most_used']);self.assertNotIn('UNKNOWN',str(result['daily'].columns))

    def test_agent_csv_does_not_require_user_column(self):
        frame=pd.DataFrame(dict(tool=['now','now'],timestamp=['2026-01-01']*2))
        result=obs.agent_statistics(frame)
        self.assertIsNone(result['success_rate']);self.assertEqual(result['daily'].to_numpy().sum(),2)

    def test_agent_missing_fields_are_unknown_not_zero_success(self):
        result=obs.agent_statistics(pd.DataFrame({'extra':['{}']}))
        self.assertIsNone(result['success_rate']);self.assertIsNone(result['most_used']);self.assertEqual(result['undated'],1)

    def test_csv_duplicate_columns_and_bad_quoting_are_rejected(self):
        for content in ('a,a\n1,2\n','a,b\n"unclosed,x\n'):
            path=self.root/'agents.csv';path.write_text(content)
            self.assertEqual(obs.csv_snapshot(path)['source']['state'],'invalid')

    def test_table_redacts_nested_unknown_secrets_and_free_text(self):
        frame=pd.DataFrame([dict(extra=json.dumps({'api_key':'UNKNOWN','nested':{'password':'HIDDEN'}}),detail='token=UNREGISTERED Authorization: Bearer PRIVATEBEARER',url='https://u:PASS@example.org/?key=ABC',tool='<script>untrusted</script>')])
        rendered=obs.safe_frame(frame).to_json()
        for secret in ('UNKNOWN','HIDDEN','UNREGISTERED','PRIVATEBEARER','PASS','ABC'):self.assertNotIn(secret,rendered)
        self.assertIn('untrusted',rendered)

    def test_redaction_handles_malformed_json_and_password_aliases(self):
        value="log {'api_key': 'UNKNOWN', 'smtp_pass': 'HIDDEN'} broken"
        text=str(audit.redact_details(value))
        self.assertNotIn('UNKNOWN',text);self.assertNotIn('HIDDEN',text)
        self.assertNotIn('HIDDEN',str(audit.redact_details({'smtp_pass':'HIDDEN'})))

    def test_observability_ui_metrics_and_refresh_preserve_all_sources(self):
        fixtures(self.root/'predictions.csv')
        write_csv(self.root/'agents.csv',['timestamp','tool'],[['2026-01-01','now']])
        before={p:p.read_bytes() for p in self.root.iterdir()}
        with patch('requests.get') as get,patch('requests.post') as post:
            app=observability_page(self.root).run()
            self.assertFalse(app.exception)
            values={m.label:m.value for m in app.metric}
            self.assertEqual(values['Predictions archivees'],'6');self.assertEqual(values['Decisions ACCEPT archivees'],'50.0%')
            self.assertEqual(values['Succes declares dans le journal'],'Indisponible')
            button(app,'Actualiser les journaux').click().run()
        get.assert_not_called();post.assert_not_called()
        for p,raw in before.items():self.assertEqual(p.read_bytes(),raw)
        self.assertFalse(audit.AUDIT_PATH.exists())

    def test_empty_ui_does_not_invent_service_metrics(self):
        app=observability_page(self.root).run()
        self.assertFalse(app.exception);self.assertFalse(app.metric)
        self.assertGreaterEqual(len(app.info),3)

    def test_invalid_agent_csv_is_isolated_from_predictions(self):
        fixtures(self.root/'predictions.csv');(self.root/'agents.csv').write_text('a,b\nx,y,z\n')
        app=observability_page(self.root).run()
        self.assertFalse(app.exception);self.assertTrue(app.error)
        self.assertEqual(next(m.value for m in app.metric if m.label=='Predictions archivees'),'6')

    def test_main_navigation_and_service_diagnostic_link(self):
        with patch('requests.get') as get,patch('requests.post') as post:
            app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=30).run()
            app.switch_page('app/observability.py').run()
            self.assertFalse(app.exception)
            button(app,'Ouvrir les diagnostics explicites des services').click().run()
            self.assertFalse(app.exception)
        get.assert_not_called();post.assert_not_called()


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.patch=patch.object(audit,'AUDIT_PATH',self.root/'new'/'audit.csv');self.patch.start()
    def tearDown(self):
        self.patch.stop();self.tmp.cleanup()

    def test_import_does_not_create_or_modify_files(self):
        with patch.object(Path,'mkdir') as mkdir:
            runpy.run_path(str(ROOT/'utils/audit.py'),run_name='audit_import_test')
        mkdir.assert_not_called()

    def test_append_has_actual_utc_timestamp_no_assumed_admin_identity(self):
        audit.append_audit('configuration_saved',details={'api_key':'PRIVATE','provider':'openai'})
        frame=obs.csv_snapshot(audit.AUDIT_PATH)['frame']
        self.assertEqual(frame.iloc[0]['actor'],'session_non_authentifiee')
        self.assertTrue(frame.iloc[0]['ts'].endswith('+00:00'));self.assertNotIn('PRIVATE',audit.AUDIT_PATH.read_text())

    def test_audit_view_rejects_unrelated_csv_schema(self):
        audit.AUDIT_PATH.parent.mkdir();audit.AUDIT_PATH.write_text('wrong,header\nx,y\n')
        result=obs.csv_snapshot(audit.AUDIT_PATH,required_columns=audit.COLUMNS)
        self.assertEqual(result['source']['state'],'invalid')

    def test_existing_incompatible_audit_is_not_overwritten(self):
        audit.AUDIT_PATH.parent.mkdir();audit.AUDIT_PATH.write_text('wrong,header\nx,y\n')
        before=audit.AUDIT_PATH.read_bytes()
        with self.assertRaises(ValueError):audit.append_audit('event')
        self.assertEqual(audit.AUDIT_PATH.read_bytes(),before)

    def test_append_preserves_no_newline_existing_record(self):
        audit.AUDIT_PATH.parent.mkdir();audit.AUDIT_PATH.write_text('ts,actor,event,details\n2026-01-01,legacy,old,{}')
        audit.append_audit('new')
        result=obs.csv_snapshot(audit.AUDIT_PATH)
        self.assertEqual(len(result['frame']),2);self.assertEqual(result['source']['state'],'available')

    def test_concurrent_in_process_append_has_one_header_and_all_events(self):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(lambda n:audit.append_audit('event_'+str(n)),range(12)))
        frame=obs.csv_snapshot(audit.AUDIT_PATH)['frame']
        self.assertEqual(len(frame),12);self.assertEqual(frame['event'].nunique(),12)


@authenticated_admin_test
class AdministrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.patch=patch.object(audit,'AUDIT_PATH',self.root/'audit.csv');self.patch.start()
    def tearDown(self):
        self.patch.stop();self.tmp.cleanup()

    def test_diagnostics_show_effective_source_and_secret_presence_only(self):
        with patch('streamlit.secrets',{'LLM_PROVIDER':'openai','OPENAI_API_KEY':'PRIVATE'}),patch.dict(os.environ,{'LLM_PROVIDER':'groq','OPENAI_MODEL':'explicit-model'}):
            snap=admin.configuration_snapshot()
        self.assertEqual(snap['provider'],'openai')
        rows={row['parametre']:row for row in snap['rows']}
        self.assertEqual(rows['LLM_PROVIDER']['source'],'secrets Streamlit')
        self.assertEqual(rows['OPENAI_MODEL']['valeur'],'explicit-model')
        self.assertEqual(rows['OPENAI_API_KEY']['valeur'],'presente');self.assertNotIn('PRIVATE',str(snap))

    def test_code_fallback_is_labelled_as_configuration_not_observation(self):
        with patch('streamlit.secrets',{}),patch.dict(os.environ,{},clear=True):
            snap=admin.configuration_snapshot()
        rows={row['parametre']:row for row in snap['rows']}
        self.assertEqual(snap['provider'],'ollama')
        self.assertIn('pas une observation',rows['OLLAMA_MODEL']['source'])
        self.assertEqual(rows['LLM_PROVIDER']['valeur'],'Non renseigne')

    def test_invalid_provider_never_becomes_an_observed_groq_status(self):
        with patch('streamlit.secrets',{}),patch.dict(os.environ,{'LLM_PROVIDER':'invalid'}):
            snap=admin.configuration_snapshot()
        self.assertFalse(snap['valid']);self.assertIsNone(snap['provider'])

    def test_invalid_configuration_file_is_not_silently_reported_as_valid(self):
        from utils import settings
        bad=self.root/'config.yaml';bad.write_text('invalid: [broken')
        with patch.object(settings,'CONFIG_PATH',bad),patch.object(settings,'SETTINGS_PATH',self.root/'absent.json'),patch.object(admin,'ENV_PATH',self.root/'absent.env'):
            rows=admin.configuration_files()
        self.assertIn('invalide',rows[1]['etat']);self.assertEqual(rows[0]['etat'],'Absent')

    def test_admin_save_roundtrip_and_audit_use_temporary_env(self):
        env=self.root/'.env'
        real_save=integration_services.save_llm_configuration
        with patch('streamlit.secrets',{}),patch.dict(os.environ,{'LLM_PROVIDER':'openai','OPENAI_MODEL':'old','OPENAI_API_KEY':'PRIVATE'}),patch.object(integration_services,'save_llm_configuration',side_effect=lambda *args:real_save(*args,path=env)):
            app=AppTest.from_file(str(ROOT/'app/admin.py'),default_timeout=30).run()
            next(i for i in app.text_input if i.label=='OPENAI_MODEL').set_value('new-model')
            next(i for i in app.text_input if 'Nouvelle cle' in i.label).set_value('NEWPRIVATE')
            button(app,'Enregistrer dans .env').click().run()
            self.assertFalse(app.exception);self.assertTrue(app.success)
            self.assertEqual(os.environ['OPENAI_MODEL'],'new-model')
            self.assertEqual(os.environ['OPENAI_API_KEY'],'NEWPRIVATE')
            app.run()
            self.assertEqual(next(i for i in app.text_input if i.label=='OPENAI_MODEL').value,'new-model')
            self.assertEqual(next(i for i in app.text_input if 'Nouvelle cle' in i.label).value,'')
        raw=audit.AUDIT_PATH.read_text()
        self.assertIn('configuration_llm_enregistree',raw);self.assertNotIn('NEWPRIVATE',raw)

    def test_audit_failure_warns_without_claiming_configuration_failed(self):
        with patch.dict(os.environ,{'LLM_PROVIDER':'openai'}),patch.object(integration_services,'save_llm_configuration'),patch.object(audit,'append_audit',side_effect=PermissionError('PRIVATE')):
            app=AppTest.from_file(str(ROOT/'app/admin.py'),default_timeout=30).run()
            button(app,'Enregistrer dans .env').click().run()
        self.assertFalse(app.exception);self.assertTrue(app.success);self.assertTrue(app.warning)
        self.assertNotIn('PRIVATE',' '.join(w.value for w in app.warning))

    def test_admin_validation_error_is_audited_without_secret(self):
        with patch.dict(os.environ,{'LLM_PROVIDER':'openai'}),patch.object(integration_services,'save_llm_configuration',side_effect=ValueError('PRIVATE')):
            app=AppTest.from_file(str(ROOT/'app/admin.py'),default_timeout=30).run()
            button(app,'Enregistrer dans .env').click().run()
        self.assertFalse(app.exception);self.assertTrue(app.error)
        self.assertNotIn('PRIVATE',' '.join(e.value for e in app.error))
        self.assertIn('configuration_llm_refusee',audit.AUDIT_PATH.read_text())

    def test_admin_test_is_timestamped_audited_and_invalidated_when_config_changes(self):
        with patch.dict(os.environ,{'LLM_PROVIDER':'openai','OPENAI_MODEL':'first'}),patch('utils.llm_providers.simple_chat',return_value='RESPONSE') as chat:
            app=AppTest.from_file(str(ROOT/'app/admin.py'),default_timeout=30).run()
            self.assertNotIn('RESPONSE',' '.join(m.value for m in app.markdown));chat.assert_not_called()
            button(app,'Tester le LLM').click().run()
            self.assertFalse(app.exception);self.assertIn('RESPONSE',' '.join(m.value for m in app.markdown))
            self.assertTrue(any('Dernier test explicite' in c.value for c in app.caption))
            os.environ['OPENAI_MODEL']='second';app.run()
            self.assertNotIn('RESPONSE',' '.join(m.value for m in app.markdown))
        self.assertIn('test_llm_termine',audit.AUDIT_PATH.read_text());self.assertNotIn('RESPONSE',audit.AUDIT_PATH.read_text())

    def test_admin_empty_prompt_does_not_call_service(self):
        with patch('utils.llm_providers.simple_chat') as chat:
            app=AppTest.from_file(str(ROOT/'app/admin.py'),default_timeout=30).run()
            app.text_area[0].set_value('   ')
            button(app,'Tester le LLM').click().run()
        self.assertFalse(app.exception);chat.assert_not_called();self.assertTrue(app.error)


if __name__=='__main__':unittest.main()
