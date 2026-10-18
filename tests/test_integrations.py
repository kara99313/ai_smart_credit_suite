"""Phase 12: active connectors, configuration and Streamlit boundaries."""
import json
import os
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import requests
from dotenv import dotenv_values
from streamlit.testing.v1 import AppTest
from utils import settings, web_search as web, integration_services as integrations

ROOT=Path(__file__).resolve().parents[1]
WEB=dict(enabled=True,domains=[],timeout=3.5,max_results=3)
HTML='<a href="https://example.org/news" class="result__a"><b>Actual</b> title</a><a class="result__snippet">Actual excerpt</a>'


def response(body=HTML,status=200):
    result=Mock(text=body,status_code=status,headers={})
    if status>=400: result.raise_for_status.side_effect=requests.HTTPError('PRIVATE')
    return result


def button(app,label):
    return next(b for b in app.button if b.label==label)


class SettingsIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.patches=[patch.object(settings,'DATA_DIR',self.root),patch.object(settings,'SETTINGS_PATH',self.root/'settings.json'),patch.object(settings,'CONFIG_PATH',self.root/'config.yaml'),patch.object(settings,'_get_secret',side_effect=lambda name,default='':{'LLM_PROVIDER':'openai','OPENAI_MODEL':'env-model','OPENAI_API_KEY':'PRIVATE'}.get(name,default))]
        for item in self.patches:item.start()
    def tearDown(self):
        for item in reversed(self.patches):item.stop()
        self.tmp.cleanup()

    def test_yaml_web_preferences_and_secret_environment_priority(self):
        (self.root/'config.yaml').write_text('web:\n  enable_search: false\n  max_results: 2\n  allowed_domains: [example.org]\n',encoding='utf-8')
        (self.root/'settings.json').write_text(json.dumps(dict(enable_web_search=True,openai_api_key='UNTRUSTED',unknown_field=True,llm_provider='groq')))
        cfg=settings.get_settings()
        self.assertTrue(cfg.enable_web_search);self.assertEqual(cfg.allowed_domains,['example.org'])
        self.assertEqual(cfg.openai_api_key,'PRIVATE');self.assertEqual(cfg.llm_provider,'openai');self.assertEqual(cfg.llm_model,'env-model')

    def test_save_preferences_excludes_all_api_keys(self):
        cfg=settings.AppSettings(groq_api_key='PRIVATE',openai_api_key='PRIVATE',serpapi_key='PRIVATE')
        settings.save_settings(cfg)
        self.assertNotIn('PRIVATE',(self.root/'settings.json').read_text())

    def test_bad_backend_does_not_disable_other_settings(self):
        with patch('utils.runtime_config.backend_url',side_effect=ValueError('invalid')):
            cfg=settings.get_settings()
        self.assertEqual(cfg.api_base_url,'');self.assertEqual(cfg.llm_provider,'openai')

    def test_get_secret_falls_back_when_secrets_file_lacks_field(self):
        with patch('streamlit.secrets',{}),patch.dict(os.environ,{'INTEGRATION_TEST_VALUE':'ENV'}):
            self.assertEqual(settings.get_setting('INTEGRATION_TEST_VALUE'),'ENV')

    def test_corrupt_preferences_do_not_crash(self):
        (self.root/'settings.json').write_text('{broken')
        self.assertIsInstance(settings.get_settings(),settings.AppSettings)


class WebIntegrationTests(unittest.TestCase):
    def setUp(self):
        original = web.get_setting
        patcher = patch.object(web, 'get_setting', side_effect=lambda name, default='': '' if name == 'TAVILY_API_KEY' else original(name, default))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_effective_environment_overrides_preferences(self):
        cfg=settings.AppSettings(enable_web_search=False,allowed_domains=['old.org'],web_search_max=2)
        env={'WEB_SEARCH_ENABLED':'true','WEB_ALLOWED_DOMAINS':'EXAMPLE.ORG, bis.org','WEB_TIMEOUT':'2.5'}
        with patch.object(settings,'get_settings',return_value=cfg),patch.object(web,'get_setting',side_effect=lambda name,default='':env.get(name,default)):
            result=web.web_configuration()
        self.assertEqual(result,dict(enabled=True,domains=['example.org','bis.org'],timeout=2.5,max_results=2))

    def test_invalid_configuration_fails_without_network(self):
        for name,value in [('WEB_TIMEOUT','nan'),('WEB_TIMEOUT','0'),('WEB_TIMEOUT','61'),('WEB_SEARCH_ENABLED','perhaps'),('WEB_ALLOWED_DOMAINS','https://example.org')]:
            with self.subTest(name=name,value=value),patch.object(settings,'get_settings',return_value=settings.AppSettings()),patch.object(web,'get_setting',side_effect=lambda key,default='':value if key==name else default),self.assertRaises(ValueError):
                web.web_configuration()

    def test_disabled_web_does_not_contact_service(self):
        with patch.object(web,'web_configuration',return_value=dict(WEB,enabled=False)),patch.object(web.requests,'post') as post,patch.object(web,'_search_web_ddg_lib') as library:
            self.assertEqual(web.search_web('query'),[])
        post.assert_not_called();library.assert_not_called()

    def test_library_success_validates_schema_and_domains(self):
        hits=[dict(href=url,title='Title',body='Body') for url in ('https://example.org/a','https://sub.example.org/a','https://example.org.evil.net/a','javascript:alert(1)','https://user:PRIVATE@example.org/a')]+[{'href':7}]
        with patch.object(web,'web_configuration',return_value=dict(WEB,domains=['example.org'])),patch.object(web,'_HAS_DDG',True),patch.object(web,'_search_web_ddg_lib',return_value=hits),patch.object(web.requests,'post') as post:
            result=web.search_web('query')
        self.assertEqual(len(result),2);post.assert_not_called()

    def test_html_fallback_attribute_order_and_real_timeout(self):
        with patch.object(web,'web_configuration',return_value=WEB),patch.object(web,'_HAS_DDG',True),patch.object(web,'_search_web_ddg_lib',side_effect=web.WebSearchError('unavailable')),patch.object(web.requests,'post',return_value=response()) as post:
            hits=web.search_web('query')
        self.assertEqual(hits,[dict(title='Actual title',href='https://example.org/news',body='Actual excerpt')])
        self.assertEqual(post.call_args.kwargs['timeout'],3.5)

    def test_http_failure_is_not_empty_success(self):
        with patch.object(web,'web_configuration',return_value=WEB),patch.object(web,'_HAS_DDG',False),patch.object(web.requests,'post',return_value=response(status=503)),self.assertRaises(web.WebSearchError) as error:
            web.search_web('query')
        self.assertNotIn('PRIVATE',str(error.exception))

    def test_timeout_is_safe_and_propagated(self):
        with patch.object(web,'web_configuration',return_value=WEB),patch.object(web,'_HAS_DDG',False),patch.object(web.requests,'post',side_effect=requests.Timeout('PRIVATE')),self.assertRaises(web.WebSearchError) as error:
            web.search_web('query')
        self.assertNotIn('PRIVATE',str(error.exception))

    def test_no_results_remains_empty(self):
        with patch.object(web,'web_configuration',return_value=WEB),patch.object(web,'_HAS_DDG',False),patch.object(web.requests,'post',return_value=response('<html>No results</html>')):
            self.assertEqual(web.search_web('query'),[])

    def test_fetch_cannot_follow_redirect_outside_allowlist(self):
        redirect=response(status=302);redirect.headers={'Location':'https://evil.net/private'}
        with patch.object(web,'web_configuration',return_value=dict(WEB,domains=['example.org'])),patch.object(web.requests,'get',return_value=redirect) as get:
            self.assertEqual(web.fetch_and_clean('https://example.org/a'),'')
        self.assertEqual(get.call_count,1);self.assertFalse(get.call_args.kwargs['allow_redirects'])

    def test_fetch_disabled_or_disallowed_never_contacts_service(self):
        for config,url in ((dict(WEB,enabled=False),'https://example.org'),(dict(WEB,domains=['bis.org']),'https://example.org')):
            with patch.object(web,'web_configuration',return_value=config),patch.object(web.requests,'get') as get:
                self.assertEqual(web.fetch_and_clean(url),'')
            get.assert_not_called()

    def test_fetch_cleans_actual_html(self):
        with patch.object(web,'web_configuration',return_value=WEB),patch.object(web.requests,'get',return_value=response('<script>hidden()</script><p>Useful text</p>')):
            self.assertEqual(web.fetch_and_clean('https://example.org'),'Useful text')


class AdminConfigurationTests(unittest.TestCase):
    def test_saved_configuration_applies_to_running_process(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{},clear=False):
            path=Path(directory)/'.env';path.write_text('OTHER=retained\nLLM_PROVIDER=groq\n')
            integrations.save_llm_configuration('openai','new-model',"key#with'quotes\\slash",path=path)
            result=dotenv_values(path,interpolate=False)
            self.assertEqual(result['OTHER'],'retained');self.assertEqual(result['OPENAI_API_KEY'],"key#with'quotes\\slash")
            self.assertEqual(os.environ['LLM_PROVIDER'],'openai');self.assertEqual(os.environ['OPENAI_MODEL'],'new-model')

    def test_invalid_values_never_write_or_change_environment(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'LLM_PROVIDER':'groq'}):
            path=Path(directory)/'.env';path.write_text('OTHER=retained\n')
            for values in [('openai','ok','bad\nINJECT=x',''),('openai','','',''),('ollama','ok','','http://user:PRIVATE@localhost:11434')]:
                with self.subTest(values=values),self.assertRaises(ValueError):
                    integrations.save_llm_configuration(*values,path=path)
                self.assertEqual(path.read_text(),'OTHER=retained\n');self.assertEqual(os.environ['LLM_PROVIDER'],'groq')

    def test_write_failure_keeps_process_configuration(self):
        with patch.dict(os.environ,{'LLM_PROVIDER':'groq'}):
            with patch.object(Path,'write_text',side_effect=PermissionError('PRIVATE')),self.assertRaises(OSError) as error:
                integrations.save_llm_configuration('openai','new',path=Path(tempfile.gettempdir())/'phase12-nonexistent.env')
            self.assertNotIn('PRIVATE',str(error.exception));self.assertEqual(os.environ['LLM_PROVIDER'],'groq')

    def test_blank_key_keeps_existing_secret(self):
        with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{'OPENAI_API_KEY':'PRIVATE'}):
            path=Path(directory)/'.env';path.write_text("OPENAI_API_KEY='PRIVATE'\n")
            integrations.save_llm_configuration('openai','new',path=path)
            self.assertEqual(dotenv_values(path)['OPENAI_API_KEY'],'PRIVATE');self.assertEqual(os.environ['OPENAI_API_KEY'],'PRIVATE')

    def test_duplicate_exported_env_value_is_replaced_once(self):
        content=integrations._set_env_var('export LLM_PROVIDER=old\nLLM_PROVIDER=older\nOTHER=x\n','LLM_PROVIDER','openai')
        self.assertEqual(content.count('LLM_PROVIDER='),1);self.assertEqual(dotenv_values(stream=__import__('io').StringIO(content))['OTHER'],'x')


class IntegrationPageTests(unittest.TestCase):
    def setUp(self):
        original = web.get_setting
        patcher = patch.object(web, 'get_setting', side_effect=lambda name, default='': '' if name == 'TAVILY_API_KEY' else original(name, default))
        patcher.start()
        self.addCleanup(patcher.stop)
        # Administration now records actions. Isolate its new audit side effect.
        self.audit_temp=tempfile.TemporaryDirectory()
        self.audit_patch=patch('utils.audit.AUDIT_PATH',Path(self.audit_temp.name)/'audit.csv')
        self.audit_patch.start()

    def tearDown(self):
        self.audit_patch.stop()
        self.audit_temp.cleanup()

    def test_page_opens_without_any_network_call_or_secret_widget(self):
        with patch('requests.get') as get,patch('requests.post') as post:
            app=AppTest.from_file(str(ROOT/'app/integrations.py'),default_timeout=30).run()
        self.assertFalse(app.exception);get.assert_not_called();post.assert_not_called()
        self.assertFalse(any('SERPAPI' in item.label for item in app.text_input))

    def test_page_invalid_backend_reply_is_not_available(self):
        reply=response(status=503);reply.json.return_value={'status':'ok'}
        with patch('requests.get',return_value=reply) as get:
            app=AppTest.from_file(str(ROOT/'app/integrations.py'),default_timeout=30).run()
            button(app,'Tester la connexion FastAPI').click().run()
        self.assertFalse(app.exception)
        self.assertIn('Indisponible',' '.join(m.value for m in app.markdown));self.assertEqual(get.call_args.kwargs['timeout'],4)

    def test_page_backend_timeout_has_no_secret(self):
        with patch('requests.get',side_effect=requests.Timeout('PRIVATE')):
            app=AppTest.from_file(str(ROOT/'app/integrations.py'),default_timeout=30).run()
            button(app,'Tester la connexion FastAPI').click().run()
        self.assertFalse(app.exception);self.assertNotIn('PRIVATE',' '.join(m.value for m in app.markdown))

    def test_page_api_form_uses_existing_client_without_logging(self):
        result=dict(pd=.2,score_1000=800,rating='B',decision='REVIEW/REJECT')
        with patch('utils.api_client.predict_credit_api',return_value=result) as predict:
            app=AppTest.from_file(str(ROOT/'app/integrations.py'),default_timeout=30).run()
            next(b for b in app.button if 'API' in b.label and 'FastAPI' not in b.label).click().run()
        self.assertFalse(app.exception);self.assertEqual(predict.call_count,1);self.assertEqual(app.metric[0].value,'20.00%')
        self.assertEqual(predict.call_args.kwargs['timeout'],30)

    def test_page_api_failure_is_sanitized(self):
        with patch('utils.api_client.predict_credit_api',side_effect=requests.Timeout('PRIVATE')):
            app=AppTest.from_file(str(ROOT/'app/integrations.py'),default_timeout=30).run()
            next(b for b in app.button if 'API' in b.label and 'FastAPI' not in b.label).click().run()
        self.assertFalse(app.exception);self.assertNotIn('PRIVATE',' '.join(e.value for e in app.error))
        self.assertFalse(app.metric)

    def test_llm_probe_uses_service_response_and_handles_provider_failures(self):
        from utils import health_diagnostics as health
        client=Mock(invoke=Mock(return_value=NS(content='OK')))
        with patch.object(health,'get_setting',side_effect=lambda name,default='': 'PRIVATE' if name=='OPENAI_API_KEY' else default),patch.object(health,'build_client',return_value=client):
            result=health.provider_check('openai')
            self.assertEqual(result['state'],'Fonctionnel')
            failures=[TimeoutError('PRIVATE'),ImportError('PRIVATE'),RuntimeError('PRIVATE')]
            for code in (401,429,503):
                error=RuntimeError('PRIVATE');error.status_code=code;failures.append(error)
            for error in failures:
                client.invoke.side_effect=error
                result=health.provider_check('openai')
                self.assertEqual(result['state'],'Indisponible');self.assertNotIn('PRIVATE',result['explanation'])
            client.invoke.side_effect=None;client.invoke.return_value=NS(content='')
            self.assertEqual(health.provider_check('openai')['state'],'Indisponible')

    def test_page_malformed_health_response_is_not_available(self):
        reply=response();reply.json.side_effect=ValueError('PRIVATE')
        with patch('requests.get',return_value=reply):
            app=AppTest.from_file(str(ROOT/'app/integrations.py'),default_timeout=30).run()
            button(app,'Tester la connexion FastAPI').click().run()
        self.assertFalse(app.exception)
        text=' '.join(m.value for m in app.markdown)
        self.assertIn('Indisponible',text);self.assertNotIn('PRIVATE',text)

    def test_page_llm_unavailable_is_displayed(self):
        from utils import health_diagnostics as health
        with patch.object(health,'provider_check',return_value=dict(state='Indisponible',explanation='Cle absente.')) as probe:
            app=AppTest.from_file(str(ROOT/'app/integrations.py'),default_timeout=30).run()
            button(app,'Tester le fournisseur (appel reel)').click().run()
        self.assertFalse(app.exception);probe.assert_called_once();self.assertIn('Cle absente.',' '.join(m.value for m in app.markdown))

    def test_web_page_failure_is_not_success(self):
        with patch.object(web,'web_configuration',return_value=WEB),patch.object(web,'search_web',side_effect=requests.Timeout('PRIVATE')):
            app=AppTest.from_file(str(ROOT/'app/integrations.py'),default_timeout=30).run()
            next(t for t in app.text_input if t.label=='Requete de test Web').set_value('question').run()
            button(app,'Tester la recherche Web').click().run()
        self.assertFalse(app.exception);text=' '.join(m.value for m in app.markdown)
        self.assertIn('Recherche Web indisponible',text);self.assertNotIn('PRIVATE',text)

    def test_web_preferences_saved_from_ui_control_real_connector(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(settings,'DATA_DIR',root),patch.object(settings,'SETTINGS_PATH',root/'preferences.json'),patch.object(settings,'CONFIG_PATH',root/'absent.yaml'),patch.object(web,'get_setting',side_effect=lambda name,default='':default):
                app=AppTest.from_file(str(ROOT/'app/integrations.py'),default_timeout=30).run()
                app.toggle[0].set_value(True)
                next(area for area in app.text_area if 'Domaines' in area.label).set_value('example.org')
                button(app,'Enregistrer les parametres Web').click().run()
                self.assertFalse(app.exception)
                persisted=json.loads((root/'preferences.json').read_text())
                self.assertTrue(persisted['enable_web_search'])
                with patch.object(web,'_HAS_DDG',False),patch.object(web.requests,'post',return_value=response()) as post:
                    result=web.search_web('news')
                self.assertEqual(len(result),1);post.assert_called_once()
                self.assertEqual(web.web_configuration()['domains'],['example.org'])

    def test_main_navigation_can_open_integrations(self):
        with patch('requests.get') as get,patch('requests.post') as post:
            app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=30).run()
            app.switch_page('app/integrations.py').run()
        self.assertFalse(app.exception);get.assert_not_called();post.assert_not_called()

    def test_admin_never_prefills_existing_api_secret(self):
        with patch.dict(os.environ,{'LLM_PROVIDER':'openai','OPENAI_API_KEY':'PRIVATE'}),patch('requests.get') as get,patch('requests.post') as post:
            app=AppTest.from_file(str(ROOT/'app/admin.py'),default_timeout=30).run()
        self.assertFalse(app.exception);self.assertFalse(any(item.value=='PRIVATE' for item in app.text_input))
        get.assert_not_called();post.assert_not_called()

    def test_admin_write_error_is_displayed_without_secret(self):
        with patch.dict(os.environ,{'LLM_PROVIDER':'openai'}),patch.object(integrations,'save_llm_configuration',side_effect=PermissionError('PRIVATE')):
            app=AppTest.from_file(str(ROOT/'app/admin.py'),default_timeout=30).run()
            button(app,'Enregistrer dans .env').click().run()
        self.assertFalse(app.exception);self.assertTrue(app.error)
        self.assertNotIn('PRIVATE',' '.join(item.value for item in app.error))

    def test_admin_save_invalidates_old_agent(self):
        with patch.dict(os.environ,{'LLM_PROVIDER':'openai'}),patch.object(integrations,'save_llm_configuration') as save:
            app=AppTest.from_file(str(ROOT/'app/admin.py'),default_timeout=30).run()
            app.session_state['agent']='old-client'
            button(app,'Enregistrer dans .env').click().run()
        self.assertFalse(app.exception);save.assert_called_once();self.assertNotIn('agent',app.session_state)


if __name__=='__main__':unittest.main()
