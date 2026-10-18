"""Offline provider tests; external success is opt-in and checked with real sources."""
import json,os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import httpx
from utils import llm_runtime as llm


def settings(values):
    return lambda name,default='':values.get(name,default)


class LlmRuntimeTests(unittest.TestCase):
    def test_groq_reasoning_model_has_compatible_budget_and_deadline(self):
        with patch('importlib.import_module',return_value=SimpleNamespace(ChatGroq=lambda **kw:kw)):
            cfg=llm.build_client('groq',getter=settings({'GROQ_API_KEY':'PRIVATE'}),max_tokens=8)
        self.assertEqual(cfg['model'],'openai/gpt-oss-20b')
        self.assertEqual(cfg['max_tokens'],128);self.assertEqual(cfg['timeout'],60)
        self.assertEqual(cfg['reasoning_effort'],'low');self.assertEqual(cfg['max_retries'],0)

    def test_installed_groq_connector_accepts_reasoning_configuration(self):
        client=llm.build_client('groq',getter=settings({'GROQ_API_KEY':'test-placeholder'}),json_mode=True)
        self.assertEqual(client.reasoning_effort,'low')

    def test_non_reasoning_model_keeps_small_smoke_budget(self):
        with patch('importlib.import_module',return_value=SimpleNamespace(ChatGroq=lambda **kw:kw)):
            cfg=llm.build_client('groq',getter=settings({'GROQ_API_KEY':'PRIVATE','GROQ_MODEL':'test-chat'}),max_tokens=8,timeout=10)
        self.assertEqual(cfg['max_tokens'],8);self.assertEqual(cfg['timeout'],10)

    def test_ollama_cold_start_has_bounded_local_timeout(self):
        with patch('importlib.import_module',return_value=SimpleNamespace(ChatOllama=lambda **kw:kw)):
            cfg=llm.build_client('ollama',getter=settings({}),max_tokens=20,json_mode=True)
        self.assertEqual(cfg['client_kwargs']['timeout'],240)
        self.assertEqual(cfg['num_ctx'],2048);self.assertEqual(cfg['keep_alive'],'5m')
        self.assertEqual(cfg['format'],'json')

    def test_openai_absent_key_does_not_import_optional_sdk(self):
        with patch('importlib.import_module') as module,self.assertRaises(llm.ProviderError):
            llm.build_client('openai',getter=settings({}))
        module.assert_not_called()

    def test_configuration_selection_has_no_runtime_provider_fallback(self):
        self.assertEqual(llm.select_provider(settings({'LLM_PROVIDER':'groq','OPENAI_API_KEY':'x'})),'groq')
        with patch('importlib.import_module',side_effect=RuntimeError('PRIVATE-KEY')) as module:
            with self.assertRaises(llm.ProviderError) as error:
                llm.build_client('groq',getter=settings({'GROQ_API_KEY':'PRIVATE-KEY','OPENAI_API_KEY':'other'}))
        module.assert_called_once_with('langchain_groq')
        self.assertNotIn('PRIVATE-KEY',str(error.exception))

    def test_timeout_returns_safe_error_not_a_false_response(self):
        client=SimpleNamespace(invoke=lambda messages:(_ for _ in ()).throw(httpx.ReadTimeout('PRIVATE-KEY')))
        with self.assertRaises(llm.ProviderError) as error:llm.invoke(client,'hello','groq')
        self.assertIn('Delai',str(error.exception));self.assertNotIn('PRIVATE-KEY',str(error.exception))

    def test_model_not_found_and_auth_errors_distinguished_without_raw_body(self):
        for status,word in ((404,'Modele'),(401,'Authentification'),(429,'Quota')):
            error=RuntimeError('PRIVATE-KEY');error.status_code=status
            self.assertIn(word,llm.explain(error));self.assertNotIn('PRIVATE-KEY',llm.explain(error))

    def test_empty_reasoning_only_result_is_not_a_success(self):
        with self.assertRaisesRegex(llm.ProviderError,'vide'):
            llm.invoke(SimpleNamespace(invoke=lambda messages:SimpleNamespace(content='')),'hello','groq')

    def test_invalid_deadlines_fail_without_network(self):
        for value in ('0','-1','inf','601'):
            with self.subTest(value=value),self.assertRaises(llm.ProviderError):
                llm.build_client('ollama',getter=settings({'OLLAMA_TIMEOUT':value}))

    def test_legacy_provider_and_rag_delegate_to_shared_factory(self):
        from utils import llm_providers,rag_service
        with patch.object(llm_providers,'build_client',return_value='client') as build:
            with patch.object(llm_providers,'_get_secret',side_effect=settings({'LLM_PROVIDER':'groq'})):
                self.assertEqual(llm_providers.load_llm(),'client')
            self.assertEqual(build.call_args.args[0],'groq')
        with patch.object(llm,'build_client',return_value='client') as build:
            self.assertEqual(rag_service.create_client('groq'),'client')
            self.assertTrue(build.call_args.kwargs['json_mode'])

    def test_configuration_badge_does_not_construct_or_call_a_model(self):
        from streamlit.testing.v1 import AppTest
        with patch.object(llm,'build_client') as build:
            page=AppTest.from_string('from utils.llm_status import llm_status_badge\nllm_status_badge()').run()
        self.assertEqual(len(page.exception),0,str(page.exception));build.assert_not_called()
        self.assertTrue(any('non testee' in item.value for item in page.caption))


@unittest.skipUnless(os.getenv('RUN_LLM_EXTERNAL')=='1','Real external LLM tests require explicit RUN_LLM_EXTERNAL=1')
class ExternalLlmTests(unittest.TestCase):
    def test_real_rag_streamlit_grounding(self):
        from utils import rag_service as rag
        from streamlit.testing.v1 import AppTest
        from test_reports import ROOT
        from test_rag import click
        provider=os.getenv('RAG_EXTERNAL_PROVIDER','groq')
        self.assertIn(provider,llm.PROVIDERS)
        page=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=300).run()
        page.switch_page('app/rag_chatbot.py').run()
        page.selectbox[0].set_value(provider)
        for question in ('Que prevoit la circulaire concernant la gestion des risques ?',
                         'Quel ratio de levier est prevu et quelle sera la meteo demain ?'):
            page.text_input[0].set_value(question)
            click(page,'Rechercher sans LLM')
            click(page,'Generer une reponse depuis les passages (appel fournisseur)')
            self.assertEqual(len(page.exception),0,str(page.exception));self.assertEqual(len(page.error),0,str(page.error))
            entry=page.session_state['rag_verified_history'][-1]
            self.assertEqual(entry['generated']['state'],'grounded',entry['generated'])
            self.assertTrue(entry['generated']['used_sources'])
            for quote in entry['generated']['quotes']:
                source=next(hit for hit in entry['retrieved']['hits'] if hit['source_id']==quote['source_id'])
                self.assertIn(quote['quote'],source['text'])
            print('EXTERNAL REAL RAG',question,'sources',entry['generated']['used_sources'],flush=True)
        page.text_input[0].set_value('Quelle est la capitale du Japon et sa population ?')
        with patch.object(rag,'create_client') as client:
            click(page,'Rechercher sans LLM')
            click(page,'Generer une reponse depuis les passages (appel fournisseur)')
            client.assert_not_called()
        self.assertEqual(page.session_state['rag_verified_history'][-1]['generated']['state'],'refused')
        self.assertGreaterEqual(len(page.session_state['rag_verified_history']),3)
        # A timeout returns control; next explicit click can succeed on the same sources.
        page.text_input[0].set_value('Que prevoit la circulaire concernant la gestion des risques ?')
        click(page,'Rechercher sans LLM')
        with patch.object(rag,'create_client',return_value=SimpleNamespace(invoke=lambda msgs:(_ for _ in ()).throw(httpx.ReadTimeout('PRIVATE')))):
            click(page,'Generer une reponse depuis les passages (appel fournisseur)')
        self.assertEqual(page.session_state['rag_verified_history'][-1]['generated']['state'],'unavailable')
        click(page,'Generer une reponse depuis les passages (appel fournisseur)')
        self.assertEqual(page.session_state['rag_verified_history'][-1]['generated']['state'],'grounded')
        self.assertEqual(len(page.exception),0,str(page.exception))
