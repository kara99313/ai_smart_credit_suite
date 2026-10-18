from synthetic_history import SyntheticHistoryTestCase
"""Truthful, isolated probes; no provider calls or index builders on load."""
import io
import os
import pickle
import runpy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import faiss
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from streamlit.testing.v1 import AppTest
from utils import health_diagnostics as h
from test_reports import ROOT,LOG


def fixture_index(root):
    root=Path(root)
    corpus=root/'docs_rag';corpus.mkdir(exist_ok=True)
    (corpus/'reference.txt').write_text('gestion du risque',encoding='utf-8')
    path=root/'vector_store/faiss_docs';path.mkdir(parents=True)
    index=faiss.IndexFlatL2(384)
    index.add(np.array([[0.]*384,[1.]*384],dtype='float32'))
    (path/'index.faiss').write_bytes(faiss.serialize_index(index).tobytes())
    documents={'a':Document(page_content='risque bancaire',metadata={'source':'reference.txt'}),
               'b':Document(page_content='credit',metadata={'source':'reference.txt'})}
    (path/'index.pkl').write_bytes(pickle.dumps((InMemoryDocstore(documents),{0:'a',1:'b'})))
    return path


def page(root):
    script=('import runpy\n'+f"ns=runpy.run_path({str(ROOT/'app/health_check.py')!r},run_name='health_test')\n"
        +f"ns['main'].__globals__['PROJECT_ROOT']={str(root)!r}\nns['main']()\n")
    return AppTest.from_string(script,default_timeout=90)


def button(app,label):
    return next(item for item in app.button if item.label==label)


class DiagnosticsTests(SyntheticHistoryTestCase):
    def test_backend_health_reports_http_payload_and_loaded_not_inference(self):
        with patch.object(h.requests,'get',return_value=SimpleNamespace(status_code=200,json=lambda:{'status':'ok','model_loaded':True})) as get:
            result=h.backend_check('http://127.0.0.1:18000')
        self.assertEqual(result['state'],'Disponible');self.assertEqual(result['http_status'],200)
        self.assertTrue(result['model_loaded']);self.assertIn('checked_at',result)
        self.assertIn('aucune inference',result['explanation'])
        get.assert_called_once_with('http://127.0.0.1:18000/health',timeout=4)

    def test_backend_failures_and_invalid_configuration_never_escape(self):
        for status,body in ((500,{'status':'error'}),(200,{'status':'bad'}),(200,'not a health schema')):
            with patch.object(h.requests,'get',return_value=SimpleNamespace(status_code=status,json=lambda:body)):
                result=h.backend_check('http://127.0.0.1:18000')
                self.assertEqual(result['state'],'Indisponible');self.assertEqual(result['http_status'],status)
        with patch.object(h.requests,'get',side_effect=ConnectionError('key=PRIVATE')):
            self.assertEqual(h.backend_check()['state'],'Indisponible')
        with patch.object(h,'backend_url',side_effect=ValueError('bad')):
            self.assertEqual(h.backend_check()['state'],'Indisponible')

    def test_health_non_json_body_is_not_echoed(self):
        response=SimpleNamespace(status_code=502,json=lambda:(_ for _ in ()).throw(ValueError('SECRET')))
        with patch.object(h.requests,'get',return_value=response):
            result=h.backend_check('http://127.0.0.1:18000')
        self.assertNotIn('SECRET',repr(result));self.assertEqual(result['http_status'],502)

    def test_inference_is_explicit_and_not_history_persistence(self):
        before=LOG.read_bytes()
        with patch.object(h,'predict_credit_api',return_value={'pd':.23,'model_id':'MODEL'}) as predict:
            result=h.inference_check('http://127.0.0.1:18000')
        self.assertEqual(result['state'],'Fonctionnel');self.assertEqual(result['pd'],.23)
        self.assertNotIn('client_id',predict.call_args.args[0])
        self.assertEqual(LOG.read_bytes(),before)
        with patch.object(h,'predict_credit_api',side_effect=RuntimeError('missing model')):
            self.assertEqual(h.inference_check()['state'],'Indisponible')

    def test_provider_inventory_uses_booleans_not_keys_or_readiness(self):
        values={'GROQ_API_KEY':'private-groq-value','OPENAI_API_KEY':''}
        with patch.object(h,'get_setting',side_effect=lambda name,default='':values.get(name,default)):
            rows=h.provider_inventory()
        self.assertEqual([row['provider'] for row in rows],['groq','openai','ollama'])
        self.assertEqual(rows[0]['state'],'Configure seulement');self.assertFalse(rows[1]['key_present'])
        self.assertNotIn('private-groq-value',repr(rows))

    def test_missing_key_never_constructs_connector(self):
        with patch.object(h,'get_setting',return_value=''),patch('importlib.import_module') as module:
            result=h.provider_check('openai')
        self.assertEqual(result['state'],'Indisponible');module.assert_not_called()

    def test_provider_requires_real_invoke_and_bounds_call(self):
        factory=[]
        class Connector:
            def __init__(self,**kwargs):factory.append(kwargs)
            def invoke(self,prompt):return SimpleNamespace(content='OK')
        with patch.object(h,'get_setting',side_effect=lambda name,default='':'private-value' if name.endswith('KEY') else ('test-chat-model' if name=='GROQ_MODEL' else default)),patch('importlib.import_module',return_value=SimpleNamespace(ChatGroq=Connector)):
            result=h.provider_check('groq')
        self.assertEqual(result['state'],'Fonctionnel')
        self.assertEqual(factory[0]['max_tokens'],8);self.assertEqual(factory[0]['timeout'],60)
        self.assertEqual(factory[0]['max_retries'],0);self.assertNotIn('private-value',repr(result))

    def test_provider_external_failure_never_echoes_key(self):
        class Connector:
            def __init__(self,**kwargs):pass
            def invoke(self,prompt):raise RuntimeError('Authorization: private-value')
        with patch.object(h,'get_setting',return_value='private-value'),patch('importlib.import_module',return_value=SimpleNamespace(ChatOpenAI=Connector)):
            result=h.provider_check('openai')
        self.assertEqual(result['state'],'Indisponible');self.assertNotIn('private-value',repr(result))

    def test_redaction_in_body_nested_keys_urls_and_exception_values(self):
        with patch.object(h,'secrets',return_value=['PRIVATE-KEY']):
            value=h.redact({'api_key':'OTHER','nested':[{'message':'PRIVATE-KEY https://user:pass@host/x?token=TOKEN'}]})
        self.assertNotIn('PRIVATE-KEY',repr(value));self.assertNotIn('OTHER',repr(value))
        self.assertNotIn('user:pass',repr(value));self.assertNotIn('=TOKEN',repr(value))

    def test_missing_model_and_environment_resolution(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'MODEL_PATH':'missing.pkl','BACKEND_URL':'http://127.0.0.1:18123','API_BASE':'http://127.0.0.1:18124'}):
            env=h.environment_inventory(tmp)
            self.assertFalse(env['model_readable']);self.assertTrue(env['api_configured'])
            self.assertEqual(env['api_url'],'http://127.0.0.1:18123')
            self.assertIn('BACKEND_URL',env['api_source'])
        with patch.object(h,'backend_url',side_effect=ValueError('bad')):
            self.assertFalse(h.environment_inventory()['api_configured'])

    def test_corpus_absent_and_present_without_extraction_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(h.corpus_inventory(tmp)['state'],'Indisponible')
            fixture_index(tmp)
            result=h.corpus_inventory(tmp)
            self.assertEqual(result['state'],'Disponible');self.assertEqual(result['count'],1)
            self.assertIn('non testee',result['explanation'])

    def test_real_faiss_load_validates_documents_mapping_and_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=fixture_index(tmp)
            before={p.name:p.read_bytes() for p in path.iterdir()}
            result=h.index_check(path,tmp)
            self.assertEqual(result['state'],'Disponible');self.assertEqual(result['vectors'],2)
            self.assertEqual(result['dimension'],384);self.assertEqual(result['missing_sources'],0)
            self.assertIn('A verifier',result['consistency'])
            self.assertEqual(before,{p.name:p.read_bytes() for p in path.iterdir()})
            (Path(tmp)/'docs_rag/reference.txt').unlink()
            self.assertEqual(h.index_check(path,tmp)['missing_sources'],2)

    def test_absent_corrupt_and_incomplete_faiss_are_isolated(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(h.index_check(Path(tmp)/'absent',tmp)['state'],'Indisponible')
            path=fixture_index(tmp)
            (path/'index.faiss').write_bytes(b'invalid')
            self.assertEqual(h.index_check(path,tmp)['state'],'Indisponible')

    def test_restricted_docstore_rejects_executable_pickle(self):
        class Dangerous:
            def __reduce__(self):return (os.system,('echo should-not-run',))
        import os
        payload=pickle.dumps(Dangerous())
        with self.assertRaises(pickle.UnpicklingError):
            h._DocstoreReader(io.BytesIO(payload)).load()

    def test_retrieval_uses_real_vector_search_with_mocked_offline_encoder(self):
        class Encoder:
            def __init__(self,*args,**kwargs):
                assert kwargs['local_files_only'] is True
            def encode(self,*args,**kwargs):
                assert kwargs['normalize_embeddings'] is True
                return np.zeros((1,384),dtype='float32')
        with tempfile.TemporaryDirectory() as tmp:
            path=fixture_index(tmp)
            with patch.dict('sys.modules',{'sentence_transformers':SimpleNamespace(SentenceTransformer=Encoder)}):
                result=h.retrieval_check(path)
            self.assertEqual(result['state'],'Fonctionnel');self.assertEqual(result['hits'],2)
            with patch.dict('sys.modules',{'sentence_transformers':SimpleNamespace(SentenceTransformer=lambda *a,**kw:(_ for _ in ()).throw(OSError('cache missing')))}):
                result=h.retrieval_check(path)
            self.assertEqual(result['state'],'Indisponible');self.assertIn('Aucun telechargement',result['explanation'])


class HealthPageTests(SyntheticHistoryTestCase):
    def test_open_from_navigation_has_no_automatic_network_llm_or_index_load(self):
        with patch.object(h,'backend_check') as api,patch.object(h,'provider_check') as llm,patch.object(h,'index_check') as index,patch.object(h,'retrieval_check') as retrieval,patch.object(h,'inference_check') as inference,patch.object(h.requests,'get') as get:
            app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=90).run()
            app.switch_page('app/health_check.py').run()
            self.assertEqual(len(app.exception),0,str(app.exception));self.assertEqual(app.title[0].value,'Health Check')
            for probe in (api,llm,index,retrieval,inference,get):probe.assert_not_called()

    def test_partial_failures_do_not_prevent_other_controls(self):
        with tempfile.TemporaryDirectory() as tmp:
            app=page(tmp).run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            self.assertTrue(any('absent ou inaccessible' in item.value for item in app.warning))
            with patch.object(h,'backend_check',return_value=h.result('Indisponible','backend arrete',http_status=None,model_loaded=None)):
                button(app,'Tester GET /health').click().run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            button(app,'Charger et verifier l index existant').click().run()
            self.assertEqual(len(app.exception),0,str(app.exception));self.assertEqual(len(app.error),0,str(app.error))

    def test_api_http_information_display_does_not_test_inference(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(h,'backend_check',return_value=h.result('Disponible','GET reussi',http_status=200,payload={'status':'ok','model_loaded':True},model_loaded=True)),patch.object(h,'inference_check') as inference:
            app=page(tmp).run();button(app,'Tester GET /health').click().run()
            inference.assert_not_called()
            self.assertEqual(len(app.exception),0,str(app.exception))
            self.assertTrue(any('model_loaded=True' in item.value for item in app.markdown))
            self.assertTrue(any('Inference du modele : Non teste' in item.value for item in app.markdown))

    def test_keys_never_in_rendered_page_or_provider_failure(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'GROQ_API_KEY':'PRIVATE-KEY-DO-NOT-PRINT'}):
            app=page(tmp).run()
            app.selectbox[0].set_value('groq')
            with patch.object(h,'provider_check',return_value=h.result('Indisponible','echec sans detail sensible')):
                button(app,'Tester le fournisseur selectionne (appel reel)').click().run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            texts=' '.join(str(getattr(element,'value','')) for element in app)
            self.assertNotIn('PRIVATE-KEY-DO-NOT-PRINT',texts)
            self.assertNotIn('PRIVATE-KEY-DO-NOT-PRINT',str(app.dataframe[0].value))

    def test_bad_api_config_disables_api_tests_only(self):
        with tempfile.TemporaryDirectory() as tmp,patch('utils.runtime_config.backend_url',side_effect=ValueError('invalid')):
            # health page imports resolution freshly; diagnostics independently reports invalidity.
            app=page(tmp).run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            self.assertTrue(button(app,'Tester GET /health').disabled)
            self.assertFalse(button(app,'Charger et verifier l index existant').disabled)
