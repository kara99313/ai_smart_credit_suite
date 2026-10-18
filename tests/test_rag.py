"""Read-only retrieval, current history, grounding, metadata and Streamlit RAG."""
import hashlib, json, pickle, runpy, tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from langchain_core.documents import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from streamlit.testing.v1 import AppTest
from utils import rag_service as r
from test_health_check import fixture_index
from test_client_dashboard import fixtures
from test_reports import ROOT,LOG


class Encoder:
    def encode(self,questions,normalize_embeddings):
        assert normalize_embeddings
        return np.zeros((len(questions),384),dtype='float32')


def index_fixture(root):
    path=fixture_index(root)
    docs={'a':Document(page_content='La gestion des risques impose une gouvernance fiable et une politique documentee.',metadata={'source':'reference.txt'}),
          'b':Document(page_content='Le ratio de levier complete les exigences de fonds propres et limite le risque.',metadata={'source':'reference.txt'})}
    (path/'index.pkl').write_bytes(pickle.dumps((InMemoryDocstore(docs),{0:'a',1:'b'})))
    hybrid=path.parent/'faiss_hybrid';hybrid.mkdir()
    for file in path.iterdir():(hybrid/file.name).write_bytes(file.read_bytes())
    fixtures(Path(root)/'data.csv') # copy to actual expected live history location
    (Path(root)/'data').mkdir()
    for file in Path(root).glob('data*.*'):
        (Path(root)/'data'/file.name.replace('data','predictions_log',1)).write_bytes(file.read_bytes())
    return path


def page(root):
    script=('import runpy\n'+f"ns=runpy.run_path({str(ROOT/'app/rag_chatbot.py')!r},run_name='rag_test')\n"
        +f"ns['main'].__globals__['PROJECT_ROOT']={str(root)!r}\nns['main']()\n")
    return AppTest.from_string(script,default_timeout=90)


def click(app,label):return next(button for button in app.button if button.label==label).click().run()


class RagTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name);self.path=index_fixture(self.root)
        self.patch=patch.object(r,'encoder',return_value=Encoder());self.patch.start()
    def tearDown(self):
        self.patch.stop();self.temp.cleanup()

    def test_docs_retrieval_has_real_distance_and_exact_source_metadata(self):
        result=r.retrieve('  GESTION   des RISQUES  ',root=self.root)
        self.assertEqual(result['question'],'GESTION des RISQUES')
        self.assertEqual(len(result['hits']),1)
        hit=result['hits'][0]
        self.assertEqual(hit['source'],'reference.txt');self.assertEqual(hit['distance'],0)
        self.assertNotIn('page',hit['metadata'])
        self.assertEqual(hit['excerpt_sha256'],hashlib.sha256(hit['text'].encode()).hexdigest())

    def test_out_of_corpus_refuses_before_any_llm_call(self):
        result=r.retrieve('Quelle est la capitale du Japon et sa population ?',root=self.root)
        self.assertEqual(result['hits'],[])
        with patch.object(r,'create_client') as client:
            answer=r.answer(result)
        self.assertEqual(answer['state'],'refused');client.assert_not_called()

    def test_partial_coverage_does_not_invent_missing_part(self):
        result=r.retrieve('Gestion des risques et meteo demain ?',root=self.root)
        with patch.object(r,'create_client',return_value=SimpleNamespace(invoke=lambda messages:SimpleNamespace(content='{"selected":["D0:0"]}'))):
            answer=r.answer(result,'ollama')
        self.assertEqual(answer['state'],'grounded')
        self.assertIn('indeterminees',answer['answer'])
        self.assertNotIn('soleil',answer['answer'])

    def test_grounded_answer_selects_only_actual_literal_evidence(self):
        result=r.retrieve('Gestion des risques',root=self.root)
        with patch.object(r,'create_client',return_value=SimpleNamespace(invoke=lambda messages:SimpleNamespace(content='{"selected":["D0:0"]}'))):
            answer=r.answer(result)
        self.assertEqual(answer['used_sources'],['D0'])
        self.assertIn(result['hits'][0]['text'],answer['answer'])

    def test_invented_ids_and_unvalidated_free_text_are_rejected(self):
        result=r.retrieve('Gestion des risques',root=self.root)
        for content in ('{"selected":["D999:0"]}','The official threshold is 0.5','{"answer":"fake","selected":[]}'):
            with self.subTest(content=content),patch.object(r,'create_client',return_value=SimpleNamespace(invoke=lambda messages:SimpleNamespace(content=content))):
                self.assertEqual(r.answer(result)['state'],'unavailable')

    def test_false_quote_or_missing_page_reference_is_never_validated(self):
        result=r.retrieve('Gestion des risques',root=self.root)
        with self.assertRaises(ValueError):
            r.validate_quotes('{"quotes":[{"source_id":"D0","quote":"A completely invented regulatory threshold."}]}',result['hits'])

    def test_provider_error_has_no_secret_and_preserves_passages(self):
        result=r.retrieve('Gestion des risques',root=self.root)
        with patch.object(r,'create_client',side_effect=RuntimeError('SECRET-KEY')):
            answer=r.answer(result)
        self.assertEqual(answer['state'],'unavailable')
        self.assertNotIn('SECRET-KEY',repr(answer));self.assertEqual(len(result['hits']),1)

    def test_hybrid_uses_fresh_archive_values_without_deriving_business_results(self):
        result=r.retrieve('Historique des predictions du client A','hybrid',root=self.root)
        history=[hit for hit in result['hits'] if hit['kind']=='history']
        self.assertEqual(len(history),3)
        payload=json.loads(history[0]['text'])
        self.assertEqual(payload['decision'],'ACCEPT');self.assertEqual(payload['score_1000'],456)
        self.assertEqual(payload['rating'],'BB');self.assertEqual(payload['threshold'],.8)
        invalid=json.loads(history[-1]['text'])
        self.assertIsNone(invalid['prob_default']);self.assertEqual(invalid['pd_status'],'indisponible ou invalide')

    def test_unknown_client_never_gets_another_clients_history(self):
        result=r.retrieve('Historique du client UNKNOWN','hybrid',root=self.root)
        self.assertFalse(any(hit['kind']=='history' for hit in result['hits']))

    def test_missing_sources_are_not_silently_used(self):
        (self.root/'docs_rag/reference.txt').unlink()
        result=r.retrieve('Gestion des risques',root=self.root)
        self.assertEqual(result['hits'],[]);self.assertGreater(result['skipped_sources'],0)

    def test_source_outside_corpus_is_rejected_even_if_file_exists(self):
        (self.root/'outside.txt').write_text('outside corpus',encoding='utf-8')
        self.assertFalse(r.source_available(self.root,'../outside.txt'))
        self.assertFalse(r.source_available(self.root,str(self.root/'outside.txt')))
        self.assertTrue(r.source_available(self.root,'reference.txt'))

    def test_no_reconstruction_even_when_missing_or_explicit_rebuild_requested(self):
        from utils import rag_utils
        with patch.object(rag_utils,'PROJECT_ROOT',self.root),patch.object(rag_utils,'_build_faiss') as build:
            store,_=rag_utils.build_or_load_faiss_docs()
            self.assertEqual(len(store.similarity_search('Gestion des risques')),1)
            with self.assertRaises(RuntimeError):rag_utils.build_or_load_faiss_docs(rebuild=True)
            build.assert_not_called()
        with self.assertRaises(FileNotFoundError):r.load_index('docs',self.root/'missing')

    def test_mode_switch_loads_correct_index_instead_of_reusing_old_session(self):
        docs=r.retrieve('Gestion des risques','docs',root=self.root)
        hybrid=r.retrieve('Historique des predictions','hybrid',root=self.root)
        self.assertEqual(docs['mode'],'docs');self.assertEqual(hybrid['mode'],'hybrid')
        self.assertTrue(any(hit['kind']=='history' for hit in hybrid['hits']))

    def test_context_bridge_does_not_fabricate_missing_decision_or_pd(self):
        from utils import context_bridge as c
        with patch.object(c,'LOG_PATH',str(self.root/'data/predictions_log.csv')):
            context,row=c.build_client_context('A')
            self.assertIn('ACCEPT',context);self.assertEqual(row['score_1000'],456)
            self.assertIn('Valeurs archivees',c.build_logs_summary())

    def test_protected_files_stay_byte_for_byte_unchanged(self):
        before={path:path.read_bytes() for path in self.root.rglob('*') if path.is_file()}
        r.retrieve('Gestion des risques',root=self.root)
        r.retrieve('Historique des predictions','hybrid',root=self.root)
        self.assertEqual(before,{path:path.read_bytes() for path in self.root.rglob('*') if path.is_file()})

    def test_bad_question_and_k_fail_explicitly(self):
        for question,k in (('',3),('x'*2001,3),('question',0),('question',7),('question',True)):
            with self.assertRaises(ValueError):r.retrieve(question,top_k=k,root=self.root)

    def test_streamlit_open_does_not_load_encoder_or_call_llm(self):
        with patch.object(r,'encoder') as encoder,patch.object(r,'create_client') as client:
            app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=90).run()
            app.switch_page('app/rag_chatbot.py').run()
            self.assertEqual(len(app.exception),0,str(app.exception))
            encoder.assert_not_called();client.assert_not_called()

    def test_streamlit_search_generation_sources_history_and_provider_switch(self):
        app=page(self.root).run()
        app.text_input[0].set_value('Gestion des risques')
        click(app,'Rechercher sans LLM')
        self.assertEqual(len(app.exception),0,str(app.exception))
        self.assertTrue(any('reference.txt' in item.value for item in app.markdown))
        app.selectbox[0].set_value('ollama')
        with patch.object(r,'create_client',return_value=SimpleNamespace(invoke=lambda messages:SimpleNamespace(content='{"selected":["D0:0"]}'))):
            click(app,'Generer une reponse depuis les passages (appel fournisseur)')
        self.assertEqual(len(app.exception),0,str(app.exception))
        self.assertTrue(any('Extraits verifies' in item.value for item in app.markdown))
        app.text_input[0].set_value('Quelle est la capitale du Japon ?')
        click(app,'Rechercher sans LLM')
        self.assertTrue(any(r.REFUSAL in item.value for item in app.info))
        self.assertTrue(any('Historique conversationnel' in item.label for item in app.expander))

    def test_streamlit_provider_failure_and_missing_index_do_not_crash(self):
        app=page(self.root).run();app.text_input[0].set_value('Gestion des risques')
        click(app,'Rechercher sans LLM')
        with patch.object(r,'create_client',side_effect=RuntimeError('SECRET-KEY')):
            click(app,'Generer une reponse depuis les passages (appel fournisseur)')
        self.assertEqual(len(app.exception),0,str(app.exception))
        self.assertNotIn('SECRET-KEY',' '.join(str(item.value) for item in app.markdown))
        self.assertTrue(any('reference.txt' in item.value for item in app.markdown))
        (self.path/'index.faiss').unlink()
        click(app,'Rechercher sans LLM')
        self.assertEqual(len(app.exception),0,str(app.exception));self.assertTrue(app.error)


@unittest.skipUnless(__import__('os').getenv('RUN_RUNTIME_SMOKE')=='1','Real local corpus/embedding checks are opt-in')
class RealRagTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        paths=[ROOT/'data/predictions_log.csv',ROOT/'model/logistic_pipeline_best.pkl']
        paths+=list((ROOT/'vector_store').rglob('*'))+list((ROOT/'docs_rag').rglob('*'))
        from utils.prediction_logs import snapshot_path
        paths.append(snapshot_path(ROOT/'data/predictions_log.csv'))
        cls.before={path:path.read_bytes() if path.is_file() else None for path in paths if path.is_file() or path.suffix=='.jsonl'}

    @classmethod
    def tearDownClass(cls):
        for path,data in cls.before.items():
            assert (path.read_bytes() if path.exists() else None)==data, f'Protected artifact changed: {path.name}'

    def test_actual_docs_hybrid_partial_and_outside_questions(self):
        covered=r.retrieve('Que prevoit la circulaire concernant la gestion des risques ?')
        self.assertEqual(len(covered['hits']),3)
        self.assertTrue(all('CIRCULAIRE' in hit['source'] for hit in covered['hits']))
        self.assertTrue(all(np.isfinite(hit['distance']) for hit in covered['hits']))
        partial=r.retrieve('Quel ratio de levier est prevu et quelle sera la meteo demain ?')
        self.assertTrue(partial['hits'])
        self.assertTrue(all('levier' in hit['text'].lower() for hit in partial['hits']))
        outside=r.retrieve('Quelle est la capitale du Japon et sa population ?')
        self.assertEqual(outside['hits'],[])
        hybrid=r.retrieve('Quelles sont les dernieres predictions de l historique ?','hybrid')
        self.assertTrue(any(hit['kind']=='history' for hit in hybrid['hits']))
        self.assertTrue(all(hit['source']!='predictions_log.csv' for hit in hybrid['hits']))
        print('REAL RAG: covered 3, partial supported evidence, outside 0, current-history retrieval verified',flush=True)

    def test_actual_streamlit_search_keeps_sources_when_generation_fails(self):
        app=AppTest.from_file(str(ROOT/'streamlit_app.py'),default_timeout=180).run()
        app.switch_page('app/rag_chatbot.py').run()
        app.text_input[0].set_value('Que prevoit la circulaire concernant la gestion des risques ?')
        click(app,'Rechercher sans LLM')
        self.assertEqual(len(app.exception),0,str(app.exception));self.assertEqual(len(app.error),0,str(app.error))
        self.assertTrue(any('CIRCULAIRE' in item.value for item in app.markdown))
        with patch.object(r,'create_client',side_effect=RuntimeError('provider outage')):
            click(app,'Generer une reponse depuis les passages (appel fournisseur)')
        self.assertEqual(len(app.exception),0,str(app.exception))
        self.assertTrue(any('passages recuperes restent consultables' in item.value for item in app.markdown))
        self.assertTrue(any('CIRCULAIRE' in item.value for item in app.markdown))
