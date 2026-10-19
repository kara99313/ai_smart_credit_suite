"""Phase 11: tool evidence, provider failures and Streamlit chat paths."""
import json
from utils.credit_agent_contract import FEATURES
CREDIT_JSON = json.dumps(dict.fromkeys(FEATURES, 1))
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch, Mock
from streamlit.testing.v1 import AppTest
from utils import agents, agent_tools as tools


class AgentTests(unittest.TestCase):
    def test_failed_tool_cannot_be_replaced_by_invented_success(self):
        executor=Mock()
        executor.invoke.return_value={'output':'PD 0%, ACCEPT', 'intermediate_steps':[(NS(tool='predict_credit'),'ERROR: API indisponible')]}
        out=agents.run_agent_once(executor,'score '+CREDIT_JSON,[])
        self.assertNotIn('ACCEPT',out['output'])
        self.assertEqual(out['intermediate_steps'][0]['status'], 'error')

    def test_success_uses_exact_observation(self):
        executor=Mock(invoke=Mock(return_value={'output':'invented','intermediate_steps':[(NS(tool='now'),'2026-10-18 UTC')]}))
        self.assertIn('2026-10-18 UTC',agents.run_agent_once(executor,'heure',[])['output'])

    def test_no_tools_is_explicit(self):
        out=agents.run_agent_once(Mock(invoke=Mock(return_value={'output':'Bonjour'})),'salut',[])
        self.assertIn('aucun resultat operationnel confirme',out['output'])

    def test_empty_response_rejected(self):
        with self.assertRaises(RuntimeError):
            agents.run_agent_once(Mock(invoke=Mock(return_value={'output':''})),'salut',[])

    def test_timeout_sanitized(self):
        with self.assertRaisesRegex(RuntimeError,'Delai'):
            agents.run_agent_once(Mock(invoke=Mock(side_effect=TimeoutError('SECRET'))),'x',[])

    def test_provider_failure_sanitized(self):
        with self.assertRaises(RuntimeError) as result:
            agents.run_agent_once(Mock(invoke=Mock(side_effect=RuntimeError('SECRET'))),'x',[])
        self.assertNotIn('SECRET',str(result.exception))

    def test_prediction_malformed_not_default_zero(self):
        with patch.object(tools,'predict_credit_api',return_value={}):
            result=tools.predict_credit_tool(CREDIT_JSON)
        self.assertNotIn('0.00%',result)

    def test_prediction_valid_result(self):
        response=dict(pd=.3,score_1000=700,rating='CCC',decision='REVIEW/REJECT',threshold=.1,used_features=['Income'])
        with patch.object(tools,'predict_credit_api',return_value=response):
            self.assertIn('30.00%',tools.predict_credit_tool(CREDIT_JSON))

    def test_prediction_timeout(self):
        with patch.object(tools,'predict_credit_api',side_effect=TimeoutError('SECRET')):
            result=tools.predict_credit_tool(CREDIT_JSON)
        self.assertIn('Delai',result);self.assertNotIn('SECRET',result)

    def test_report_no_archive_never_defaults_pd(self):
        with patch.object(tools,'load_client_record',return_value=None):
            self.assertIn('ERROR',tools.generate_report_tool('{"pd_value":0}'))

    def test_report_uses_archive_not_model_values(self):
        row=dict(client_id='C1',prob_default=.3,threshold=.1,inputs_json='{}',decision='REVIEW/REJECT')
        with patch.object(tools,'load_client_record',return_value=row):
            result=tools.generate_report_tool('{"client_id":"C1","pd_value":0}')
        self.assertIn('30.00%',result);self.assertIn('</html>',result)

    def test_web_unavailable(self):
        with patch.object(tools,'search_web',side_effect=TimeoutError('SECRET')):
            result=tools.tool_web_search('x')
        self.assertIn('indisponible',result);self.assertNotIn('SECRET',result)

    def test_web_removed_from_executor(self):
        from utils import multitask_agent as m
        with patch.object(m,'load_llm'),patch.object(m,'create_tool_calling_agent'),patch.object(m,'AgentExecutor') as executor:
            m.create_multitask_agent(use_web=False,system_prompt='English')
        self.assertNotIn('search_web',[t.name for t in executor.call_args.kwargs['tools']])
        self.assertEqual(executor.call_args.kwargs['max_execution_time'],90)

    def test_assistant_ui_normal_and_reset(self):
        with patch('utils.llm_status.llm_status_badge'),patch('utils.agents.create_credit_agent'),patch('utils.agents.run_agent_once',return_value={'output':'Observation verifiee'}):
            app=AppTest.from_file('app/chatbot_assistant.py',default_timeout=30).run()
            app.chat_input[0].set_value('Bonjour').run()
            self.assertFalse(app.exception)
            self.assertIn('Observation verifiee',' '.join(m.value for m in app.markdown))
            app.chat_input[0].set_value('/reset').run()
            self.assertEqual(app.session_state['assistant_msgs'],[])

    def test_multitask_ui_error_and_navigation_history(self):
        with patch('utils.llm_status.llm_status_badge'),patch('utils.multitask_agent.create_multitask_agent',side_effect=ImportError('SECRET')):
            app=AppTest.from_file('app/agent.py',default_timeout=30).run()
            app.chat_input[0].set_value('Calculer').run()
            self.assertFalse(app.exception)
            self.assertNotIn('SECRET',' '.join(m.value for m in app.markdown))
            self.assertEqual(len(app.session_state['agent_history']),2)

    def test_web_html_fallback_returns_dicts(self):
        from utils import web_search as web
        response=Mock(text='<a class="result__a" href="https://example.org">Title</a><a class="result__snippet">Text</a>')
        with patch.object(web.requests,'post',return_value=response):
            hits=web._search_web_ddg_html('query',2)
        self.assertEqual(hits[0]['href'],'https://example.org')

    def test_rag_adapter_unavailable_without_rebuild(self):
        with patch.object(tools,'build_or_load_faiss_docs',side_effect=FileNotFoundError()) as loader:
            result=tools.rag_retrieve_tool('risque')
        loader.assert_called_once_with(rebuild=False)
        self.assertIn('indisponible',result)

    def test_simple_chat_keeps_history_and_handles_web_failure(self):
        client=Mock(invoke=Mock(return_value=NS(content='Reponse simple')))
        with patch('utils.llm_status.llm_status_badge'),patch('utils.llm_providers.load_llm',return_value=client),patch('utils.web_search.search_web',side_effect=TimeoutError()):
            app=AppTest.from_file('app/chatbot_assistant.py',default_timeout=30).run()
            app.checkbox[1].uncheck().run()
            app.chat_input[0].set_value('Bonjour').run()
            app.chat_input[0].set_value('Et ensuite ?').run()
            self.assertFalse(app.exception)
            self.assertTrue(app.warning)
            messages=client.invoke.call_args.args[0]
            self.assertIn({'role':'assistant','content':'Reponse simple'},messages)

    def test_simple_chat_provider_timeout_displayed(self):
        client=Mock(invoke=Mock(side_effect=TimeoutError('SECRET')))
        with patch('utils.llm_status.llm_status_badge'),patch('utils.llm_providers.load_llm',return_value=client):
            app=AppTest.from_file('app/chatbot_assistant.py',default_timeout=30).run()
            app.checkbox[0].uncheck().run()
            app.checkbox[1].uncheck().run()
            app.chat_input[0].set_value('Bonjour').run()
            self.assertFalse(app.exception)
            text=' '.join(m.value for m in app.markdown)
            self.assertIn('Delai',text);self.assertNotIn('SECRET',text)

if __name__=='__main__':unittest.main()
