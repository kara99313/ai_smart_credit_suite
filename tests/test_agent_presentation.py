"""Lot 3A: evidence presentation and persistent artifacts on both interfaces."""
import json
from utils.credit_agent_contract import FEATURES
CREDIT_JSON = json.dumps(dict.fromkeys(FEATURES, 1))
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch
import pytest
from streamlit.testing.v1 import AppTest
from utils import agents
from utils import web_search as web

HTML = '<!DOCTYPE html><html><body>' + 'REPORT_PRIVATE ' * 500 + '</body></html>'


def execute(steps, output='Succ?s invent?'):
    return agents.run_agent_once(Mock(invoke=Mock(return_value=dict(output=output, intermediate_steps=steps))), 'rapport', [])


def test_report_separation_and_full_download():
    out = execute([(NS(tool='build_report', tool_input=HTML), HTML)])
    assert out['artifacts'][0]['data'] == HTML
    assert 'REPORT_PRIVATE' not in out['output']
    assert 'REPORT_PRIVATE' not in str(out['intermediate_steps'])
    assert 'généré' in out['output']


@pytest.mark.parametrize('error', ['ERROR: indisponible', '? ?chec API', '', 'HTML incomplet <html>'])
def test_report_failure_has_no_artifact(error):
    out = execute([(NS(tool='build_report'), error)])
    assert not out['artifacts']
    assert 'Échec' in out['output']
    assert 'Succ?s invent?' not in out['output']


def test_technical_observations_bounded():
    out = execute([(NS(tool='search_web', tool_input='q' * 5000), 'x' * 20000)])
    assert len(out['intermediate_steps'][0]['observation']) <= 2000
    assert len(out['intermediate_steps'][0]['tool_input']) <= 1000
    assert out['output'] != out['intermediate_steps'][0]['observation']


@pytest.mark.parametrize('file,namespace,history', [('app/agent.py', 'agent', 'agent_history'), ('app/chatbot_assistant.py', 'assistant', 'assistant_msgs')])
def test_interfaces_reports_survive_rerun_without_history_html(file, namespace, history):
    executor = Mock(invoke=Mock(return_value=dict(output=HTML, intermediate_steps=[(NS(tool='build_report'), HTML)])))
    with patch('utils.llm_status.llm_status_badge'), patch('utils.multitask_agent.create_multitask_agent', return_value=executor):
        app = AppTest.from_file(file, default_timeout=30).run()
        for _ in range(2):
            app.chat_input[0].set_value('rapport').run()
            assert not app.exception
        app.run()
        assert not app.exception
        results = app.session_state[namespace + '_results']
        artifacts = [a for r in results for a in r['artifacts']]
        assert len(artifacts) == 2
        assert len({a['id'] for a in artifacts}) == 2
        assert all(a['data'] == HTML for a in artifacts)
        assert 'REPORT_PRIVATE' not in str(app.session_state[history])
        assert 'REPORT_PRIVATE' not in ' '.join(m.value for m in app.markdown)
        buttons = app.get('download_button')
        assert len(buttons) == 2
        assert len({b.proto.id for b in buttons}) == 2
        executor.invoke.return_value = dict(output='Rapport g?n?r?', intermediate_steps=[(NS(tool='build_report'), 'ERROR: rapport indisponible')])
        app.chat_input[0].set_value('autre rapport').run()
        assert not app.exception
        assert len(app.get('download_button')) == 2
        assert 'Échec' in app.session_state[history][-1][1]
        assert 'REPORT_PRIVATE' not in str(executor.invoke.call_args)


def test_web_direct_fetch_is_disabled():
    page = '<html><header>MENU</header><nav>NAV</nav><div class="newsletter">SUBSCRIBE</div><main><article><h1>Titre utile</h1><p>Texte utile.</p></article></main><footer>FOOTER</footer><script>JS</script></html>'
    with patch.object(web, 'web_configuration', return_value=dict(enabled=True, domains=[], timeout=2)), patch.object(web.requests, 'get', return_value=Mock(status_code=200, text=page, content=page.encode(), headers={})):
        text = web.fetch_and_clean('https://example.org')
    assert text == ''


@pytest.mark.parametrize('full', ['', 'Menu seulement', 'Access denied ' * 30])
def test_web_fallback_and_metadata_budget(full):
    hits = [dict(title='Titre ' + str(i), href='https://example.org/' + str(i), body='Extrait fournisseur ' * 100) for i in range(3)]
    with patch.object(web, 'fetch_and_clean', return_value=full):
        text = web.build_web_context(hits, max_chars_total=220, fetch_full=True)
    for i, hit in enumerate(hits, 1):
        assert hit['title'] in text and hit['href'] in text and f'[W{i}]' in text
    assert 'Extrait' in text
    assert len(text) <= 220


def test_web_ignores_full_content_and_uses_provider_snippet():
    with patch.object(web, 'fetch_and_clean', return_value='Contenu nettoy? pertinent ' * 30):
        text = web.build_web_context([dict(title='Titre', href='https://example.org', body='Fournisseur')], fetch_full=True)
    assert 'Contenu nettoy?' not in text
    assert 'Fournisseur' in text


@pytest.mark.parametrize('fetch', [False, True])
@pytest.mark.parametrize('pollution', ['%PDF-1.7 endobj ViewerPreferences', 'Home Menu Contact Login', '<div>broken HTML</div>'])
def test_agent_web_presentation_rejects_unsafe_evidence(fetch, pollution):
    hit = dict(title='Rapport BCEAO', href='https://example.org/rapport', body=pollution)
    with patch.object(web, 'fetch_and_clean', return_value=pollution):
        context = web.build_web_context([hit], fetch_full=fetch)
    out = execute([(NS(tool='search_web'), context)], output=context)
    presented = str(out)
    assert '[W1]' in presented
    assert hit['title'] in presented and hit['href'] in presented
    assert 'Aucun extrait exploitable' in presented
    for marker in ('%PDF', 'endobj', 'ViewerPreferences', 'Home Menu', '<div>'):
        assert marker not in presented


WEB = '[W1] Source one\nhttps://example.org/one\nTECHNICAL_EXCERPT_ONE\n\n[W2] Source two\nhttps://example.org/two\nTECHNICAL_EXCERPT_TWO'
FINAL = 'DISTINCTIVE_FINAL: analysis supported by [W1] and [W2].'


def test_web_final_and_verified_sources_separated():
    result = execute([(NS(tool='search_web'), WEB)], FINAL)
    assert FINAL in result['output']
    for marker in ('[W1]', '[W2]', 'Source one', 'Source two', 'https://example.org/one', 'https://example.org/two'):
        assert marker in result['output']
    assert 'TECHNICAL_EXCERPT' not in result['output']
    assert 'TECHNICAL_EXCERPT' in str(result['intermediate_steps'])
    assert result['intermediate_steps'][0]['status'] == 'success'


@pytest.mark.parametrize('final', ['Invented [W99]', 'Invented https://fake.example', '', 'Agent stopped due to iteration limit or time limit.'])
def test_invalid_web_final_is_conservative(final):
    result = execute([(NS(tool='search_web'), WEB)], final)
    assert 'Invented' not in result['output']
    assert 'TECHNICAL_EXCERPT' not in result['output']
    assert '[W1]' in result['output']
    assert result['intermediate_steps'][0]['status'] == 'success'


@pytest.mark.parametrize('observation,status', [
    ('Recherche Web indisponible', 'error'),
    ('Aucun resultat Web', 'no_result'),
    ('unknown format', 'uncertain'),
    ('[W1] Source one\nhttps://example.org/one', 'no_result'),
    (WEB + '\n\n[W3] Empty\nhttps://example.org/three', 'partial'),
])
def test_insufficient_web_does_not_use_model_final(observation, status):
    result = execute([(NS(tool='search_web'), observation)], FINAL)
    assert FINAL not in result['output']
    assert result['intermediate_steps'][0]['status'] == status
    assert 'TECHNICAL_EXCERPT' not in result['output']


@pytest.mark.parametrize('observation', ['Format JSON invalide pour predict_credit', 'Echec appel /api/predict', 'unknown response'])
def test_prediction_text_failure_never_succeeds(observation):
    result = execute([(NS(tool='predict_credit'), observation)], 'SUCCESS PD 0% ACCEPT')
    assert 'ACCEPT' not in result['output']
    assert result['intermediate_steps'][0]['status'] != 'success'


def test_prediction_exact_values_override_model():
    from utils import agent_tools
    response = dict(pd=.3, score_1000=700, rating='CCC', decision='REVIEW/REJECT', threshold=.1, used_features=['Income'])
    with patch.object(agent_tools, 'predict_credit_api', return_value=response):
        observation = agent_tools.predict_credit_tool(CREDIT_JSON)
    result = execute([(NS(tool='predict_credit'), observation)], 'PD 0% ACCEPT')
    assert result['output'] == observation
    assert result['intermediate_steps'][0]['status'] == 'success'


@pytest.mark.parametrize('file,history', [('app/agent.py', 'agent_history'), ('app/chatbot_assistant.py', 'assistant_msgs')])
def test_clean_web_history(file, history):
    executor = Mock(invoke=Mock(return_value=dict(output=FINAL, intermediate_steps=[(NS(tool='search_web', tool_input='PRIVATE_QUERY'), WEB)])))
    with patch('utils.llm_status.llm_status_badge'), patch('utils.multitask_agent.create_multitask_agent', return_value=executor):
        app = AppTest.from_file(file, default_timeout=30).run()
        app.chat_input[0].set_value('web').run()
        assert not app.exception
        saved = str(app.session_state[history])
        assert FINAL in saved
        assert 'TECHNICAL_EXCERPT' not in saved
        assert 'PRIVATE_QUERY' not in saved


def test_real_executor_final_and_presentation_are_distinct():
    from langchain.agents import AgentExecutor
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.runnables import RunnableLambda
    from langchain_core.tools import StructuredTool
    calls = []
    def search():
        calls.append('search')
        return WEB
    def plan(payload):
        if payload['intermediate_steps']:
            return AgentFinish(return_values={'output': FINAL}, log='')
        return AgentAction(tool='search_web', tool_input={}, log='')
    executor = AgentExecutor(agent=RunnableLambda(plan), stream_runnable=False,
        tools=[StructuredTool.from_function(search, name='search_web', description='deterministic')], return_intermediate_steps=True)
    raw = executor.invoke({'input': 'web', 'chat_history': []})
    assert raw['output'] == FINAL
    assert raw['intermediate_steps'][0][1] == WEB
    presented = agents._present_result(raw)
    assert FINAL in presented['output']
    assert 'TECHNICAL_EXCERPT' not in presented['output']
    assert 'TECHNICAL_EXCERPT' in str(presented['intermediate_steps'])
    assert calls == ['search']


@pytest.mark.parametrize('final', [WEB, 'TECHNICAL_EXCERPT_ONE', 'Facts [W1]'])
def test_raw_observation_or_ambiguous_reference_cannot_be_synthesis(final):
    observation = WEB
    if final == 'Facts [W1]':
        observation += '\n\n[W1] Different source\nhttps://other.example\nOther excerpt'
    result = execute([(NS(tool='search_web'), observation)], final)
    assert final not in result['output']
    assert 'TECHNICAL_EXCERPT' not in result['output']
    assert result['intermediate_steps'][0]['status'] == 'success'
