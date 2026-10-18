"""Focused Web provider, filtering and agent compatibility tests."""
import unittest
from unittest.mock import Mock, patch
import requests
from utils import web_search as web

CONFIG = dict(enabled=True, domains=['bceao.int'], timeout=2.5, max_results=3)
HTML = '<a class="result__a" href="https://www.bceao.int/">BCEAO</a><div class="result__snippet">Banque Centrale</div>'


class WebSearchTests(unittest.TestCase):
    def setUp(self):
        original = web.get_setting
        patcher = patch.object(web, 'get_setting', side_effect=lambda name, default='': '' if name == 'TAVILY_API_KEY' else original(name, default))
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_rate_limit_falls_back_to_real_html_results(self):
        client = Mock()
        client.__enter__ = Mock(return_value=client)
        client.__exit__ = Mock(return_value=False)
        client.text.side_effect = web.RatelimitException('PRIVATE')
        response = Mock(status_code=200, text=HTML)
        with patch.object(web, 'web_configuration', return_value=CONFIG), patch.object(web, '_HAS_DDG', True), patch.object(web, 'DDGS', return_value=client) as ddgs, patch.object(web.requests, 'post', return_value=response) as post:
            hits = web.search_web('BCEAO')
        self.assertEqual(hits, [dict(title='BCEAO', href='https://www.bceao.int/', body='Banque Centrale')])
        client.text.assert_called_once()
        ddgs.assert_called_once_with(timeout=2.5)
        self.assertEqual(post.call_args.kwargs['timeout'], 2.5)

    def test_block_pages_are_explicit_errors(self):
        for status, body in [(202, ''), (403, ''), (429, ''), (200, '<form id="challenge-form">captcha</form>')]:
            with self.subTest(status=status), patch.object(web.requests, 'post', return_value=Mock(status_code=status, text=body)):
                with self.assertRaises(web.WebSearchRateLimitError):
                    web._search_web_ddg_html('BCEAO', 3)

    def test_unrecognized_response_is_not_empty_success(self):
        with patch.object(web.requests, 'post', return_value=Mock(status_code=200, text='<html>Unexpected</html>')):
            with self.assertRaises(web.WebSearchError):
                web._search_web_ddg_html('BCEAO', 3)

    def test_combined_errors_keep_rate_limit_without_private_details(self):
        with patch.object(web, 'web_configuration', return_value=CONFIG), patch.object(web, '_HAS_DDG', True), patch.object(web, '_search_web_ddg_lib', side_effect=web.WebSearchRateLimitError('rate limit')), patch.object(web.requests, 'post', side_effect=requests.ConnectionError('PRIVATE')):
            with self.assertRaisesRegex(web.WebSearchError, 'rate limit') as error:
                web.search_web('BCEAO')
        self.assertNotIn('PRIVATE', str(error.exception))

    def test_agent_alias_returns_sources_and_safe_failure(self):
        # Load only the Web adapters: unrelated RAG dependencies are optional here.
        import ast
        from pathlib import Path
        from types import SimpleNamespace
        tree = ast.parse((Path(__file__).resolve().parents[1] / 'utils/agent_tools.py').read_text(encoding='utf-8'))
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {'tool_web_search', 'search_web_tool'}]
        namespace = dict(search_web=Mock(), build_web_context=web.build_web_context, _utf8=str, explain=lambda exc: 'Service indisponible')
        exec(compile(tree, 'utils/agent_tools.py', 'exec'), namespace)
        agent_tools = SimpleNamespace(**namespace)
        hits = [dict(title='BCEAO', href='https://www.bceao.int/', body='Banque Centrale')]
        with patch.dict(namespace, search_web=Mock(return_value=hits)):
            self.assertIn('https://www.bceao.int/', agent_tools.search_web_tool('BCEAO'))
        with patch.dict(namespace, search_web=Mock(side_effect=web.WebSearchRateLimitError('rate limit'))):
            self.assertIn('indisponible', agent_tools.search_web_tool('BCEAO'))


class StreamlitWebPathTests(unittest.TestCase):
    def setUp(self):
        original = web.get_setting
        patcher = patch.object(web, 'get_setting', side_effect=lambda name, default='': '' if name == 'TAVILY_API_KEY' else original(name, default))
        patcher.start()
        self.addCleanup(patcher.stop)

    def app(self):
        from pathlib import Path
        from streamlit.testing.v1 import AppTest
        return AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app/integrations.py'), default_timeout=30)

    def search(self, app):
        app.run()
        next(t for t in app.text_input if t.label == 'Requete de test Web').set_value('BCEAO').run()
        next(b for b in app.button if b.label == 'Tester la recherche Web').click().run()
        self.assertFalse(app.exception)
        return app.session_state['integration_web_test'][1]

    def test_button_uses_effective_streamlit_secrets_and_filters(self):
        from utils import settings
        cfg = settings.AppSettings(enable_web_search=False, allowed_domains=['old.example'], web_search_max=3)
        app = self.app()
        app.secrets['WEB_SEARCH_ENABLED'] = 'true'
        app.secrets['WEB_ALLOWED_DOMAINS'] = 'BCEAO.INT'
        app.secrets['WEB_TIMEOUT'] = '2.5'
        hits = [dict(title='BCEAO', href='https://www.bceao.int/', body='Banque Centrale'), dict(title='Other', href='https://other.example/', body='Other')]
        with patch.object(settings, 'get_settings', return_value=cfg), patch.object(web, '_HAS_DDG', True), patch.object(web, '_search_web_ddg_lib', return_value=hits) as library:
            message, results = self.search(app)
        library.assert_called_once_with('BCEAO', max_results=2, region='fr-fr', timeout=2.5)
        self.assertEqual(results, hits[:1])
        self.assertEqual(message, 'Resultats Web recus. Fournisseur : DuckDuckGo.')

    def test_button_reports_tavily_without_secret(self):
        app = self.app()
        app.secrets['WEB_SEARCH_ENABLED'] = 'true'
        app.secrets['TAVILY_API_KEY'] = 'test-token'
        response = Mock(status_code=200)
        response.json.return_value = {'results': [dict(title='BCEAO', url='https://bceao.int/', content='Banque Centrale')]}
        original = web.get_setting.side_effect
        with patch.object(web, 'get_setting', side_effect=lambda name, default='': 'test-token' if name == 'TAVILY_API_KEY' else original(name, default)), patch.object(web.requests, 'post', return_value=response), patch.object(web, '_search_web_ddg_lib') as ddg:
            message, hits = self.search(app)
        self.assertIn('Fournisseur : Tavily.', message)
        self.assertEqual(len(hits), 1)
        self.assertNotIn('test-token', ' '.join(str(element.value) for kind in ('markdown', 'caption', 'json', 'error', 'text_input') for element in getattr(app, kind)))
        ddg.assert_not_called()

    def test_disabled_streamlit_secret_prevents_network(self):
        app = self.app()
        app.secrets['WEB_SEARCH_ENABLED'] = 'false'
        with patch.object(web, 'search_web') as search:
            app.run()
            next(t for t in app.text_input if t.label == 'Requete de test Web').set_value('BCEAO').run()
        self.assertTrue(next(b for b in app.button if b.label == 'Tester la recherche Web').disabled)
        search.assert_not_called()

    def test_button_keeps_safe_rate_limit_diagnostic(self):
        app = self.app()
        app.secrets['WEB_SEARCH_ENABLED'] = 'true'
        with patch.object(web, 'search_web', side_effect=web.WebSearchRateLimitError('DuckDuckGo rate limit HTTP 202')):
            message, hits = self.search(app)
        self.assertIn('rate limit HTTP 202', message)
        self.assertEqual(hits, [])

    def test_button_reports_network_permissions_without_exception_body(self):
        app = self.app()
        app.secrets['WEB_SEARCH_ENABLED'] = 'true'
        with patch.object(web, '_HAS_DDG', True), patch.object(web, 'DDGS', side_effect=RuntimeError('PRIVATE WinError 10013')), patch.object(web.requests, 'post', side_effect=requests.ConnectionError('PRIVATE WinError 10013')):
            message, hits = self.search(app)
        self.assertIn('WinError 10013', message)
        self.assertIn('Repli HTML', message)
        self.assertNotIn('PRIVATE', message)
        self.assertEqual(hits, [])

    def test_unexpected_exception_body_is_not_displayed(self):
        app = self.app()
        app.secrets['WEB_SEARCH_ENABLED'] = 'true'
        with patch.object(web, 'search_web', side_effect=RuntimeError('PRIVATE')):
            message, hits = self.search(app)
        self.assertNotIn('PRIVATE', message)
        self.assertIn('indisponible', message)
        self.assertEqual(hits, [])

class TavilyTests(unittest.TestCase):
    def test_disabled_with_key_never_contacts_providers(self):
        with patch.object(web, 'web_configuration', return_value=dict(CONFIG, enabled=False)), patch.object(web, 'get_setting', return_value='test-token'), patch.object(web.requests, 'post') as post, patch.object(web, '_search_web_ddg_lib') as ddg:
            self.assertEqual(web.search_web('BCEAO'), [])
        post.assert_not_called()
        ddg.assert_not_called()

    def test_primary_filters_domains_limits_and_preserves_agent_format(self):
        response = Mock(status_code=200)
        response.json.return_value = {'results': [
            dict(title='BCEAO', url='https://www.bceao.int/', content='Banque Centrale'),
            dict(title='Other', url='https://other.example/', content='Other'),
            dict(title='Duplicate', url='https://www.bceao.int/', content='Duplicate')]}
        with patch.object(web, 'web_configuration', return_value=CONFIG), patch.object(web, 'get_setting', return_value='test-token'), patch.object(web.requests, 'post', return_value=response) as post, patch.object(web, '_search_web_ddg_lib') as ddg:
            hits = web.search_web('BCEAO', max_results=2)
        self.assertEqual(hits, [dict(title='BCEAO', href='https://www.bceao.int/', body='Banque Centrale')])
        self.assertEqual(hits.provider, 'Tavily')
        ddg.assert_not_called()
        self.assertEqual(post.call_args.kwargs['timeout'], 2.5)
        self.assertEqual(post.call_args.kwargs['json']['include_domains'], ['bceao.int'])
        self.assertEqual(post.call_args.kwargs['json']['max_results'], 2)
        self.assertFalse(post.call_args.kwargs['allow_redirects'])

    def test_provider_failures_fall_back_without_exposing_details(self):
        failures = [requests.Timeout('PRIVATE'), requests.ConnectionError('PRIVATE')]
        for status in (401, 403, 429, 432, 433, 500, 302):
            failures.append(Mock(status_code=status))
        invalid = Mock(status_code=200)
        invalid.json.return_value = {'results': 'PRIVATE'}
        failures.append(invalid)
        malformed = Mock(status_code=200)
        malformed.json.return_value = {'results': [{'url': 7}]}
        failures.append(malformed)
        for failure in failures:
            kwargs = {'side_effect': failure} if isinstance(failure, Exception) else {'return_value': failure}
            with self.subTest(kind=type(failure).__name__), patch.object(web, 'web_configuration', return_value=CONFIG), patch.object(web, 'get_setting', return_value='test-token'), patch.object(web.requests, 'post', **kwargs), patch.object(web, '_HAS_DDG', True), patch.object(web, '_search_web_ddg_lib', return_value=[dict(title='BCEAO', href='https://bceao.int/', body='Central bank')]):
                hits = web.search_web('BCEAO')
                self.assertEqual(hits.provider, 'DuckDuckGo')
                self.assertEqual(len(hits), 1)

    def test_both_providers_fail_with_safe_diagnostic(self):
        with patch.object(web, 'web_configuration', return_value=CONFIG), patch.object(web, 'get_setting', return_value='test-token'), patch.object(web.requests, 'post', side_effect=requests.Timeout('PRIVATE test-token')), patch.object(web, '_HAS_DDG', False):
            with self.assertRaises(web.WebSearchError) as error:
                web.search_web('BCEAO')
        self.assertIn('Tavily', str(error.exception))
        self.assertIn('DuckDuckGo', str(error.exception))
        self.assertNotIn('PRIVATE', str(error.exception))
        self.assertNotIn('test-token', str(error.exception))

    def test_empty_tavily_results_are_not_fabricated(self):
        response = Mock(status_code=200)
        response.json.return_value = {'results': []}
        with patch.object(web, 'web_configuration', return_value=CONFIG), patch.object(web, 'get_setting', return_value='test-token'), patch.object(web.requests, 'post', return_value=response), patch.object(web, '_search_web_ddg_lib') as ddg:
            hits = web.search_web('BCEAO')
        self.assertEqual(hits, [])
        self.assertEqual(hits.provider, 'Tavily')
        ddg.assert_not_called()
