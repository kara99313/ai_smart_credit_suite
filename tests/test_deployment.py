"""Deployment entry point checks; no inference or resource mutation."""
import os
import unittest
from unittest.mock import patch
from utils import launch


class DeploymentTests(unittest.TestCase):
    def test_render_web_resolves_internal_api_and_port(self):
        with patch.dict(os.environ, {'INTERNAL_API_HOSTPORT':'ai-smart-api:8000',
                'BACKEND_URL':'http://localhost:1', 'PORT':'10000', 'WEB_HOST':'0.0.0.0'}, clear=True):
            args = launch.command('web')
            self.assertEqual(os.environ['BACKEND_URL'], 'http://ai-smart-api:8000')
            self.assertEqual(args[args.index('--server.port')+1], '10000')
            self.assertEqual(args[args.index('--server.address')+1], '0.0.0.0')

    def test_api_render_port_overrides_local_port(self):
        with patch.dict(os.environ, {'PORT':'8000','API_PORT':'18000','API_HOST':'0.0.0.0'}, clear=True):
            args = launch.command('api')
            self.assertEqual(args[-1], '8000')
            self.assertEqual(args[args.index('--host')+1], '0.0.0.0')

    def test_missing_model_stops_api(self):
        with patch.dict(os.environ, {'MODEL_PATH':'model/absent-phase14.pkl'}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'MODEL_PATH'):
                launch.command('api')

    def test_invalid_port_stops_startup(self):
        with patch.dict(os.environ, {'PORT':'70000'}, clear=True):
            with self.assertRaises(ValueError):
                launch.command('web')

    def test_invalid_api_url_stops_web(self):
        with patch.dict(os.environ, {'BACKEND_URL':'not-an-http-url'}, clear=True):
            with self.assertRaises(ValueError):
                launch.command('web')

    def test_without_environment_local_defaults_start_both_services(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(launch.command('api')[-1], '18000')
            args = launch.command('web')
            self.assertEqual(args[args.index('--server.port')+1], '8501')


if __name__ == '__main__':
    unittest.main()
