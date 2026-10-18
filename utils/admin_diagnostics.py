"""Effective configuration provenance for Administration, without implicit probes."""
import os
from pathlib import Path
from utils.runtime_config import ENV_PATH
from utils.llm_runtime import PROVIDERS, select_provider
from utils.integration_services import effective_llm_identity
from utils.audit import redact_details


def setting_entry(name, default=None, sensitive=False):
    value, source = None, 'absent'
    try:
        import streamlit as st
        item = st.secrets.get(name)
        if item is not None:
            value, source = str(item), 'secrets Streamlit'
    except Exception:
        pass
    if value is None and name in os.environ:
        value, source = os.environ[name], 'processus (incluant .env charge au demarrage)'
    if value is None and default is not None:
        value, source = default, 'repli du code (configuration, pas une observation)'
    if sensitive:
        display = 'presente' if value and value.strip() else 'absente'
    else:
        display = redact_details(value) if value is not None else 'Non renseigne'
    return dict(parametre=name, valeur=display, source=source)


def configuration_snapshot():
    try:
        provider = select_provider()
    except Exception:
        return dict(valid=False, provider=None, rows=[setting_entry('LLM_PROVIDER')], identity=effective_llm_identity())
    key_name, model_name, model_default, _, _ = PROVIDERS[provider]
    rows = [setting_entry('LLM_PROVIDER'), setting_entry(model_name, model_default)]
    if key_name:
        rows.append(setting_entry(key_name, sensitive=True))
    if provider == 'ollama':
        rows.extend([setting_entry('OLLAMA_BASE_URL','http://127.0.0.1:11434'),setting_entry('OLLAMA_TIMEOUT','240')])
    else:
        rows.append(setting_entry('LLM_TIMEOUT','60'))
        rows.append(setting_entry('GROQ_BASE_URL' if provider == 'groq' else 'OPENAI_BASE_URL'))
    return dict(valid=True, provider=provider, rows=rows, identity=effective_llm_identity())


def configuration_files():
    from utils import settings
    paths = {'Environnement .env': ENV_PATH, 'Configuration YAML': settings.CONFIG_PATH,
             'Preferences JSON': settings.SETTINGS_PATH}
    result=[]
    for name, path in paths.items():
        try:
            path = Path(path)
            if not path.exists():
                state = 'Absent'
            else:
                with path.open('rb') as stream:
                    data = stream.read()
                if name == 'Configuration YAML':
                    import yaml
                    parsed = yaml.safe_load(data.decode('utf-8-sig'))
                    if not isinstance(parsed, dict): raise ValueError()
                elif name == 'Preferences JSON':
                    import json
                    if not isinstance(json.loads(data), dict): raise ValueError()
                state = 'Lisible (syntaxe verifiee)' if name != 'Environnement .env' else 'Lisible (contenu non affiche)'
        except (ValueError, UnicodeError):
            state = 'Syntaxe invalide ; preferences de repli possibles dans les pages utilisatrices'
        except OSError:
            state = 'Inaccessible'
        except Exception:
            state = 'Lecture ou syntaxe invalide'
        result.append(dict(fichier=name, etat=state))
    return result
