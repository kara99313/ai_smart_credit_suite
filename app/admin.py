"""Administration of the active provider; no external call on page load."""
from __future__ import annotations
import streamlit as st
import hashlib
from datetime import datetime, timezone
from utils.admin_auth import render_gate, require_admin, sensitive
from utils import audit
from utils.admin_diagnostics import configuration_snapshot, configuration_files
from utils.llm_runtime import PROVIDERS, get_setting, select_provider, explain
from utils.llm_providers import simple_chat
from utils.integration_services import save_llm_configuration, effective_llm_identity



def _record_action(event, details):
    try:
        audit.append_audit(event, actor=require_admin(), details=details)
        return True
    except Exception:
        st.warning('Journal audit indisponible ; cette action n est pas confirmee dans le journal.')
        return False

@sensitive
def test_admin_llm(prompt):
    return simple_chat(prompt)


def main():
    if not render_gate():
        return
    st.markdown('## Administration — Configuration LLM')
    try:
        current = select_provider()
    except Exception:
        current = 'groq'
        st.warning('Configuration fournisseur invalide.')
    providers = list(PROVIDERS)
    provider = st.selectbox('Fournisseur LLM', providers, index=providers.index(current))
    key_name, model_name, default, _, _ = PROVIDERS[provider]
    st.caption('Les secrets Streamlit sont prioritaires sur les variables du processus et .env. Les valeurs existantes ne sont pas renvoyees au navigateur.')
    form_identity = effective_llm_identity()[:12]
    st.caption('Le formulaire contient des valeurs de configuration modifiables ; il ne constitue pas une mesure de disponibilite du fournisseur.')
    with st.form('llm_configuration', clear_on_submit=True):
        api_key = ''
        if key_name:
            st.caption(f'{key_name} : ' + ('presente' if get_setting(key_name).strip() else 'absente'))
            api_key = st.text_input('Nouvelle cle API (vide = conserver)', type='password', key='admin_new_key_' + provider + form_identity)
        model = st.text_input(model_name, value=get_setting(model_name, default), key='admin_' + model_name + form_identity)
        base = ''
        if provider == 'ollama':
            # Do not prefill potentially credential-bearing malformed URLs.
            from utils.integration_services import validate_base_url
            try:
                base = validate_base_url(get_setting('OLLAMA_BASE_URL', 'http://127.0.0.1:11434'))
            except ValueError:
                st.warning('OLLAMA_BASE_URL invalide ; saisir une URL sans identifiants.')
            base = st.text_input('OLLAMA_BASE_URL', value=base, key='admin_base_' + form_identity)
        saved = st.form_submit_button('Enregistrer dans .env')
    if saved:
        try:
            save_llm_configuration(provider, model, api_key, base)
            # Cached clients in this browser must not retain the previous provider/key.
            st.session_state.pop('agent', None)
            st.session_state.pop('assistant_agent_config', None)
            st.session_state.pop('admin_llm_test', None)
            _record_action('configuration_llm_enregistree', {'provider_demande': provider, 'nouvelle_cle_fournie': bool(api_key.strip())})
            st.success('Configuration enregistree et appliquee au processus courant. Les secrets Streamlit restent prioritaires.')
        except ValueError:
            _record_action('configuration_llm_refusee', {'provider_demande': provider})
            st.error('Parametres invalides ou fichier .env illisible ; aucune sauvegarde confirmee.')
        except Exception:
            _record_action('configuration_llm_echec', {'provider_demande': provider})
            st.error('Enregistrement impossible ; verifier les droits du fichier .env.')
    st.subheader('Configuration effective et sources')
    snapshot = configuration_snapshot()
    st.dataframe(snapshot['rows'], hide_index=True)
    st.dataframe(configuration_files(), hide_index=True)
    st.caption('Les valeurs de repli sont des parametres du code, pas des observations du service. YAML et preferences JSON ne pilotent pas le client LLM partage ; la table ci-dessus expose ses sources effectives.')
    st.subheader('Test du fournisseur effectif')
    st.caption('Le test utilise la configuration appliquee, pas les modifications non enregistrees. Il peut consommer du quota.')
    identity = effective_llm_identity()
    try:
        st.write('Fournisseur effectif : ' + select_provider())
    except Exception:
        st.write('Fournisseur effectif : configuration invalide')
    prompt = st.text_area('Prompt', 'Dis simplement : Bonjour.')
    prompt_identity = hashlib.sha256(prompt.encode()).hexdigest()
    if st.button('Tester le LLM'):
        if not prompt.strip():
            st.error('Prompt vide ; aucun appel effectue.')
        else:
            checked = datetime.now(timezone.utc).isoformat(timespec='seconds')
            try:
                output = test_admin_llm(prompt)
                st.session_state['admin_llm_test'] = (identity, True, audit.redact_details(output), checked, prompt_identity)
                _record_action('test_llm_termine', {'provider_effectif': snapshot['provider'], 'succes': True})
            except Exception as error:
                st.session_state['admin_llm_test'] = (identity, False, audit.redact_details(explain(error)), checked, prompt_identity)
                _record_action('test_llm_termine', {'provider_effectif': snapshot['provider'], 'succes': False})
    result = st.session_state.get('admin_llm_test')
    if result and len(result) == 5 and result[0] == identity and result[4] == prompt_identity:
        st.caption('Dernier test explicite (UTC) : ' + result[3] + '. Ce resultat ne garantit pas la disponibilite actuelle.')
        if result[1]:
            st.write(result[2])
        else:
            st.error(result[2])
    else:
        st.caption('Aucun test disponible pour cette configuration et ce prompt ; service non teste ici.')


if __name__ == '__main__':
    main()
