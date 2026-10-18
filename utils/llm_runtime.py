"""One explicit provider factory; bounded calls and secret-free errors. No runtime fallback."""
import importlib
import math
import os
from utils.runtime_config import ROOT  # canonical .env loading, override=False

PROVIDERS={
 'groq':('GROQ_API_KEY','GROQ_MODEL','openai/gpt-oss-20b','langchain_groq','ChatGroq'),
 'openai':('OPENAI_API_KEY','OPENAI_MODEL','gpt-4o-mini','langchain_openai','ChatOpenAI'),
 'ollama':(None,'OLLAMA_MODEL','llama3.2:3b','langchain_ollama','ChatOllama'),
}


def get_setting(name,default=''):
    try:
        import streamlit as st
        value=st.secrets.get(name)
        if value is not None:return str(value)
    except Exception:pass
    return os.getenv(name,default)


class ProviderError(RuntimeError):
    pass


def explain(error):
    if isinstance(error,ProviderError):return str(error)
    code=getattr(error,'status_code',None)
    if code in (401,403):return 'Authentification refusee (cle/droits).'
    if code==404:return 'Modele absent ou non accessible (HTTP 404).'
    if code==429:return 'Quota ou limite fournisseur atteint (HTTP 429).'
    if isinstance(error,(ImportError,ModuleNotFoundError)):return 'Connecteur absent ou incompatible.'
    name=type(error).__name__.lower()
    if 'timeout' in name:return 'Delai fournisseur depasse ; passages conserves, nouvel essai possible.'
    current=error
    for _ in range(6):
        if getattr(current,'winerror',None)==10013:return 'Connexion bloquee par les permissions reseau (WinError 10013).'
        current=getattr(current,'__cause__',None)
        if current is None:break
    return 'Connexion ou appel fournisseur en echec.'


def select_provider(getter=get_setting):
    selected=getter('LLM_PROVIDER','').strip().lower()
    if selected and selected!='auto':
        if selected not in PROVIDERS:raise ProviderError('Fournisseur inconnu.')
        return selected
    # Initial configuration selection only, never fallback after an invocation error.
    if getter('GROQ_API_KEY','').strip():return 'groq'
    if getter('OPENAI_API_KEY','').strip():return 'openai'
    return 'ollama'


def build_client(provider,*,getter=get_setting,max_tokens=768,timeout=None,json_mode=False):
    if provider not in PROVIDERS:raise ProviderError('Fournisseur inconnu.')
    key_name,model_name,default,module,cls=PROVIDERS[provider]
    key=getter(key_name,'').strip() if key_name else None
    if key_name and not key:raise ProviderError('Cle absente ; aucun appel effectue.')
    model=getter(model_name,default).strip()
    if not model:raise ProviderError('Modele non configure.')
    try:
        deadline=float(timeout if timeout is not None else getter('OLLAMA_TIMEOUT' if provider=='ollama' else 'LLM_TIMEOUT','240' if provider=='ollama' else '60'))
    except (ValueError,TypeError):
        raise ProviderError('Delai invalide.') from None
    if not math.isfinite(deadline) or not 0<deadline<=600:raise ProviderError('Delai invalide (0 < timeout <= 600 secondes).')
    kwargs=dict(model=model,temperature=0)
    if provider in {'groq','openai'}:
        budget=max(128,max_tokens) if provider=='groq' and 'gpt-oss' in model else max_tokens
        kwargs.update(api_key=key,max_tokens=budget,timeout=deadline,max_retries=0)
        extras={}
        if provider=='groq' and 'gpt-oss' in model:kwargs['reasoning_effort']='low'
        if json_mode:extras['response_format']={'type':'json_object'}
        if extras:kwargs['model_kwargs']=extras
        configured_base=getter('GROQ_BASE_URL' if provider=='groq' else 'OPENAI_BASE_URL','')
        if configured_base:kwargs['base_url']=configured_base
    else:
        kwargs.update(base_url=getter('OLLAMA_BASE_URL','http://127.0.0.1:11434'),num_predict=max_tokens,
                      num_ctx=2048,keep_alive='5m',client_kwargs={'timeout':deadline})
        if json_mode:kwargs['format']='json'
    try:return getattr(importlib.import_module(module),cls)(**kwargs)
    except Exception as error:raise ProviderError(explain(error)) from None


def invoke(client,messages,provider):
    try:
        response=client.invoke(messages)
        if not isinstance(getattr(response,'content',None),str) or not response.content.strip():
            raise ProviderError('Reponse finale vide ; verifier budget de tokens et modele.')
        return response
    except Exception as error:
        raise ProviderError(f'{provider} indisponible : {explain(error)}') from None
