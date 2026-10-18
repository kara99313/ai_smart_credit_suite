"""Legacy entry points delegate to the shared explicit provider implementation."""
from typing import Optional
from utils.llm_runtime import build_client,get_setting,select_provider
_get_secret=get_setting


def _groq_or_raise():
    return build_client('groq',getter=_get_secret)


def _openai_or_raise():
    return build_client('openai',getter=_get_secret)


def _ollama_or_raise():
    return build_client('ollama',getter=_get_secret)


def load_llm():
    return build_client(select_provider(_get_secret),getter=_get_secret)


def simple_chat(prompt: str,system: Optional[str]=None):
    from utils.llm_runtime import invoke
    provider=select_provider(_get_secret)
    messages=([{'role':'system','content':system}] if system else [])+[{'role':'user','content':prompt}]
    return invoke(build_client(provider,getter=_get_secret),messages,provider).content
