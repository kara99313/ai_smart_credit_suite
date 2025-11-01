# utils/multitask_agent.py
from __future__ import annotations
from typing import List, Tuple, Literal, Optional

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import StructuredTool  # ✅ LC 0.3.x
from langchain.agents import create_tool_calling_agent, AgentExecutor

from utils.llm_providers import load_llm
from utils.agent_tools import (
    predict_credit_tool,
    rag_retrieve_tool,
    tool_web_search,
    generate_report_tool,
    now_tool,
)

# ------------------------------------------------------------------
# Import facultatif du tool "client_context_tool" (si présent)
# ------------------------------------------------------------------
_HAS_CLIENT_CTX = False
try:
    from utils.agent_tools import client_context_tool  # type: ignore
    _HAS_CLIENT_CTX = True
except Exception:
    client_context_tool = None  # type: ignore

# ====================== Schémas Pydantic (outils) ======================

class PredictInput(BaseModel):
    """Chaîne JSON contenant les 15 variables + 'threshold' (float)."""
    payload_json: str = Field(
        ...,
        description=(
            'Chaîne JSON avec les variables du client. '
            'Exemple: {"client_id":"CLI-001","DTIRatio":0.35,...,"threshold":0.10}'
        ),
    )

class RagInput(BaseModel):
    """Recherche dans les index internes (docs / hybride)."""
    query: str = Field(..., description="Question utilisateur à chercher dans les documents.")
    mode: Literal["docs", "hybrid"] = Field(
        "docs",
        description="Index à utiliser : 'docs' (documents) ou 'hybrid' (docs + logs)."
    )
    k: int = Field(3, ge=1, le=6, description="Nombre de passages à retourner (1–6).")

class WebSearchInput(BaseModel):
    """Recherche Web via DuckDuckGo (résumés courts)."""
    query: str = Field(..., description="Requête Web.")
    k: int = Field(2, ge=1, le=6, description="Nombre de résultats (1–6).")
    fetch: bool = Field(False, description="Télécharger les pages pour un résumé plus propre (HTML→texte).")

class ReportInput(BaseModel):
    """Génération d'un mini-rapport HTML (scoring)."""
    ctx_json: str = Field(
        ...,
        description=(
            "Chaîne JSON avec au minimum: client_id, pd_value, threshold, inputs_json. "
            "Optionnels: model_tag, logo_path."
        ),
    )

class NowInput(BaseModel):
    """Heure actuelle formatée (UTC)."""
    fmt: Optional[str] = Field("%Y-%m-%d %H:%M:%S", description="Format strftime. Ex: %Y-%m-%d %H:%M:%S")

class ClientCtxInput(BaseModel):
    """Contexte client depuis le journal (optionnel si le tool existe)."""
    client_id: Optional[str] = Field("", description="Client ID (facultatif, sinon dernier log).")
    max_chars: int = Field(700, ge=200, le=1200, description="Longueur max du bloc contexte.")

# ====================== Wrappers (pas de lambda) ======================

def _rag_retrieve_wrapper(query: str, mode: Literal["docs", "hybrid"] = "docs", k: int = 3) -> str:
    return rag_retrieve_tool(query=query, mode=mode, k=k)

def _web_search_wrapper(query: str, k: int = 2, fetch: bool = False) -> str:
    return tool_web_search(query=query, k=k, fetch=fetch)

def _client_ctx_wrapper(client_id: str = "", max_chars: int = 700) -> str:
    if not _HAS_CLIENT_CTX or client_context_tool is None:
        return (
            "Contexte client indisponible : l'outil 'client_context_tool' n'est pas chargé.\n"
            "Ajoute-le dans utils/agent_tools.py pour l'activer."
        )
    return client_context_tool(client_id=client_id, max_chars=max_chars)  # type: ignore

# ====================== Déclaration des tools ======================

def _tools() -> List[StructuredTool]:
    tools: List[StructuredTool] = [
        StructuredTool.from_function(
            name="predict_credit",
            description=(
                "Prédit PD / score (0–1000) / rating / décision à partir d'un JSON "
                "(15 variables + threshold)."
            ),
            func=predict_credit_tool,
            args_schema=PredictInput,
        ),
        StructuredTool.from_function(
            name="rag_retrieve",
            description=(
                "Extrait des passages depuis les documents internes (mode 'docs') "
                "ou hybride (mode 'hybrid')."
            ),
            func=_rag_retrieve_wrapper,
            args_schema=RagInput,
        ),
        StructuredTool.from_function(
            name="search_web",
            description="Recherche Web (DuckDuckGo). Retourne un contexte court avec 1–6 résultats.",
            func=_web_search_wrapper,
            args_schema=WebSearchInput,
        ),
        StructuredTool.from_function(
            name="build_report",
            description="Génère un mini-rapport **HTML** de scoring de crédit à partir d’un JSON de contexte.",
            func=generate_report_tool,
            args_schema=ReportInput,
        ),
        StructuredTool.from_function(
            name="now",
            description="Retourne l'heure actuelle UTC formatée (strftime).",
            func=now_tool,
            args_schema=NowInput,
        ),
    ]

    if _HAS_CLIENT_CTX:
        tools.append(
            StructuredTool.from_function(
                name="client_context",
                description=(
                    "Récupère le contexte du client courant (résultat de prédiction + variables clés) "
                    "depuis le journal."
                ),
                func=_client_ctx_wrapper,
                args_schema=ClientCtxInput,
            )
        )

    return tools

_SYSTEM = (
    "Tu es un agent pour une plateforme de scoring de crédit. "
    "Utilise les OUTILS quand pertinent : predict_credit, "
    + ("client_context, " if _HAS_CLIENT_CTX else "")
    + "rag_retrieve, search_web, build_report, now. "
    "Si l'utilisateur fournit un JSON de variables, commence par predict_credit. "
    "Sois concis et cite les sources [C] (client), [D#] (docs) / [W#] (web) si tu utilises RAG/Web."
)

# ====================== Création de l'agent ======================

def create_multitask_agent() -> AgentExecutor:
    llm = load_llm()
    tools = _tools()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", _SYSTEM),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)

    # LangChain 0.3.x : early_stopping_method="generate" n'est plus supporté
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=4,              # limite pour paliers gratuits
        early_stopping_method="force", # ✅ compatible LC 0.3.x
    )

# ====================== Utilitaires ======================

def as_messages(history: List[Tuple[str, str]], max_pairs: int = 6):
    """
    Convertit l'historique (role, content) en messages LangChain.
    Tronque à 'max_pairs' derniers échanges pour limiter le contexte.
    """
    if history and len(history) > max_pairs:
        history = history[-max_pairs:]
    msgs = []
    for role, content in history:
        msgs.append(HumanMessage(content) if role == "user" else AIMessage(content))
    return msgs
