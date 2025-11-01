# utils/agents.py
from __future__ import annotations
from typing import List, Tuple
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langchain_core.tools import StructuredTool

from utils.llm_providers import load_llm
from utils.agent_tools import (
    predict_credit_tool,
    rag_retrieve_tool,
    tool_web_search,
    generate_report_tool,
    now_tool,
)

_SYSTEM = (
    "Tu es un agent d'assistance pour le scoring de crédit. "
    "Choisis intelligemment les outils (predict_credit, rag_retrieve, search_web, build_report, now), "
    "explique tes étapes si nécessaire, et reste concis."
)

def _tools():
    return [
        StructuredTool.from_function(
            func=predict_credit_tool,
            name="predict_credit",
            description="Prédit PD/score/rating/décision à partir d'un JSON des caractéristiques client."
        ),
        StructuredTool.from_function(
            func=rag_retrieve_tool,
            name="rag_retrieve",
            description="Recherche des passages dans l’index RAG (mode 'docs' par défaut)."
        ),
        StructuredTool.from_function(
            func=tool_web_search,
            name="search_web",
            description="Recherche Web (résumés courts) avec possibilité de fetch."
        ),
        StructuredTool.from_function(
            func=generate_report_tool,
            name="build_report",
            description="Génère un mini-rapport HTML depuis un contexte JSON."
        ),
        StructuredTool.from_function(
            func=now_tool,
            name="now",
            description="Donne l'heure/horodatage actuel."
        ),
    ]

def create_credit_agent() -> AgentExecutor:
    llm = load_llm()               # Groq/OpenAI/Ollama selon .env
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

    # ⚠️ PAS d’early_stopping_method ici (incompatible LC 0.3)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
        max_iterations=6,
    )

def run_agent_once(agent: AgentExecutor, user_query: str, chat_history: List[BaseMessage] | List[Tuple[str, str]]):
    return agent.invoke({"input": user_query, "chat_history": chat_history})
