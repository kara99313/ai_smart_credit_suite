# utils/lc_agent.py
from __future__ import annotations
from typing import List
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama  # ou OpenAI si tu utilises l'API
from utils.agent_tools import (
    predict_credit_tool, rag_retrieve_tool, search_web_tool,
    generate_report_tool,
)

def make_tools() -> List[Tool]:
    return [
        Tool(
            name="predict_credit",
            description="Calcule PD, score (0–1000), rating, décision à partir d'un JSON de 15 variables.",
            func=predict_credit_tool,
        ),
        Tool(
            name="rag_retrieve",
            description="Récupère des passages depuis les documents internes (RAG). Entrée: question.",
            func=rag_retrieve_tool,
        ),
        Tool(
            name="search_web",
            description="Recherche web (DuckDuckGo HTML). Entrée: question/mots-clés.",
            func=search_web_tool,
        ),
        Tool(
            name="build_report",
            description="Génère un rapport HTML prêt à télécharger. Entrée: JSON context (client_id, pd_value...)",
            func=generate_report_tool,
        ),
    ]

def make_agent(model: str = "llama3.2:3b", base_url: str | None = None) -> AgentExecutor:
    llm = ChatOllama(model=model, base_url=base_url or "http://localhost:11434", temperature=0)
    tools = make_tools()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Tu es un agent d'analyse crédit. Utilise les OUTILS quand c'est pertinent et cite les sources [D1]/[W1]. Réponds en français si l'utilisateur parle français."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False)
