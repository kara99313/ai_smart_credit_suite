# 💠 AI Smart Credit Suite

## 📌 Présentation

**AI Smart Credit Suite** est une plateforme intelligente de *scoring de
crédit inclusif* basée sur l'Intelligence Artificielle.\
Elle combine **modélisation prédictive**, **analyse documentaire RAG**,
**agents IA autonomes**, et **interface interactive Streamlit**, tout en
respectant les standards réglementaires (Bâle III/IV, IFRS 9, RGPD, AI
Act).

> 🎯 Objectif : offrir une solution *banque-ready*, explicable et
> inclusive pour évaluer le risque de crédit dans les environnements
> bancaires et semi-formels.

------------------------------------------------------------------------

## 🧩 Architecture Générale

L'écosystème repose sur quatre couches principales :

    ───────────────────────────────────────────────────────────────
                      🧠 AI SMART CREDIT SUITE
    ───────────────────────────────────────────────────────────────

                        ┌────────────────────────┐
                        │   Streamlit (Frontend) │
                        │  → Interface utilisateur │
                        └────────────┬───────────┘
                                     │
                                     ▼
                          ┌────────────────────┐
                          │     Agent IA       │
                          │ (multitâche LangChain) │
                          └────────┬───────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
       ┌─────────────┐     ┌─────────────┐       ┌────────────────┐
       │ predict_credit │  │ rag_retrieve │      │  search_web     │
       │ (API FastAPI)  │  │ (FAISS Docs) │      │ (DuckDuckGo)   │
       └─────────────┘     └─────────────┘       └────────────────┘
              │                    │                    │
              └──────────┬─────────┴──────────┬─────────┘
                         ▼                    ▼
                  ┌────────────────┐   ┌────────────────────┐
                  │  build_report  │   │   now_tool (UTC)   │
                  └────────────────┘   └────────────────────┘
                                   │
                                   ▼
                       ┌────────────────────────┐
                       │  LLM (Groq / Ollama / OpenAI) │
                       │  ↳ Llama3, GPT-4, etc.       │
                       └────────────────────────┘
                                   │
                                   ▼
                         ┌──────────────────────┐
                         │  Backend FastAPI     │
                         │  /api/predict etc.   │
                         └──────────────────────┘
                                   │
                                   ▼
                         📊 Base de données locale (CSV / FAISS)
    ───────────────────────────────────────────────────────────────

------------------------------------------------------------------------

## ⚙️ Fonctionnalités principales

  -----------------------------------------------------------------------
  Domaine                       Description
  ----------------------------- -----------------------------------------
  🧠 **Agent IA multitâche**    Orchestration LangChain avec outils :
                                prédiction, RAG, web, rapport, horodatage

  📚 **RAG (Retrieval-Augmented Recherche sémantique dans les documents
  Generation)**                 internes + web contextuel

  🤖 **Chatbots intelligents**  Chatbot assistant (finance/scoring) &
                                chatbot RAG (documents + web)

  📊 **Dashboards analytiques** Tableau global et client, indicateurs
                                clés de risque et performance

  🧾 **Rapport automatique**    Génération de rapports HTML explicatifs
                                et décisionnels

  🌍 **Multilingue (FR/EN)**    Interface et IA bilingues

  🔐 **Conformité IA**          Aligné sur BCBS 239, IFRS 9, RGPD et AI
                                Act (UE)
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 🛠️ Technologies Utilisées

  -----------------------------------------------------------------------
  Catégorie                                  Outils
  ------------------------------------------ ----------------------------
  **Langages**                               Python 3.12

  **Frameworks IA**                          LangChain 0.3.x, LangGraph
                                             (futur), Pydantic 2.x

  **Fournisseurs LLM**                       Groq (Llama3), Ollama
                                             (local), OpenAI (optionnel)

  **Frontend**                               Streamlit 1.50

  **Backend**                                FastAPI (API scoring)

  **RAG**                                    FAISS, Sentence-Transformers

  **Stockage**                               CSV, FAISS Vector Store

  **DevOps / CI/CD**                         PowerShell scripts
                                             `start.ps1`, `stop.ps1`

  **Environnement**                          `.env`, venv,
                                             requirements.txt
  -----------------------------------------------------------------------

------------------------------------------------------------------------

## 🗂️ Structure du projet

    ai_smart_app_v1/
    │
    ├── app/                      # Pages Streamlit
    │   ├── home.py
    │   ├── prediction.py
    │   ├── client_dashboard.py
    │   ├── global_dashboard.py
    │   ├── report.py
    │   ├── agent.py
    │   ├── chatbot_assistant.py
    │   ├── rag_chatbot.py
    │   └── ...
    │
    ├── utils/                    # Modules utilitaires
    │   ├── api_client.py
    │   ├── agent_tools.py
    │   ├── multitask_agent.py
    │   ├── lang_utils.py
    │   ├── web_search.py
    │   ├── rag_utils.py
    │   ├── llm_providers.py
    │   ├── settings.py
    │   └── ...
    │
    ├── data/                     # Logs, prédictions, indices FAISS
    │   └── predictions_log.csv
    │
    ├── docs_rag/                 # Documents internes indexés
    │
    ├── server.py                 # Backend FastAPI
    ├── streamlit_app.py          # Application principale
    ├── start.ps1 / stop.ps1      # Scripts PowerShell
    ├── requirements.txt          # Dépendances
    ├── .env                      # Variables d'environnement
    └── README.md                 # Documentation projet

------------------------------------------------------------------------

## ⚙️ Installation et configuration

### 1️⃣ Cloner le dépôt

``` bash
git clone https://github.com/votrecompte/ai_smart_credit_suite.git
cd ai_smart_credit_suite
```

### 2️⃣ Créer et activer un environnement virtuel

``` bash
python -m venv .venv
.venv\Scripts\activate
```

### 3️⃣ Installer les dépendances

``` bash
pip install -r requirements.txt
```

### 4️⃣ Configurer l'environnement

Créer un fichier `.env` à la racine :

``` bash
LLM_PROVIDER=groq
GROQ_API_KEY=VOTRE_CLE
GROQ_MODEL=llama3-70b-8192
BACKEND_URL=http://127.0.0.1:18000
```

------------------------------------------------------------------------

## 🚀 Lancement

### Option A : via script PowerShell

``` bash
.\start.ps1
```

### Option B : manuellement

``` bash
uvicorn server:app --host 127.0.0.1 --port 18000
streamlit run streamlit_app.py
```

### Option C : déploiement cloud

-   **Streamlit Cloud** (gratuit)
-   **Render / HuggingFace Spaces**
-   **Docker / Azure / AWS** (production)

------------------------------------------------------------------------

## 🧠 LangChain & Agents

  Élément                              Description
  ------------------------------------ ------------------------------------------------
  **LangChain 0.3.x**                  Framework d'orchestration des agents IA
  **LangGraph (2025)**                 Future version graphique des agents
  **Pydantic v2**                      Validation stricte des schémas JSON
  **Groq**                             Fournisseur LLM ultra-rapide hébergeant Llama3
  **FAISS**                            Recherche sémantique (RAG)
  **StructuredTool / AgentExecutor**   Gestion automatique des outils
  **as_messages()**                    Conversion historique utilisateur ↔ IA

------------------------------------------------------------------------

## 🧮 Compatibilités techniques (versions validées)

  Composant               Version   Rôle
  ----------------------- --------- ----------------------------
  Python                  3.12      Langage principal
  LangChain               0.3.27    Framework principal
  LangChain-Core          0.3.78    Gestion interne des agents
  LangChain-Groq          0.3.8     Intégration Groq
  LangChain-Community     0.3.31    FAISS, outils RAG
  LangChain-Ollama        0.3.10    Support IA locale
  Pydantic                2.12.x    Schémas structurés
  FAISS                   1.12.0    Vector store
  Streamlit               1.50.0    Interface
  Torch                   2.2.2     Support embeddings
  Sentence-Transformers   3.0.1     Génération d'embeddings
  Transformers            4.41.1    Modèles HF
  Groq SDK                0.32.0    API officielle

------------------------------------------------------------------------

## 👤 Auteur

**Idriss Beman Kara**\
🎓 Master 2 Data Science & Risk Banking / INSSEDS\
🏢 Datakori / AI Smart Credit Initiative\
🌍 Côte d'Ivoire -- Paris\
📧 <contact@datakori.com>

> 🧩 Projet de recherche et d'innovation appliquée : "Scoring de crédit
> inclusif basé sur l'intelligence artificielle".

------------------------------------------------------------------------

## 🏁 Licence

Ce projet est distribué sous licence **MIT** pour encourager la
recherche ouverte et la collaboration académique.
