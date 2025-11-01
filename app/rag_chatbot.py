# app/rag_chatbot.py
from __future__ import annotations
import os
import streamlit as st
from utils.llm_status import llm_status_badge
from utils.llm_providers import load_llm
from utils.rag_utils import build_or_load_faiss_docs, build_or_load_faiss_hybrid
from utils.web_search import search_web, build_web_context
from utils.context_bridge import build_client_context  # ← client courant

try:
    from utils.lang_utils import detect_lang, system_prompt_for
except Exception:
    def detect_lang(text: str) -> str:
        return "FR" if any(c in text for c in "éèàùçôîï") else "EN"
    def system_prompt_for(lang: str) -> str:
        if lang == "FR":
            return ("Tu es un assistant RAG : réponds STRICTEMENT à partir du CONTEXTE fourni "
                    "(client courant + documents internes + web si présent). Si l'information n’y est pas, dis-le.")
        return ("You are a RAG assistant. Answer STRICTLY from the supplied CONTEXT "
                "(current client + internal docs + optional web). If missing, say so.")

def _style():
    st.markdown(
        """
        <style>
          .rag-title { font-size:1.6rem; font-weight:700; margin-bottom:.25rem; }
          .rag-sub   { color:#6b7280; margin-bottom:1rem; }
          .ctx-block { border:1px solid #e5e7eb; border-radius:10px; padding:12px; background:#fafafa; }
        </style>
        """,
        unsafe_allow_html=True,
    )

def _format_docs_hits(docs):
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "inconnu")
        page = d.metadata.get("page", None)
        tag = f"{src}" + (f" (page {page})" if page is not None else "")
        text = (d.page_content or "").strip().replace("\n", " ")
        if len(text) > 900:
            text = text[:900] + "…"
        parts.append(f"[D{i}] {tag}\n{text}")
    return "\n\n".join(parts)

def main():
    _style()
    st.markdown("<div class='rag-title'>📚 Chatbot RAG — Documents & Hybride + Web</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='rag-sub'>Répond à partir du CONTEXTE CLIENT COURANT, de tes documents internes (FAISS) et, en option, du web.</div>",
        unsafe_allow_html=True,
    )
    llm_status_badge()

    with st.sidebar:
        st.markdown("### Options Web")
        use_web = st.checkbox("🔎 Ajouter des résultats Web", value=False)
        web_k = st.slider("Résultats Web", 1, 6, 2, disabled=not use_web)
        fetch = st.checkbox("Télécharger les pages (résumé propre)", value=False, disabled=not use_web)

        st.divider()
        st.markdown("### Contexte client")
        include_client = st.checkbox("Inclure le **client courant** (dernier log de prédiction)", value=True)
        client_id_opt = st.text_input("Client ID (optionnel, sinon dernier log)", value="")

    mode = st.radio("Index :", ["Hybride (Docs + Logs)", "Docs"], horizontal=True, index=0)

    with st.expander("⚙️ Indexation / Chargement"):
        rebuild = st.checkbox("Reconstruire maintenant", value=False)
        client_filter = st.text_input("Filtre client (Hybride, optionnel)", value="")
        if st.button("Indexer / Charger"):
            try:
                if mode.startswith("Docs"):
                    vs, path = build_or_load_faiss_docs(rebuild=rebuild)
                else:
                    vs, path = build_or_load_faiss_hybrid(rebuild=rebuild, client_filter=client_filter or None)
                st.session_state["rag_vs"] = vs
                st.session_state["rag_idx_path"] = str(path)
                if vs:
                    st.success(f"Index disponible : {path}")
                else:
                    st.warning("Aucune donnée indexée (vérifiez docs_rag/ et/ou data/predictions_log.csv).")
            except Exception as e:
                st.error(f"Indexation/chargement impossible : {e}")

    if "rag_vs" not in st.session_state:
        try:
            if mode.startswith("Docs"):
                vs, path = build_or_load_faiss_docs(rebuild=False)
            else:
                vs, path = build_or_load_faiss_hybrid(rebuild=False)
            st.session_state["rag_vs"] = vs
            st.session_state["rag_idx_path"] = str(path)
        except Exception as e:
            st.warning(f"Index non disponible (encore). Détails : {e}")

    top_k = 3
    q = st.text_input(
        "Votre question (FR/EN) :",
        placeholder="Ex: Parle-moi des 10 dernières prédictions (scores, décisions, tendances).",
    )

    if st.button("Rechercher & Répondre", type="primary") and q.strip():
        vs = st.session_state.get("rag_vs")
        if not vs:
            st.warning("Index absent. Ouvrez '⚙️ Indexation / Chargement' et cliquez sur 'Indexer / Charger'.")
            return

        # 1) Contexte CLIENT
        ctx_client = ""
        if include_client:
            ctx_client, _raw = build_client_context(client_id_opt or None, max_chars=700)

        # 2) Contexte DOCS / LOGS
        try:
            docs = vs.similarity_search(q, k=top_k)
        except Exception as e:
            st.error(f"Erreur RAG (recherche): {e}")
            return
        context_docs = _format_docs_hits(docs) if docs else ""

        # 3) Contexte WEB optionnel
        context_web = ""
        web_hits = []
        if use_web:
            lang = detect_lang(q)
            try:
                web_hits = search_web(q, max_results=web_k, region="fr-fr" if lang == "FR" else "en-us")
                context_web = build_web_context(web_hits, max_chars_total=900, fetch_full=fetch)
            except Exception as e:
                st.warning(f"Recherche web indisponible : {e}")

        blocks = []
        if ctx_client:
            blocks.append("=== CONTEXTE — CLIENT COURANT ===\n" + ctx_client)
        if context_docs:
            blocks.append("=== CONTEXTE — DOCUMENTS INTERNES ===\n" + context_docs)
        if context_web:
            blocks.append("=== CONTEXTE — WEB ===\n" + context_web)

        if not blocks:
            st.info("Aucun contexte disponible. Ajoutez des documents dans docs_rag/ ou lancez des prédictions (logs).")
            return

        full_context = "\n\n".join(blocks)

        lang = detect_lang(q)
        system = system_prompt_for(lang)
        human = (
            "Tu dois répondre STRICTEMENT avec les informations du CONTEXTE ci-dessous. "
            "S’il manque une info, dis-le. Cite les sources comme [C] pour client courant, [D1] pour documents internes, "
            "et [W1] pour le web.\n\n"
            f"{full_context}\n\nQUESTION:\n{q}"
        )

        try:
            llm = load_llm()
            resp = llm.invoke(
                [
                    {"type": "system", "content": system},
                    {"type": "human", "content": human},
                ]
            )
            answer = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            answer = f"Erreur LLM : {e}"

        st.subheader("Réponse")
        st.write(answer)

        with st.expander("📎 Sources"):
            if ctx_client:
                st.markdown("**Client courant**  \n[C] Voir bloc *CONTEXTE — CLIENT COURANT* ci-dessus.")
            if docs:
                st.markdown("**Documents internes**")
                for i, d in enumerate(docs, 1):
                    src = d.metadata.get("source", "inconnu")
                    page = d.metadata.get("page", None)
                    tag = f"{src}" + (f" (page {page})" if page is not None else "")
                    st.markdown(f"**[D{i}]** {tag}")
            if web_hits:
                st.markdown("**Web**")
                for i, r in enumerate(web_hits, 1):
                    st.markdown(f"**[W{i}] {r['title']}**  \n{r['href']}  \n_{r['body'][:220]}…_")

        with st.expander("🧩 Contexte donné au LLM (debug)"):
            st.markdown("<div class='ctx-block'>", unsafe_allow_html=True)
            st.text(full_context)
            st.markdown("</div>", unsafe_allow_html=True)

    if os.environ.get("LLM_PROVIDER", "ollama").lower() == "ollama":
        st.caption("💡 Si vous voyez “WinError 10061”, lancez Ollama : `ollama serve` puis `ollama run llama3.2:3b`.")
        
if __name__ == "__main__":
    main()
