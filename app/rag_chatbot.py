"""Retrieval independent of generation; exact, source-validated quotations only."""
import streamlit as st
from utils import rag_service as rag
from utils import health_diagnostics as health
from utils.runtime_config import ROOT
PROJECT_ROOT=ROOT


def display(entry):
    st.subheader('Question')
    st.write(health.redact(entry['retrieved']['question']))
    st.caption(entry['retrieved']['method'])
    if entry['retrieved']['skipped_sources']:
        st.warning(f"{entry['retrieved']['skipped_sources']} passages exclus : source absente ou hors corpus local.")
    if entry.get('generated'):
        st.subheader('Reponse fondee sur les sources')
        st.write(health.redact(entry['generated']['answer']))
        st.caption('Sources citees : '+(', '.join(entry['generated']['used_sources']) or 'Aucune'))
    elif not entry['retrieved']['hits']:st.info(rag.REFUSAL)
    else:st.info('Generation non demandee. Passages disponibles sans LLM.')
    with st.expander('Sources et passages recuperes',expanded=True):
        for hit in entry['retrieved']['hits']:
            st.write(f"[{hit['source_id']}] {hit['source']}")
            if 'page' in hit['metadata']:st.caption(f"Page (metadata originale) : {hit['metadata']['page']}")
            if 'vector_id' in hit:st.caption(f"Chunk dans l index : {hit['vector_id']} ; distance FAISS : {hit['distance']:.4f}")
            if 'record_number' in hit['metadata']:st.caption(f"Occurrence historique : {hit['metadata']['record_number']} ; valeurs archivees non recalculees")
            st.caption(f"Empreinte du passage : {hit['excerpt_sha256'][:16]}")
            st.text(health.redact(hit['text']))


def main():
    st.title('Chatbot RAG / Documents et historique')
    st.caption(rag.CONSISTENCY+' : aucun manifeste ancien ne lie les empreintes actuelles aux embeddings.')
    st.caption('Documents + historique combine deux sources : FAISS documentaire et CSV/JSONL courant ; les anciens extraits vectorises de logs sont exclus. Pas de fusion BM25/vectorielle.')
    st.info('Aucun index construit ou modifie. Les PDF locaux doivent accompagner les index au deploiement. Web et contexte client implicite exclus du parcours documentaire.')
    mode=st.radio('Sources',['Documents','Documents + historique courant'],horizontal=True)
    top_k=st.slider('Passages maximum par source',1,6,3)
    default=health.get_setting('LLM_PROVIDER','groq').lower();options=list(health.PROVIDERS)
    provider=st.selectbox('Fournisseur LLM',options,index=options.index(default) if default in options else 0)
    model=health.get_setting(health.PROVIDERS[provider][1],health.PROVIDERS[provider][2])
    timeout=health.get_setting('OLLAMA_TIMEOUT' if provider=='ollama' else 'LLM_TIMEOUT','240' if provider=='ollama' else '60')
    st.caption(f'Fournisseur : {provider} ; modele : {health.redact(model)} ; delai par lecture : {health.redact(timeout)} s. Aucun repli automatique vers un autre fournisseur.')
    st.caption('Aucun appel LLM au chargement. La generation est une action distincte et peut consommer du quota.')
    query=st.text_input('Votre question')
    history=st.session_state.setdefault('rag_verified_history',[])
    if st.button('Rechercher sans LLM',type='primary'):
        try:
            with st.spinner('Recherche locale dans les index existants...'):
                retrieved=rag.retrieve(query,'docs' if mode=='Documents' else 'hybrid',top_k,PROJECT_ROOT)
            history.append(dict(retrieved=retrieved,generated=None))
        except Exception:
            st.error('Recherche indisponible : question invalide, corpus/index absent ou cache local des embeddings inaccessible. Aucun index reconstruit ni telechargement.')
    if history:
        if st.button('Generer une reponse depuis les passages (appel fournisseur)'):
            with st.spinner('Generation et verification des citations...'):
                history[-1]['generated']=rag.answer(history[-1]['retrieved'],provider)
        display(history[-1])
        if len(history)>1:
            with st.expander('Historique conversationnel (questions independantes)'):
                for entry in history[:-1]:display(entry)
        if st.button('Effacer l historique de conversation'):
            st.session_state['rag_verified_history']=[]
            st.rerun()
    st.caption('Filtre lexical conservateur, puis verification exacte des citations. La pertinence n est pas garantie ; des questions couvertes peuvent etre refusees.')


if __name__=='__main__':main()
