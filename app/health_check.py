"""Explicit, isolated, read-only health probes; never build an index on page load."""
from pathlib import Path
import streamlit as st
from utils.admin_auth import is_admin, render_gate
from utils.runtime_config import ROOT, backend_url
from utils import health_diagnostics as checks

PROJECT_ROOT=ROOT


def show(label,probe):
    st.write(f"**{label} : {probe['state']}**")
    st.caption(f"{probe['explanation']} Dernier controle : {probe['checked_at']} (UTC).")
    details={key:value for key,value in probe.items() if key not in {'state','explanation','checked_at'}}
    if details:
        st.json(checks.redact(details))


def main():
    st.title('Health Check')
    if not render_gate():
        return
    st.caption('Configure seulement = configuration presente ; Disponible = ressource joignable/chargeable ; Fonctionnel = test minimal reel reussi. Non teste ne signifie pas en panne.')
    env=checks.environment_inventory(PROJECT_ROOT)
    st.subheader('Environnement et configuration (aucun appel externe)')
    st.json(env)
    st.caption("BACKEND_URL puis API_BASE, puis config.yaml et adresse de repli : resolution commune utils.runtime_config. .env est charge sans ecraser les variables existantes. Versions detectees ne signifie pas imports fonctionnels.")
    if not env['model_readable']:
        st.warning('Artefact du modele absent ou inaccessible sur ce poste ; cela ne renseigne pas un backend distant.')
    state_key='health_checks'
    state=st.session_state.setdefault(state_key,{})
    try:
        base=backend_url()
    except Exception:
        base=None
    # Never show a result obtained for a different URL or a different local project.
    identity=(base,str(PROJECT_ROOT))
    if state.get('identity')!=identity:
        state={'identity':identity}
        st.session_state[state_key]=state
    st.subheader('FastAPI et modele')
    if st.button('Tester GET /health',disabled=base is None):
        state['api']=checks.backend_check(base)
    if 'api' in state:
        show('FastAPI',state['api'])
        declared=state['api'].get('model_loaded')
        st.write('Modele charge :', 'declaration model_loaded=True du backend' if declared is True else 'non confirme')
    else:
        st.write('FastAPI : Configure seulement' if base else 'FastAPI : Indisponible (configuration invalide)')
        st.write('Modele charge : Non teste')
    st.caption('/health renvoie actuellement model_loaded=True sans nouveau test d inference. Le warm-up intercepte ses erreurs ; /health ne certifie donc pas une inference reussie.')
    if st.button('Tester une inference synthetique',disabled=base is None):
        state['inference']=checks.inference_check(base)
    if 'inference' in state:
        show('Inference du modele',state['inference'])
    else:
        st.write('Inference du modele : Non teste')
    st.caption('Test explicite via /api/predict, donnees synthetiques non journalisees. Aucun ajout au CSV ni au JSONL.')
    st.subheader('Fournisseurs LLM')
    inventory=checks.provider_inventory()
    st.dataframe(inventory,hide_index=True,width='stretch')
    st.caption('Groq, OpenAI et Ollama sont supportes. Une cle presente ou un connecteur installe ne prouve pas une reponse. Un test ci-dessous peut consommer un petit quota (budget minimal de 8 tokens (128 pour les modeles a raisonnement), sans retry).')
    if not is_admin():
        st.caption('Connexion administrateur requise dans Administration pour tester un fournisseur.')
    provider=st.selectbox('Fournisseur a tester',list(checks.PROVIDERS))
    if st.button('Tester le fournisseur selectionne (appel reel)', disabled=not is_admin()):
        state['llm_'+provider]=checks.provider_check(provider)
    for item in inventory:
        name=item['provider']
        if 'llm_'+name in state:
            show(name,state['llm_'+name])
        else:
            st.write(f"{name} : {item['state']} ; appel fonctionnel Non teste")
    st.subheader('RAG / FAISS')
    show('Corpus documentaire',checks.corpus_inventory(PROJECT_ROOT))
    index_name=st.selectbox('Index a diagnostiquer',['faiss_docs','faiss_hybrid'])
    path=Path(PROJECT_ROOT)/'vector_store'/index_name
    present=(path/'index.faiss').is_file() and (path/'index.pkl').is_file()
    st.write('Fichiers index : Configure seulement (paire presente)' if present else 'Fichiers index : Indisponible (paire absente/incomplete)')
    if st.button('Charger et verifier l index existant'):
        state['index_'+index_name]=checks.index_check(path,PROJECT_ROOT,hybrid=index_name=='faiss_hybrid')
    if 'index_'+index_name in state:
        show('Chargement FAISS',state['index_'+index_name])
    else:
        st.write('Chargement FAISS : Non teste')
    if st.button('Tester le retrieval local sans telechargement'):
        state['retrieval_'+index_name]=checks.retrieval_check(path)
    if 'retrieval_'+index_name in state:
        show('Retrieval RAG',state['retrieval_'+index_name])
    else:
        st.write('Retrieval RAG : Non teste')
    st.caption("Aucun build_or_load, reconstruction, sauvegarde ou telechargement. Metadata chargees avec une liste restreinte de classes. "
               "Sources verifiees seulement par existence ; fraicheur, identite des embeddings et pertinence restent A verifier. La generation de reponse RAG n'est pas testee.")


if __name__=='__main__':
    main()
