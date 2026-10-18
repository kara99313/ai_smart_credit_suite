"""Read-only diagnostics. No index builders, implicit provider fallback or secret output."""
from __future__ import annotations
import importlib.util
import importlib.metadata
import math
import os
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
import requests
from utils.runtime_config import ROOT, backend_url
from utils.api_client import predict_credit_api

from utils.llm_runtime import PROVIDERS, build_client, invoke, explain
TEST_INPUT = dict(DTIRatio=.35,TrustScorePsychometric=.62,HouseholdSize=4,
    NumCreditLines=2,Income=300000.,CommunityGroupMember=False,HasMortgage=False,
    MonthsEmployed=36,HasSocialAid=False,MobileMoneyTransactions=120,Age=32,
    InterestRate=12.,LoanTerm=24,LoanAmount=800000.,InformalIncome=50000.,threshold=.10)


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def get_setting(name, default=''):
    try:
        import streamlit as st
        value = st.secrets.get(name)
        if value is not None:
            return str(value)
    except Exception:
        pass
    return os.getenv(name,default)


def secrets():
    names = set(name for name in os.environ if any(part in name.upper() for part in ('KEY','TOKEN','SECRET','PASSWORD')))
    names.update(('GROQ_API_KEY','OPENAI_API_KEY','SERPAPI_KEY'))
    return [get_setting(name) for name in names if get_setting(name)]


def redact(value):
    if isinstance(value,dict):
        return {str(key): ('[masque]' if any(word in str(key).lower() for word in ('key','token','secret','password','authorization')) else redact(item))
                for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [redact(item) for item in value]
    if isinstance(value,str):
        for secret in sorted(secrets(),key=len,reverse=True):
            value=value.replace(secret,'[masque]')
        value=re.sub(r'(https?://)[^/\s@]+@',r'\1[masque]@',value)
        value=re.sub(r'(?i)([?&](?:api_key|key|token|password|secret)=)[^&\s]+',r'\1[masque]',value)
        return value[:4000]
    return value if value is None or isinstance(value,(bool,int,float)) else '[type non affichable]'


def result(state, explanation, **details):
    return dict(state=state,explanation=explanation,checked_at=now(),**details)


def backend_check(base=None):
    try:
        target=backend_url(base)
        response=requests.get(target+'/health',timeout=4)
        try: body=response.json()
        except (ValueError,TypeError): body={'response':'Corps non JSON ; non affiche.'}
        valid=response.status_code==200 and isinstance(body,dict) and body.get('status')=='ok'
        state='Disponible' if valid else 'Indisponible'
        return result(state,'GET /health reussi ; aucune inference testee.' if valid else 'Reponse HTTP ou schema /health inattendu.',
                      http_status=response.status_code,payload=redact(body),
                      model_loaded=body.get('model_loaded') if isinstance(body,dict) and isinstance(body.get('model_loaded'),bool) else None)
    except Exception:
        return result('Indisponible','Connexion impossible, delai depasse ou configuration API invalide.',http_status=None,model_loaded=None)


def inference_check(base=None):
    # /api/predict does not persist a row: only the Streamlit prediction page logs it.
    try:
        response=predict_credit_api(dict(TEST_INPUT),base=backend_url(base),timeout=10)
        return result('Fonctionnel','Inference synthetique via /api/predict reussie ; aucun enregistrement historique.',
                      pd=response['pd'],model_id=response.get('model_id'))
    except Exception:
        return result('Indisponible','Echec de l inference synthetique (connexion, chargement ou probabilite invalide).')


def provider_inventory():
    rows=[]
    for name,(key_name,model_name,default,module,_) in PROVIDERS.items():
        key_present=bool(get_setting(key_name).strip()) if key_name else None
        model=get_setting(model_name,default).strip()
        base=get_setting('OLLAMA_BASE_URL','http://127.0.0.1:11434') if name=='ollama' else None
        configured=bool(model) and (key_present if key_name else bool(base))
        try: connector=importlib.util.find_spec(module) is not None
        except Exception: connector=False
        rows.append(dict(provider=name,key_present=key_present,configured=configured,
                         connector=connector,model=redact(model),
                         state='Configure seulement' if configured else 'Indisponible'))
    return rows


from utils.admin_auth import sensitive


@sensitive
def provider_check(name):
    if name not in PROVIDERS:
        return result('Indisponible','Fournisseur non supporte.')
    key_name,model_name,default,module,class_name=PROVIDERS[name]
    key=get_setting(key_name).strip() if key_name else None
    if key_name and not key:
        return result('Indisponible','Cle absente ; aucun appel effectue.')
    try:
        llm=build_client(name,getter=get_setting,max_tokens=8,timeout=240 if name=='ollama' else 60)
        response=invoke(llm,'Reply only with OK.',name)
        content=getattr(response,'content',None)
        if not isinstance(content,str) or not content.strip():
            return result('Indisponible','Appel termine sans reponse textuelle exploitable.')
        return result('Fonctionnel','Reponse textuelle recue ; contenu non affiche. Ce test ne valide pas un agent.')
    except Exception as error:
        return result('Indisponible',explain(error))


def corpus_inventory(root=ROOT):
    folder=Path(root)/'docs_rag'
    try:
        files=[path for path in folder.rglob('*') if path.is_file() and path.suffix.lower() in {'.txt','.md','.pdf','.docx'}]
        return result('Disponible' if files else 'Indisponible',
                      'Fichiers sources presents ; extraction de texte non testee.' if files else 'Corpus absent ou sans document supporte.',
                      count=len(files))
    except OSError:
        return result('Indisponible','Corpus inaccessible.',count=0)


class _DocstoreReader(pickle.Unpickler):
    """Only the two data classes serialized by this project's LangChain FAISS store."""
    def find_class(self,module,name):
        if (module,name) not in {
            ('langchain_community.docstore.in_memory','InMemoryDocstore'),
            ('langchain_core.documents.base','Document')}:
            raise pickle.UnpicklingError('Unsupported docstore class')
        return super().find_class(module,name)


def _load_store(path):
    import faiss
    import numpy as np
    # Reading bytes avoids Windows FAISS path issues with non-ASCII project paths.
    index=faiss.deserialize_index(np.frombuffer((Path(path)/'index.faiss').read_bytes(),dtype='uint8'))
    with (Path(path)/'index.pkl').open('rb') as stream:
        store,mapping=_DocstoreReader(stream).load()
    if not isinstance(mapping,dict) or set(mapping)!=set(range(index.ntotal)) or not index.ntotal:
        raise ValueError('Index mapping mismatch or empty index')
    if len(mapping)!=len(set(mapping.values())):
        raise ValueError('Duplicate document identifiers')
    for identifier in mapping.values():
        document=store.search(identifier)
        if not isinstance(getattr(document,'page_content',None),str) or not document.page_content.strip():
            raise ValueError('Missing document content')
    return index,store,mapping


def index_check(path,root=ROOT,hybrid=False):
    path=Path(path)
    if not (path/'index.faiss').is_file() or not (path/'index.pkl').is_file():
        return result('Indisponible','Paire index.faiss / index.pkl absente ou incomplete.')
    try:
        index,store,mapping=_load_store(path)
        missing=unknown=0
        for identifier in mapping.values():
            source=store.search(identifier).metadata.get('source')
            if not source:
                unknown+=1;continue
            if hybrid and source=='predictions_log.csv':
                exists=(Path(root)/'data/predictions_log.csv').is_file()
            else:
                source_path=Path(source)
                source_path=source_path if source_path.is_absolute() else Path(root)/'docs_rag'/source_path
                exists=source_path.is_file()
            missing+=not exists
        return result('Disponible','Index et documents charges ; recherche semantique non testee.',
                      vectors=int(index.ntotal),dimension=int(index.d),missing_sources=int(missing),unknown_sources=unknown,
                      consistency='Sources manquantes detectees.' if missing else 'Existence des sources verifiee ; fraicheur et embeddings A verifier (aucun manifeste de reference).')
    except Exception:
        return result('Indisponible','Index invalide/inaccessible, metadata incompatibles ou dependance manquante.')


def retrieval_check(path):
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        index,store,mapping=_load_store(path)
        # Same embedding name and normalization as utils.rag_utils._embeddings.
        # Local cache only: no download, rebuild, fit, save or external service.
        encoder=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',device='cpu',local_files_only=True)
        vector=encoder.encode(['gestion du risque de credit'],normalize_embeddings=True)
        vector=np.asarray(vector,dtype='float32')
        if vector.shape!=(1,index.d) or not np.isfinite(vector).all():
            raise ValueError('Embedding dimensions mismatch')
        scores,identifiers=index.search(vector,min(3,index.ntotal))
        hits=[int(identifier) for identifier in identifiers[0] if int(identifier) in mapping]
        if not hits or not np.isfinite(scores[0]).all():
            raise ValueError('No retrieval result')
        if not all(store.search(mapping[identifier]).page_content.strip() for identifier in hits):
            raise ValueError('Empty retrieval')
        return result('Fonctionnel','Recherche semantique locale reussie ; ni pertinence metier ni generation LLM validees.',
                      hits=len(hits),embedding_identity='Compatibilite dimensionnelle verifiee ; identite historique des embeddings A verifier.')
    except Exception:
        return result('Indisponible','Retrieval impossible : cache embeddings local absent/incompatible, dependance, index ou documents invalides. Aucun telechargement tente.')


def environment_inventory(root=ROOT):
    root=Path(root)
    model=Path(os.getenv('MODEL_PATH') or 'model/logistic_pipeline_best.pkl')
    model=model if model.is_absolute() else root/model
    def readable(path):
        try:
            with path.open('rb') as stream: stream.read(1)
            return True
        except OSError: return False
    try:
        api_url=redact(backend_url())
        api_configured=True
    except Exception:
        api_url='Indisponible'
        api_configured=False
    versions={}
    for distribution in ('numpy','pandas','scikit-learn','joblib','fastapi','requests','faiss-cpu','sentence-transformers',
                         'langchain-groq','langchain-openai','langchain-ollama'):
        try: versions[distribution]=importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError: versions[distribution]='Non detectee'
    return dict(api_configured=api_configured,api_url=api_url,api_source=('BACKEND_URL (env/.env)' if os.getenv('BACKEND_URL') else 'API_BASE (env/.env)' if os.getenv('API_BASE') else 'config.yaml / repli'),model_path=redact(str(model)),dotenv_present=(root/'.env').is_file(),
                model_readable=readable(model),history_readable=readable(root/'data/predictions_log.csv'),
                versions=versions)
