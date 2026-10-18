# Reproductibilit? et corpus facultatif

Le profil minimal d?marre sans secret ni corpus. Une recherche RAG sans index/cache
pr?sente une erreur explicite, sans reconstruction ou t?l?chargement automatique.
`docs_rag/test.txt` est un exemple technique, pas le corpus r?glementaire.

## Installer le corpus

Obtenir s?par?ment les documents aupr?s des ?diteurs officiels BCEAO/Commission
bancaire de l?UMOA et BIS. V?rifier pour chaque document version, provenance et droits
d?usage/redistribution avant tout partage. Aucun droit n?est pr?sum? pour les sept PDF
locaux exclus. Une URL de t?l?chargement ne constitue pas une licence.

Dans un environnement priv?, placer uniquement les documents autoris?s dans `docs_rag/`
(formats PDF, TXT, MD, DOCX), avec des noms stables. Ne pas y placer ce guide ou des
historiques clients. Conserver hors Git un manifeste indiquant source officielle,
date/version, licence ou autorisation et SHA-256 de chaque document. Le manifeste
local `audit/phase16_pdf_inventory.json` identifie les sept fichiers existants sans
redistribuer leur contenu ; il ne prouve ni leur origine ni leurs droits.

## Reconstruire les index dans un emplacement neuf

Le runtime est volontairement en lecture seule : `rebuild=True` est refus?. Ne pas
r?utiliser les anciens index priv?s. Dans une copie de travail priv?e, apr?s autorisation
des ressources et provisionnement explicite du mod?le d?embeddings, ex?cuter ce script
Python depuis la racine. Adapter `output` vers un dossier neuf ; ne pas ?craser les
ressources surveill?es. Cette proc?dure est document?e, pas ex?cut?e en Phase 16.

```python
from pathlib import Path
import hashlib, json
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from utils.rag_utils import _build_docs_from_files

class LocalEmbeddings(Embeddings):
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',
                                        device='cpu', local_files_only=True)
    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()
    def embed_query(self, text):
        return self.embed_documents([text])[0]

output = Path('rag_builds/new/faiss_docs')
if output.exists():
    raise RuntimeError('Choisir un dossier neuf ; aucun ?crasement autoris?')
documents = _build_docs_from_files()  # d?coupage 900 caract?res, recouvrement 150
if not documents:
    raise RuntimeError('Corpus vide ou extraction impossible')
embeddings = LocalEmbeddings()  # cache pr?install? uniquement ; dimension 384
store = FAISS.from_documents(documents, embeddings)
store.save_local(str(output))
manifest = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path('docs_rag').rglob('*') if p.is_file()}
(output/'corpus-sha256.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
```

Provisionner le cache du mod?le par une op?ration s?par?e, explicite, depuis une source
fiable ; conserver sa r?vision et les versions des paquets avec le manifeste. Contr?ler
extraction PDF (OCR non fourni), chunks, m?tadonn?es `source` relatives ? `docs_rag/`,
dimension 384, normalisation, empreintes et r?sultats sur des questions de r?f?rence.
Aucune recherche binaire identique n?est garantie entre environnements.

Apr?s validation, installer la paire `index.faiss`/`index.pkl` dans
`vector_store/faiss_docs/` uniquement dans l?environnement cible priv?. Tester via
Health Check puis RAG sans LLM. Ne jamais charger un pickle d?origine inconnue.
Le runtime v?rifie les fichiers et les sources mais ne certifie pas l?identit? historique
des embeddings ; le manifeste doit ?tre conserv? et v?rifi? s?par?ment.

Le mode hybride exige aujourd?hui ?galement `vector_store/faiss_hybrid/`. Pour une
nouvelle installation, on peut y installer une copie de la paire documentaire valid?e :
l?historique courant est lu s?par?ment au moment de la recherche. Ne pas reconstruire
avec `_build_docs_hybrid` : cette ancienne fonction incorpore des extraits clients et
recalcule certaines valeurs. Ne jamais publier l?ancien `faiss_hybrid`, ni les anciens
`faiss_index*`. Le mode documentaire suffit au profil sans historique.
