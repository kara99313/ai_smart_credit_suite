# utils/rag_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Iterable, Optional, Tuple, List

import json
import pandas as pd
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma

PROJECT_ROOT = Path.cwd()
DOCS_DIR     = PROJECT_ROOT / "docs_rag"
DATA_DIR     = PROJECT_ROOT / "data"
VSTORE_DIR   = PROJECT_ROOT / "vector_store"
VSTORE_DIR.mkdir(parents=True, exist_ok=True)

FAISS_DOCS_PATH   = VSTORE_DIR / "faiss_docs"
FAISS_HYBRID_PATH = VSTORE_DIR / "faiss_hybrid"
CHROMA_DOCS_PATH  = str(VSTORE_DIR / "chroma_docs")
CHROMA_HYB_PATH   = str(VSTORE_DIR / "chroma_hybrid")

def _try_import_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
        return HuggingFaceEmbeddings
    except Exception as e:
        print("[RAG] embeddings indisponibles:", str(e))
        return None

def _embeddings():
    HF = _try_import_embeddings()
    if not HF:
        return None
    try:
        return HF(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        print("[RAG] chargement embeddings echoue:", str(e))
        return None

def _read_txt(path: Path) -> str:  return path.read_text(encoding="utf-8", errors="ignore")
def _read_md(path: Path) -> str:   return path.read_text(encoding="utf-8", errors="ignore")

def _read_pdf(path: Path) -> str:
    try:
        import pypdf
        reader = pypdf.PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception:
        return ""

def _read_docx(path: Path) -> str:
    try:
        import docx2txt
        return docx2txt.process(str(path)) or ""
    except Exception:
        return ""

def _iter_source_texts(folder: Path) -> Iterable[Tuple[str, str]]:
    if not folder.exists():
        return
    for p in folder.rglob("*"):
        if not p.is_file():
            continue
        suf = p.suffix.lower()
        text = ""
        if suf == ".txt": text = _read_txt(p)
        elif suf == ".md": text = _read_md(p)
        elif suf == ".pdf": text = _read_pdf(p)
        elif suf == ".docx": text = _read_docx(p)
        if text.strip():
            yield (str(p.relative_to(folder)), text)

def _split_text(text: str) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=150, separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)

def _build_docs_from_files() -> List[Document]:
    docs: List[Document] = []
    for src, fulltxt in _iter_source_texts(DOCS_DIR):
        for chunk in _split_text(fulltxt):
            docs.append(Document(page_content=chunk, metadata={"source": src}))
    return docs

def _build_docs_hybrid(client_filter: Optional[str] = None, max_logs: int = 500) -> List[Document]:
    """Docs RAG + historique des prédictions (chaque ligne devient un Document)."""
    docs = _build_docs_from_files()
    log_path = DATA_DIR / "predictions_log.csv"
    if log_path.exists():
        try:
            df = pd.read_csv(log_path).sort_values("timestamp", ascending=False).head(int(max_logs))
            if client_filter:
                df = df[df["client_id"].astype(str).str.contains(client_filter, na=False)]
            for _, r in df.iterrows():
                pdv = float(r.get("prob_default", 0.0))
                score = round((1 - pdv) * 1000)
                thr = float(r.get("threshold", 0.10))
                decision = "ACCEPT" if pdv < thr else "REVIEW/REJECT"
                # on injecte aussi quelques features pour la recherche
                try:
                    inputs = json.loads(r.get("inputs_json","{}")) or {}
                except Exception:
                    inputs = {}
                keys = ["DTIRatio","Income","LoanAmount","InterestRate","MonthsEmployed","Age",
                        "CommunityGroupMember","HasMortgage","HasSocialAid","MobileMoneyTransactions"]
                pairs = " | ".join([f"{k}={inputs.get(k,'?')}" for k in keys if k in inputs])
                snippet = (
                    f"[LOG] {r.get('timestamp','')} | client={r.get('client_id','N/A')} | "
                    f"PD={pdv:.2%} | Score={score}/1000 | Seuil={thr:.2%} | Décision={decision} | "
                    f"{pairs}"
                )
                docs.append(Document(page_content=snippet, metadata={"source":"predictions_log.csv"}))
        except Exception as e:
            print("[RAG] lecture logs KO:", e)
    return docs

def _build_faiss(docs: List[Document], path: Path):
    if not docs:
        return None, path
    emb = _embeddings()
    if not emb:
        print("[RAG] embeddings absents, impossible de construire FAISS.")
        return None, path
    vs = FAISS.from_documents(docs, emb)
    vs.save_local(str(path))
    return vs, path

def _load_faiss(path: Path):
    emb = _embeddings()
    if not emb:
        print("[RAG] embeddings absents, impossible de charger FAISS.")
        return None
    try:
        return FAISS.load_local(str(path), emb, allow_dangerous_deserialization=True)
    except Exception:
        return None

def _build_chroma(docs: List[Document], persist_dir: str):
    if not docs:
        return None
    emb = _embeddings()
    if not emb:
        print("[RAG] embeddings absents, fallback Chroma impossible.")
        return None
    return Chroma.from_documents(docs, emb, persist_directory=persist_dir)

def _load_chroma(persist_dir: str):
    emb = _embeddings()
    if not emb:
        print("[RAG] embeddings absents, impossible de charger Chroma.")
        return None
    try:
        return Chroma(persist_directory=persist_dir, embedding_function=emb)
    except Exception:
        return None

def build_or_load_faiss_docs(rebuild: bool = False):
    if (not rebuild) and FAISS_DOCS_PATH.exists():
        vs = _load_faiss(FAISS_DOCS_PATH)
        if vs: return vs, FAISS_DOCS_PATH
    try:
        docs = _build_docs_from_files()
        return _build_faiss(docs, FAISS_DOCS_PATH)
    except Exception:
        vs = _load_chroma(CHROMA_DOCS_PATH)
        if vs: return vs, Path(CHROMA_DOCS_PATH)
        vs = _build_chroma(_build_docs_from_files(), CHROMA_DOCS_PATH)
        return vs, Path(CHROMA_DOCS_PATH)

def build_or_load_faiss_hybrid(rebuild: bool = False, client_filter: Optional[str] = None):
    if (not rebuild) and FAISS_HYBRID_PATH.exists():
        vs = _load_faiss(FAISS_HYBRID_PATH)
        if vs: return vs, FAISS_HYBRID_PATH
    try:
        docs = _build_docs_hybrid(client_filter=client_filter)
        return _build_faiss(docs, FAISS_HYBRID_PATH)
    except Exception:
        vs = _load_chroma(CHROMA_HYB_PATH)
        if vs: return vs, Path(CHROMA_HYB_PATH)
        vs = _build_chroma(_build_docs_hybrid(client_filter=client_filter), CHROMA_HYB_PATH)
        return vs, Path(CHROMA_HYB_PATH)
