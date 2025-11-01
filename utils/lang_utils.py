# utils/lang_utils.py
from __future__ import annotations

def detect_lang(text: str) -> str:
    """
    Retourne 'fr' ou 'en'. Utilise langdetect si dispo, sinon heuristique.
    """
    try:
        from langdetect import detect
        code = detect(text or "")
        return "fr" if code.startswith("fr") else "en"
    except Exception:
        t = (text or "").lower()
        if any(w in t for w in [" the ", " what ", " how ", " loan ", "interest", "risk "]):
            return "en"
        return "fr"

def system_prompt_for(lang: str = "fr") -> str:
    lang = (lang or "fr").lower()  # normalisation
    if lang == "fr":
        return (
            "Tu es un assistant RAG fiable. Réponds en français, de manière concise et sourcée.\n"
            "Règles :\n"
            "- Ne réponds qu'avec les informations du contexte fourni.\n"
            "- Si une info n'y est pas, dis-le explicitement.\n"
            "- Cite les extraits utilisés en fin de réponse sous la forme [1], [2], ...\n"
        )
    return (
        "You are a trustworthy RAG assistant. Answer in English, concisely and with sources.\n"
        "Rules:\n"
        "- Only use the information from the given context.\n"
        "- If something is missing, say it explicitly.\n"
        "- Cite used snippets at the end as [1], [2], ...\n"
    )
