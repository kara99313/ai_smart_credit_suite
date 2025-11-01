# app/integrations.py
from __future__ import annotations
import os
import streamlit as st
from utils.api_client import ping_backend, predict_credit_api
from utils.settings import get_settings, save_settings

# ---------------- Utils ----------------

def _get_secret(k: str, default: str = "") -> str:
    """Lit d'abord st.secrets (Cloud), sinon ENV (local)."""
    try:
        return st.secrets.get(k, default)
    except Exception:
        return os.getenv(k, default)

# -------------- API tab -----------------

def _form_default_values():
    return dict(
        DTIRatio=0.35,
        TrustScorePsychometric=0.62,
        HouseholdSize=4,
        NumCreditLines=2,
        Income=300000.0,
        CommunityGroupMember=False,
        HasMortgage=False,
        MonthsEmployed=36,
        HasSocialAid=False,
        MobileMoneyTransactions=120,
        Age=32,
        InterestRate=12.0,
        LoanTerm=24,
        LoanAmount=800000.0,
        InformalIncome=50000.0,
        threshold=0.10,
    )

def _api_tab():
    st.subheader("API de prédiction (backend FastAPI)")
    ok, msg = ping_backend()
    st.info(f"État backend : {'✅ en ligne' if ok else '❌ hors ligne'} — {msg}")

    with st.form("api_predict_form"):
        st.caption("Renseigne les 15 variables puis envoie au backend `/api/predict`.")
        c1, c2, c3 = st.columns(3)
        defaults = _form_default_values()

        with c1:
            DTIRatio = st.number_input("DTI (Dette/Revenu)", 0.0, 5.0, defaults["DTIRatio"], 0.01)
            Trust = st.number_input("Score psychométrique", 0.0, 1.0, defaults["TrustScorePsychometric"], 0.01)
            HH = st.number_input("Taille foyer", 1, 20, defaults["HouseholdSize"], 1)
            Lines = st.number_input("Lignes de crédit actives", 0, 50, defaults["NumCreditLines"], 1)
            Income = st.number_input("Revenu principal", 0.0, 1e12, defaults["Income"], 1000.0)
        with c2:
            CGM = st.selectbox("Membre groupe communautaire", ["Non", "Oui"])
            Mort = st.selectbox("Prêt immobilier (hypothèque)", ["Non", "Oui"])
            MonthsEmp = st.number_input("Mois d’emploi actuel", 0, 600, defaults["MonthsEmployed"], 1)
            Aid = st.selectbox("Aide sociale", ["Non", "Oui"])
            MMTx = st.number_input("Transactions Mobile Money", 0, 1000000, defaults["MobileMoneyTransactions"], 1)
        with c3:
            Age = st.number_input("Âge", 18, 90, defaults["Age"], 1)
            IR = st.number_input("Taux d’intérêt (%)", 0.0, 100.0, defaults["InterestRate"], 0.1)
            Term = st.number_input("Durée du prêt (mois)", 1, 360, defaults["LoanTerm"], 1)
            Amount = st.number_input("Montant du prêt", 0.0, 1e13, defaults["LoanAmount"], 1000.0)
            Informal = st.number_input("Revenu informel estimé", 0.0, 1e12, defaults["InformalIncome"], 1000.0)

        thr = st.slider("Seuil décision (PD)", 0.01, 0.50, defaults["threshold"], 0.01)
        submitted = st.form_submit_button("📡 Appeler l’API")

    if submitted:
        payload = dict(
            DTIRatio=float(DTIRatio),
            TrustScorePsychometric=float(Trust),
            HouseholdSize=int(HH),
            NumCreditLines=int(Lines),
            Income=float(Income),
            CommunityGroupMember=(CGM == "Oui"),
            HasMortgage=(Mort == "Oui"),
            MonthsEmployed=int(MonthsEmp),
            HasSocialAid=(Aid == "Oui"),
            MobileMoneyTransactions=int(MMTx),
            Age=int(Age),
            InterestRate=float(IR),
            LoanTerm=int(Term),
            LoanAmount=float(Amount),
            InformalIncome=float(Informal),
            threshold=float(thr),
        )
        try:
            res = predict_credit_api(payload)
            st.success("Réponse API reçue.")
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("PD", f"{res['pd']*100:.2f}%")
            with c2: st.metric("Score (0–1000)", f"{res['score_1000']}")
            with c3: st.metric("Notation", res["rating"])
            with c4: st.metric("Décision", res["decision"])
            with st.expander("Payload envoyé"): st.json(payload)
            with st.expander("Réponse brute"): st.json(res)
        except Exception as e:
            st.error(f"Échec d’appel API : {e}")

# -------------- Web tab -----------------

def _web_tab():
    st.subheader("Recherche Web & clés")
    cfg = get_settings()
    st.caption("Ces réglages sont lus par les chatbots lorsqu’on active la recherche web.")

    # valeurs par défaut robustes (pas d'attribut -> valeur vide)
    serp_default = _get_secret("SERPAPI_KEY", "") or getattr(cfg, "serpapi_key", "")
    enable_default = bool(getattr(cfg, "enable_web_search", False))
    allowed_default = getattr(cfg, "allowed_domains", []) or []

    with st.form("web_settings"):
        enable = st.toggle("Activer la recherche Web", value=enable_default)
        serp = st.text_input("Clé SERPAPI (optionnel)", value=serp_default, type="password")
        domains = st.text_area("Domaines autorisés (un par ligne)", value="\n".join(allowed_default), height=120)
        saved = st.form_submit_button("💾 Enregistrer")
    if saved:
        # sauve « en douceur » même si l’objet n’avait pas les attributs
        setattr(cfg, "enable_web_search", bool(enable))
        setattr(cfg, "serpapi_key", serp.strip())
        setattr(cfg, "allowed_domains", [d.strip() for d in domains.splitlines() if d.strip()])
        save_settings(cfg)
        st.success("Paramètres Web enregistrés.")

# -------------- LLM tab -----------------

def _llm_tab():
    st.subheader("LLM — Ollama (local) & OpenAI (optionnel)")
    cfg = get_settings()
    st.write(f"**Provider configuré** : `{getattr(cfg, 'llm_provider', '—')}`")
    st.write(f"**Ollama** : {getattr(cfg, 'ollama_base_url', 'http://127.0.0.1:11434')} — **Model** : {getattr(cfg, 'ollama_model', 'llama3.2:3b')}")

    # Ping Ollama (info)
    import requests
    try:
        base = getattr(cfg, "ollama_base_url", "http://127.0.0.1:11434")
        r = requests.get(base.rstrip("/") + "/api/version", timeout=2.5)
        st.success(f"Ollama disponible — version: {r.text.strip()[:50]}")
    except Exception as e:
        st.warning(f"Ollama indisponible: {e}")

# -------------- Main --------------------

def main():
    st.title("🔗 Intégrations & API")
    t1, t2, t3 = st.tabs(["🛰️ API de prédiction", "🌐 Web & Clés", "🧠 LLM"])
    with t1: _api_tab()
    with t2: _web_tab()
    with t3: _llm_tab()

if __name__ == "__main__":
    main()
