# app/integrations.py
from __future__ import annotations
import streamlit as st
from utils.admin_auth import is_admin
from utils.api_client import predict_credit_api
from utils import health_diagnostics as checks
from utils.runtime_config import backend_url
from utils.llm_runtime import explain, select_provider
from utils.integration_services import effective_llm_identity
from utils.settings import get_settings, save_settings

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
    try:
        target = backend_url()
        st.caption('URL effective : ' + checks.redact(target))
    except Exception:
        target = None
        st.warning('Configuration backend invalide.')
    state = st.session_state.setdefault('integration_api_check', {})
    if state.get('target') != target:
        state.clear()
        state['target'] = target
    if st.button('Tester la connexion FastAPI', disabled=target is None):
        state['probe'] = checks.backend_check(target)
    if 'probe' in state:
        st.write(state['probe']['state'] + ' : ' + state['probe']['explanation'])
    else:
        st.caption('Connexion non testee ; aucun appel automatique.')

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
            res = predict_credit_api(payload, timeout=30.0)
            st.success("Réponse API reçue.")
            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("PD", f"{res['pd']*100:.2f}%")
            with c2: st.metric("Score (0–1000)", f"{res['score_1000']}")
            with c3: st.metric("Notation", res["rating"])
            with c4: st.metric("Décision", res["decision"])
            with st.expander("Payload envoyé"): st.json(payload)
            with st.expander("Réponse brute"): st.json(checks.redact(res))
        except Exception as e:
            st.error("Echec d appel API : " + explain(e))

# -------------- Web tab -----------------

def _web_tab():
    from utils.web_search import web_configuration, normalize_domains, search_web, WebSearchError
    st.subheader('Recherche Web - Tavily / DuckDuckGo')
    st.caption('Tavily est prioritaire si TAVILY_API_KEY est configuree. DuckDuckGo prend le relais en cas d absence ou d indisponibilite de Tavily.')
    try:
        cfg = get_settings()
        effective = web_configuration()
    except Exception:
        st.error('Configuration Web invalide ; recherche indisponible.')
        return
    st.write('Recherche effective : ' + ('activee' if effective['enabled'] else 'desactivee'))
    st.caption('Activation globale pour tous les appels du connecteur, y compris ceux des chatbots et agents.')
    st.caption('WEB_SEARCH_ENABLED et WEB_ALLOWED_DOMAINS (separes par des virgules) sont prioritaires sur les preferences. WEB_TIMEOUT borne chaque requete. Une liste de domaines vide autorise tous les domaines HTTP(S).')
    with st.form('web_settings'):
        enable = st.toggle('Activer la recherche Web', value=cfg.enable_web_search)
        domains = st.text_area('Domaines autorises (un par ligne)', value='\n'.join(cfg.allowed_domains))
        saved = st.form_submit_button('Enregistrer les parametres Web')
    if saved:
        try:
            cfg.allowed_domains = normalize_domains([d.strip() for d in domains.splitlines() if d.strip()])
            cfg.enable_web_search = bool(enable)
            save_settings(cfg)
            st.session_state.pop('integration_web_test', None)
            st.success('Preferences Web enregistrees.')
            st.rerun()
        except ValueError as error:
            st.error(str(error))
        except Exception:
            st.error('Enregistrement impossible ; verifier les droits des preferences.')
    st.json(dict(domaines_effectifs=effective['domains'], timeout=effective['timeout']))
    query = st.text_input('Requete de test Web')
    identity = (query, repr(effective))
    if st.button('Tester la recherche Web', disabled=not effective['enabled'] or not query.strip()):
        try:
            hits = search_web(query, max_results=2)
            result = ('Resultats Web recus.' if hits else 'Aucun resultat exploitable apres filtrage ; fonctionnement non confirme.', hits)
            provider = getattr(hits, 'provider', None)
            if provider:
                result = (result[0] + f' Fournisseur : {provider}.', hits)
        except WebSearchError as error:
            result = (f'Recherche Web indisponible : {error}', [])
        except Exception:
            result = ('Recherche Web indisponible : erreur reseau, HTTP, timeout ou reponse invalide.', [])
        st.session_state['integration_web_test'] = (identity, result)
    previous = st.session_state.get('integration_web_test')
    if previous and previous[0] == identity:
        st.write(previous[1][0])
        if previous[1][1]:
            st.json(previous[1][1])


def _llm_tab():
    st.subheader('Fournisseurs LLM - Groq, OpenAI et Ollama')
    st.caption('La configuration seule ne confirme pas une reponse. Aucun appel fournisseur a l ouverture de la page. Modifier la configuration dans Administration.')
    inventory = checks.provider_inventory()
    st.dataframe(inventory, hide_index=True)
    try:
        current = select_provider()
        st.write('Fournisseur effectif : ' + current)
    except Exception:
        current = 'groq'
        st.warning('Configuration fournisseur invalide.')
    names = list(checks.PROVIDERS)
    if not is_admin():
        st.caption('Connexion administrateur requise dans Administration pour tester un fournisseur.')
    provider = st.selectbox('Fournisseur a tester', names, index=names.index(current))
    identity = (provider, effective_llm_identity())
    if st.button('Tester le fournisseur (appel reel)', disabled=not is_admin()):
        st.session_state['integration_llm_test'] = (identity, checks.provider_check(provider))
    st.caption('Test explicite avec budget minimal ; peut consommer du quota. Les erreurs ne declenchent aucun basculement vers un autre fournisseur.')
    previous = st.session_state.get('integration_llm_test')
    if previous and previous[0] == identity:
        result = previous[1]
        st.write(result['state'] + ' : ' + result['explanation'])

# -------------- Main --------------------

def main():
    st.title("🔗 Intégrations & API")
    t1, t2, t3 = st.tabs(["🛰️ API de prédiction", "🌐 Web & Clés", "🧠 LLM"])
    with t1: _api_tab()
    with t2: _web_tab()
    with t3: _llm_tab()

if __name__ == "__main__":
    main()
