# Authentification interservice

Configurer INTERSERVICE_SECRET avec la meme valeur privee dans le backend et Streamlit (environnement, ou secrets Streamlit pour le client). Aucun secret par defaut. Le backend lit uniquement son environnement. Les valeurs vides, avec espaces ou caracteres de controle, ou non ASCII sont refusees. Generer une valeur aleatoire longue, par exemple avec `python -c "import secrets; print(secrets.token_urlsafe(48))"`, dans un terminal prive. Ne jamais la copier dans le depot ou les journaux.

Le header X-Interservice-Secret protege /api/predict, /api/predict_ping, /api/predict_echo et /tools/*. Une configuration backend invalide renvoie 503 ; un header absent, incorrect ou duplique renvoie 401, avant validation et traitement. /health reste public. Aucune exemption localhost.

Les clients de prediction et explication ajoutent le header uniquement a BACKEND_URL/API_BASE (selon la resolution existante), et refusent les redirections. Une destination explicite differente de la configuration est refusee avant envoi. Ne jamais configurer BACKEND_URL vers un fournisseur externe. Les clients fournisseurs conservent leurs propres credentials et ne recoivent pas ce header.

/tools/predict_credit conserve l'outil et son appel HTTP, avec une destination imposee a http://127.0.0.1:<port> dans un contexte limite a cet appel. Port : PORT, puis API_PORT, puis 8000. Le secret reste obligatoire. Un lancement uvicorn manuel sur un autre port doit renseigner API_PORT ou PORT pour correspondre au port reel. `API_PORT=8000` reste seulement pertinent pour notre configuration locale si necessaire ; ne pas imposer `PORT=8000` sur Render.

/docs, /redoc et /openapi.json renvoient 404 par defaut. ENABLE_LOCAL_API_DOCS=1 autorise ces pages seulement pour une connexion dont le pair ASGI est 127.0.0.1 ou ::1. Ne pas activer cette option derriere un proxy local expose : son adresse ne distingue pas les visiteurs distants. Les routes metier exigent toujours le secret.

## Render : procedure pour la production reelle

Architecture constatee manuellement par l'operateur : `ai_smart_credit_suite-1` (frontend, Web Service public) ? `https://ai-smart-credit-suite.onrender.com` ? `ai_smart_credit_suite` (backend, Web Service public). Les services `ai-smart-api` / `ai-smart-web` du Blueprint `render.yaml` ne sont pas ceux de cette production. Ne pas deduire du Blueprint la branche deployee, les commandes de demarrage, les ports ni le reglage de deploiement automatique des services reels.

### Variables des deux services actuels

| Service reel | Ajout ou conservation |
| --- | --- |
| `ai_smart_credit_suite-1` (frontend) | Ajouter `INTERSERVICE_SECRET` ; conserver `BACKEND_URL=https://ai-smart-credit-suite.onrender.com`. |
| `ai_smart_credit_suite` (backend) | Ajouter exactement le meme `INTERSERVICE_SECRET` ; conserver `ENABLE_LOCAL_API_DOCS` absent ou a `0`. |

Aucune variable `INTERNAL_API_HOSTPORT` n'est necessaire. La laisser absente : si elle est renseignee et que le frontend utilise `python -m utils.launch web`, ce lanceur remplace `BACKEND_URL` par son adresse HTTP interne. `API_BASE` n'a pas besoin d'etre ajoutee ou modifiee : `BACKEND_URL` est prioritaire. Sur le frontend, une valeur `INTERSERVICE_SECRET` dans les secrets Streamlit est prioritaire sur l'environnement : si elle existe deja, elle doit correspondre au meme secret.

L'ajout du secret sur les deux services suffit pour l'authentification : la commande backend verifiee et le loopback utilisent le meme port fourni par Render via `PORT`. Le loopback `/tools/predict_credit` ? `/api/predict` vise `http://127.0.0.1:<port>` avec priorite `PORT`, puis `API_PORT`, puis `8000`, et ajoute le secret lu dans l'environnement backend. Il ignore `BACKEND_URL`, `API_BASE` et `INTERNAL_API_HOSTPORT`. Le port public HTTPS 443 n'est pas ce port local.

La Start Command du backend reel a ete verifiee : `uvicorn server:app --host 0.0.0.0 --port $PORT`. Render fournit le port via `PORT` : uvicorn ecoute donc sur ce port et le loopback utilise prioritairement la meme variable `PORT`. Aucune modification de la Start Command n'est necessaire. Ne pas imposer `PORT=8000` sur Render ; `API_PORT=8000` reste seulement pertinent pour notre configuration locale si necessaire.

### Ordre de deploiement

Cette procedure est a executer ulterieurement par l'operateur ; aucune publication ni operation Render n'est effectuee par cette verification.

1. Identifier la branche/revision des deux services reels ; conserver la Start Command backend verifiee ci-dessus. Preparer la revision validee sans declencher le deploiement du backend en premier ; verifier les reglages de deploiement automatique reels avant toute publication. Ne pas appliquer le Blueprint.
2. Generer un secret aleatoire long dans un terminal prive, puis configurer `INTERSERVICE_SECRET` sur le frontend `ai_smart_credit_suite-1`, en conservant son `BACKEND_URL` HTTPS actuel. Deployer d'abord la revision frontend validee avec ce secret. Le backend encore ancien doit accepter les requetes portant ce header supplementaire ; verifier prediction et explication contre ce backend avant de poursuivre.
3. Attendre que le nouveau frontend soit actif. Recharger les sessions Streamlit deja ouvertes pour qu'elles utilisent le nouveau code envoyant le secret. Ne pas activer le backend protege tant que des clients utilises restent sur l'ancien code.
4. Configurer le meme `INTERSERVICE_SECRET` sur le backend `ai_smart_credit_suite`, puis deployer la revision backend validee. Si la sauvegarde des variables declenche un deploiement, ne le faire qu'apres les etapes frontend. Garder `ENABLE_LOCAL_API_DOCS` absent ou a `0` et conserver le port fourni par Render via `PORT`, utilise aussi par le loopback. Aucun changement de modele, de stockage ou d'URL publique n'est requis.
5. Verifier sur le backend public : `/health` (200 sans header), `/api/predict_ping` (401 sans header, 200 avec le secret depuis un environnement prive), puis prediction et explication depuis le frontend et `/tools/predict_credit` authentifie. Verifier `/docs`, `/redoc` et `/openapi.json` (404). Ne pas afficher le secret ni les headers dans les traces.
6. En cas d'echec du frontend a l'etape 2, ne pas deployer le backend protege. Apres activation de la protection, tout retour arriere doit conserver un frontend qui envoie le secret et un backend qui l'exige. Coordonner toute rotation du secret sur les deux services.

Cet ordre evite la fenetre d'incompatibilite ou un ancien frontend sans header appelle un nouveau backend protege. Il ne garantit pas a lui seul l'absence de coupure d'infrastructure ou la migration de toutes les sessions Streamlit existantes ; celles-ci doivent etre rechargees avant la bascule backend.

### Portee de start.ps1 et verification du code

`start.ps1` est un lanceur local Windows ; sa modification n'est pas requise pour cette correction de deploiement Render. Le changement actuel accompagne le masquage des secrets dans la sortie de `utils.runtime_config` : il evite de transmettre la chaine `[REDACTED]` aux processus enfants, qui chargent eux-memes `.env`. C'est un ajustement du lancement local, pas une dependance de l'authentification en production. Il ne configure pas le port du loopback a lui seul : en local aussi, `PORT`/`API_PORT` doit correspondre au port uvicorn.

Relecture statique des changements le 19 octobre 2026 : `backend_url()` accepte HTTPS et donne priorite a `BACKEND_URL` ; les clients prediction/explication ajoutent le header a cette destination et refusent les redirections ; le contexte de `/tools/predict_credit` force uniquement son appel interne sur le loopback et conserve l'authentification. Aucun ajout de code ne requiert un Private Service. Le support optionnel de `INTERNAL_API_HOSTPORT` dans `utils.launch` est preexistant. Cette verification ne constitue pas un test live des services Render ; aucun test n'a ete relance et aucune variable Render n'a ete modifiee.

## Validation locale du 19 octobre 2026

- Tests cibles : 84 reussis, 3 avertissements.
- Integration : RUN_RUNTIME_SMOKE=1 avec tests/test_runtime_integration.py, tests/test_integrations.py, tests/test_deployment.py et tests/test_lot2b1_boundaries.py : 60 reussis, 8 avertissements. Inclut de vrais serveurs FastAPI/Streamlit et le loopback HTTP.
- Suite complete : 566 reussis, 11 ignores, 69 avertissements (444,77 secondes). Les tests live sont opt-in et ont ete executes dans la campagne precedente.
- Syntaxe PowerShell de start.ps1 validee par le parseur natif.
- git diff --check : aucune erreur.

Les avertissements de depreciation existants ne sont pas corriges dans ce changement. Les tests ne modifient pas les variables Render. Le lancement d'une instance existante necessite un redemarrage apres configuration du secret.
