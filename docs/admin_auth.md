# Administration ? lot 2A

Un seul compte serveur, sans inscription et sans nouveau package. La page Administration reste dans la navigation comme point de connexion ; son contenu et ses op?rations sont contr?l?s s?par?ment. Les fonctionnalit?s ordinaires restent publiques.

## Configuration serveur

- `ADMIN_USERNAME` : identifiant non vide, imprimable, au plus 128 caract?res.
- `ADMIN_PASSWORD_HASH` : format `pbkdf2_sha256$iterations$salt_hex$digest_hex`. PBKDF2-HMAC-SHA256 sur le mot de passe encod? UTF-8, de 600000 ? 2000000 it?rations, sel al?atoire unique de 16 ? 64 octets encod? en hexad?cimal, r?sultat de 32 octets encod? en hexad?cimal. Pr?parer cette empreinte hors de l?application avec un outil de confiance ; ne jamais enregistrer le mot de passe en clair ni afficher ces ?l?ments dans l?application ou les journaux.
- `ADMIN_SESSION_TTL_SECONDS` : facultatif, entier de 60 ? 28800 secondes ; d?faut 1800 secondes.

Ces variables proviennent de l?environnement du processus serveur (dont le chargement serveur existant). Configuration absente ou invalide : refus. Aucun secret de production n?est fourni ou ?crit par cette impl?mentation.

## Sessions et autorisation

La connexion v?rifie les deux champs avec comparaison constante et cr?e un jeton al?atoire opaque. Un registre en m?moire serveur lie ce jeton ? la session Streamlit d?origine, ? l?identit? v?rifi?e, ? la configuration et ? une ?ch?ance absolue monotone. Un bool?en ou un acteur dans session_state ne donne aucun droit. Nouvelle session, red?marrage du processus, expiration, modification de configuration ou d?connexion : nouvelle authentification n?cessaire.

Cinq ?checs dans une fen?tre glissante de cinq minutes bloquent les nouvelles tentatives dans le processus, y compris celles d?autres navigateurs. Cette limitation volontairement simple peut bloquer temporairement le compte sous attaque ; elle ne remplace pas une limitation au niveau du proxy et n?est pas partag?e entre plusieurs processus.

Les contr?les pr?c?dent toute ?criture de configuration et tout test explicite LLM, y compris les tests dans Int?grations et Health Check. Les appels LLM ordinaires des assistants ne sont pas soumis ? ce contr?le. Les op?rations autoris?es sont audit?es avec l?acteur v?rifi?, leur nom et un indicateur de terminaison ; ni arguments ni r?sultats sensibles ne sont enregistr?s. Les ?v?nements historiques restent inchang?s. Une panne du journal est signal?e sans exposer l?exception.

## P?rim?tre restant

Pas de traitement des routes `/debug/*`, de la persistance Render, de l?authentification interservices, ni du durcissement complet Ollama/SSRF (lot 2B). Le registre et la limitation restent locaux au processus. Le journal existant n?est ni sign? ni garanti exhaustif.
