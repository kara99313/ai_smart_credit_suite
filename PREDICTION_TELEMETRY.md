# Lot 4B : telemetrie technique des predictions

Stockage local : `logs/prediction_telemetry.sqlite3` (ignore par Git), table
`prediction_events`. SQLite est fourni par Python, sans dependance ajoutee.

Une ligne terminale par POST /api/predict, y compris JSON invalide, validation
Pydantic et erreur interne. Le middleware observe la reponse complete sans lire
le corps. Le client et les outils Agent ne produisent pas de second evenement.
Le prechauffage utilise le meme mecanisme avec call_type=warmup.

Colonnes :
- schema_version : 1 ;
- request_id : UUID4 genere localement, jamais repris des en-tetes ;
- timestamp_utc : debut de tentative, ISO 8601 avec +00:00 ;
- duration_ms : temps ecoule perf_counter, non negatif, hors ecriture SQLite ;
- status : success (2xx), validation_error (4xx), internal_error (autres/exception) ;
- error_category : NULL, request_validation ou prediction_failure ;
- model_id : SHA-256 des octets du modele charge, NULL si identite non conforme ;
- origin : business ou synthetic ;
- call_type : api_predict ou warmup.

Le diagnostic transmet X-Prediction-Origin: synthetic. Seule cette valeur exacte
est reconnue ; les autres valeurs ne sont jamais conservees. Ce marqueur est
declaratif et ne constitue pas une preuve authentifiee de provenance.

Aucun contenu client, resultat individuel, hash d'entree, URL, corps HTTP,
exception ou secret n'est accepte dans le schema. Aucune lecture de .env n'est
ajoutee. Les archives CSV et results.jsonl restent distinctes et inchangees.

Conservation logique glissante de 30 jours, purge lors de chaque ecriture reussie,
index sur timestamp_utc. Une installation inactive ne purge pas automatiquement.
La suppression SQL ne garantit pas l'effacement physique des pages SQLite.
Transactions et cle primaire UUID evitent les lignes partielles et doublons ;
connexions fermees apres chaque tentative. Attente de verrou limitee a 50 ms.
Echec de stockage : evenement perdu, aucune exception ni contenu sensible emis,
reponse metier preservee. Ce stockage local ne garantit pas la livraison.

Perimetre examine : prediction Streamlit et predict_credit_tool passent par le
client puis /api/predict. /tools/predict_credit ne recoit pas de telemetrie Agent.
/health, predict_ping et predict_echo ne calculent pas de prediction. /tools/explain
recalcule une probabilite pour verifier la provenance d'une explication et reste
hors du comptage des tentatives de prediction. Les helpers locaux predictor et
scoring_runtime ne sont pas appeles par ces chemins actifs et restent inchanges.
Les appels Python directs a api_predict ne traversent pas la frontiere HTTP.
Les echecs de transport avant reception serveur ne creent pas d'evenement serveur.
Aucune refonte de app/observability.py, Agent, RAG, Web ou logique scientifique.
