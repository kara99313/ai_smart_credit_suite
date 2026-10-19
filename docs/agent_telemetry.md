# Lot 4C — Télémétrie Agent

La base locale `logs/agent_telemetry.sqlite3` est distincte de la télémétrie de
prédiction : aucun changement du schéma, du contrat ou des archives de prédiction.
Cette isolation limite le couplage et la contention entre API de prédiction et
callbacks Agent. SQLite et la bibliothèque standard suffisent, sans dépendance
supplémentaire. Le futur lot 4D pourra lire les trois tables techniques.

- `agent_runs` : clé `request_id`, version 1, début UTC explicite, durée monotone
  en millisecondes, statut, nombre de tentatives, fournisseur/modèle connus,
  compteurs des outils démarrés, terminés et en erreur.
- `agent_attempts` : clé composée `(request_id, attempt)`, début UTC, durée et
  statut `running`, `retry`, `success` ou `error`.
- `agent_tools` : clé composée `(request_id, tool_id)`, tentative de démarrage,
  début UTC, nom autorisé, durée et statut. Le numéro local `tool_id` est attribué
  une seule fois par identifiant de callback LangChain ; ce dernier reste en mémoire.

Le `request_id` existant des diagnostics est créé une seule fois, avant la boucle
de retry, et transmis à toutes les lignes. Les démarrages et terminaisons mettent
à jour les mêmes clés par upsert. Un callback terminal répété ne change pas le
compteur. Les étapes retournées par le modèle ne créent aucune ligne outil.
La règle existante interdit toujours tout retry dès le premier démarrage d'outil.
Le nombre de tentatives décrit les appels réels à `invoke`, pas les appels LLM
internes éventuels du fournisseur.

Un outil terminé peut avoir le statut `success`, `error`, `no_result` ou `unknown`.
Une exception donne `error`. Les sorties sont examinées seulement en mémoire :
statut explicite ToolMessage en erreur prioritaire, enveloppe HTML pour
`build_report`, puis sources `[D1]` / `[W1]` propres aux outils RAG/Web,
avec espaces et sauts de ligne initiaux. Sinon, seuls les prefixes initiaux
`ERROR:` / `Erreur:` et les contrats connus de chaque outil donnent `error` :
JSON invalide pour predict_credit, echec d'appel /api/predict, RAG ou recherche
Web indisponible (avec leurs pictogrammes eventuels). Aucun mot du corps
metier ne suffit. Les sorties vides et les prefixes connus d'absence de
resultat donnent `no_result`, les autres textes `success`, le non-texte `unknown`.
La premiere terminaison d'une tentative fixe son statut et sa duree.
Cette classification reste une heuristique ; elle ne valide pas le contenu métier
ni l'exactitude des documents. Un Agent qui termine normalement peut avoir des
outils en erreur : le statut de son exécution et les compteurs outils restent distincts.

Un outil sans callback terminal lors de la fin de l'Agent devient `interrupted` ;
il n'est compté ni terminé ni en erreur confirmée. Sa durée mesure le temps
observé jusqu'à l'interruption, sans prétendre connaître la fin réelle du travail.
Un arrêt brutal du processus peut laisser des lignes `running` : aucune terminaison
n'est inventée. Les écritures sont best effort, avec attente SQLite limitée à 50 ms
par connexion. Les erreurs de stockage sont absorbées sans journaliser leur texte.
Les lignes de plus de 30 jours sont supprimées lors des écritures, comme au lot 4B.

Seules les métadonnées fixes sont transmises au stockage. Les noms d'outils
applicatifs sont autorisés explicitement ; les autres deviennent `unknown`.
Le fournisseur est reconnu depuis le type réel du chat model et le modèle depuis
ses champs de configuration, jamais depuis la réponse. Un modèle non identifiable
reste NULL. Aucun prompt, historique, argument, observation, donnée client,
contenu RAG/Web, HTML, secret ou message d'exception n'est transmis à SQLite.

`build_report` mesure uniquement la génération du rapport. Aucun événement de
réception navigateur ni téléchargement confirmé n'est créé. La présentation et
les boutons de téléchargement restent gérés par le code existant.
