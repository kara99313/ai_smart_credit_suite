# Lot 4D — Observabilité technique

La page Administration protégée lit exclusivement les télémétries 4B et 4C.
Le contrôle Admin précède toute lecture. Aucun appel externe n'est déclenché.
Les archives métier, agent_usage.csv et l'audit administratif ne sont pas lus.

`utils/observability_telemetry.py` centralise les lectures et agrégations. Les bases
sont ouvertes en URI `mode=ro`, avec `query_only=ON`, timeout 50 ms, transaction
cohérente par source et fermeture garantie. Aucun fichier absent n'est créé ;
aucune migration, écriture ou purge n'est effectuée. Une source absente, vide,
verrouillée, corrompue ou incompatible ne bloque pas l'autre. Les messages sont
génériques et ne contiennent pas les exceptions. Une base sans tables est
indisponible ; une base valide sans événements dans la fenêtre est vide.

La fenêtre par défaut est de 30 jours ; 24 heures et 7 jours sont disponibles.
Les bornes inclusives sont appliquées en SQL sur les débuts UTC, avec exclusion
des événements futurs. Les dates des graphiques sont UTC. Chaque table est filtrée
sur son propre début : les tentatives/outils d'un run plus ancien peuvent donc
figurer dans la fenêtre. Les producteurs assurent une rétention de 30 jours lors
des écritures ; la page ne purge jamais.

Les tentatives API sont les seuls `api_predict`. Le taux de succès est
success / toutes ces tentatives (success + validation_error + internal_error).
Moyenne et médiane en ms couvrent leurs durées terminales finies et positives ou
nulles. Le nombre de latences valides est fourni dans les agrégats. Les warmups
sont comptés séparément, jamais dans ces KPI. Le filtre Tous/business/synthetic
s'applique aux appels et aux préchauffages. business est déclaratif et ne prouve
pas un client réel ; synthetic désigne les appels déclarés synthétiques.

Agent : exécutions = lignes de runs dans la fenêtre. Taux = success /
(success + error) ; running en est exclu. Les latences couvrent uniquement ces
runs terminaux. Les tentatives sont comptées dans agent_attempts, sans sommer
les compteurs des runs. Une tentative au statut retry indique une décision de
retry ; seul un numéro attempt > 1 prouve le démarrage effectif d'un retry.
provider/model absents sont présentés comme « Non renseigné ».

Outils démarrés = lignes agent_tools. Terminés = success + error + no_result +
unknown. interrupted et running sont séparés ; les durées agrégées couvrent les
terminés uniquement. running signifie qu'aucune terminaison n'est observée,
éventuellement après un arrêt brutal, sans preuve d'activité actuelle.
Les agrégats par outil et statut ne montrent aucun argument ou résultat.
rag_retrieve/search_web/build_report sont uniquement des appels techniques.
**build_report ne constitue pas un téléchargement confirmé.**

Les écritures best effort et la couverture des parcours instrumentés ne permettent
pas de déduire la disponibilité globale des services, les appels LLM internes,
la qualité métier, les coûts ou les tokens. Les statuts des outils reposent sur
les heuristiques des producteurs. Les latences absentes/invalides sont exclues,
une absence de dénominateur donne « Indisponible ».

Aucun request_id ni événement individuel n'est retourné. Les colonnes lues sont
explicites et limitées aux métadonnées techniques. Les noms d'outils et fournisseurs
sont limités à des listes autorisées ; les identifiants de modèles sont validés.
Aucun prompt, réponse, historique, donnée client, PD, décision, secret, contenu
RAG/Web, HTML ou exception brute n'est exposé.
