# AI Smart Credit Suite

Prototype de recherche en scoring de cr?dit : API FastAPI, interface Streamlit,
rapports HTML/PDF, tableaux de bord et outils documentaires/LLM optionnels.

Aucune conformit? ? B?le III/IV, BCBS 239, IFRS 9, au RGPD ou ? l?AI Act
n?est d?montr?e ou certifi?e par ce d?p?t. Le prototype n?est pas qualifi? pour
une utilisation bancaire en production. Les PD sont des sorties du mod?le livr? ;
leur calibration, leur horizon et leurs performances op?rationnelles restent ? valider.
L?administration ne poss?de pas de contr?le d?acc?s : un h?bergement public avec
secrets n?cessite un travail de s?curit? suppl?mentaire.

## Installation locale

Python 3.12 est l?environnement test?. Depuis la racine du projet sous Windows :

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m utils.launch api
# Dans un second terminal :
.venv/Scripts/python.exe -m utils.launch web
```

Le d?marrage et le scoring utilisent les trois artefacts suivis dans `model/`.
Ne charger que des pickles de provenance fiable. Aucun entra?nement n?est requis.
Sans `.env`, cl?s, historique, corpus ou index, les services d?marrent ; les tableaux
de bord sont vides et les fonctionnalit?s documentaires/fournisseurs peuvent ?tre
indisponibles. Le CSV priv? n?est jamais n?cessaire pour ex?cuter les tests ordinaires.

Les options sont d?crites dans `.env.example`. Un `.env` priv? est facultatif pour
le d?marrage et requis uniquement selon le fournisseur choisi. Ne pas publier ses
cl?s. Aucun fournisseur r?el ni recherche Web n?est valid? par les tests simul?s.
Le parcours RAG documentaire exclut le Web ; sa g?n?ration est une action distincte.

## Tests et livraison candidate

```powershell
.venv/Scripts/python.exe -B -m unittest discover -s tests -v
.venv/Scripts/python.exe -B audit/phase16_verify.py
```

Le manifeste `audit/phase16_candidates.txt` ?num?re les fichiers propos?s, y compris
les douze nouveaux modules utils et les fixtures synth?tiques. Il ne constitue pas
un staging. Le second contr?le copie uniquement ces fichiers dans un dossier temporaire,
retire les variables applicatives/secr?tes h?rit?es, puis v?rifie tests et d?marrage.
Les tests runtime, corpus r?el et fournisseur restent opt-in ; leur omission est signal?e.
Les d?pendances d?j? install?es sont utilis?es : installation neuve, Linux, construction
Docker et d?ploiement cloud restent ? v?rifier. Les d?pendances transitives et l?image
de base ne sont pas int?gralement verrouill?es.

## Corpus et index facultatifs

Voir [la proc?dure de provisionnement](docs/reproducibility.md). Les sept PDF locaux
restent exclus tant que leurs droits de redistribution ne sont pas ?tablis. La licence
MIT du code ne vaut pas autorisation de redistribution de ces documents, des donn?es
ou des mod?les tiers. Aucun index ni historique priv? n?est livr? avec les candidats.

Auteur : Idriss Beman Kara ? projet de Master, INSSEDS.
