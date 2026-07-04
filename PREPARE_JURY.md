# Préparation soutenance — Bloc 3 : The North Face e-commerce (NLP)

Certification Jedha CDSD (RNCP35288) — Bloc 3 : Analyse prédictive de données structurées, apprentissage non supervisé (NLP).

---

## 1. Pitch en 60 secondes

The North Face veut booster ses ventes en ligne à partir des **descriptions produits** de son catalogue. À partir de 500 fiches produits, je construis un pipeline **NLP non supervisé** : je nettoie et **lemmatise** le texte avec spaCy, je le vectorise en **TF-IDF**, puis je **regroupe les produits similaires avec DBSCAN** (distance cosinus). Ces clusters alimentent un **système de recommandation** content-based (`find_similar_items` → 5 produits proches). Enfin, un **topic modeling LSA (TruncatedSVD)** extrait des thèmes latents pour challenger les catégories existantes du catalogue. Les clusters et topics sont visualisés par **wordclouds**.

---

## 2. Chiffres clés à connaître par cœur

| Élément | Valeur |
|---|---|
| Produits (SKU) | **500** |
| Colonnes | `id`, `description` |
| Vocabulaire TF-IDF (uni+bigrammes, min_df=2) | **8 109 features** |
| Algo de clustering retenu | **DBSCAN**, métrique **cosine** |
| Paramètres DBSCAN | `eps = 0,75`, `min_samples = 3` |
| Nombre de clusters | **14** (hors bruit) |
| Outliers (cluster -1) | **31 produits (~6,2 %)** |
| Plus gros cluster | cluster 1 = 237 produits (coton/textile générique) |
| Topic modeling | **LSA / TruncatedSVD**, `n_components = 12` |
| Recommander | similarité **cosinus** intra-cluster, top-5 |
| Modèle spaCy | `en_core_web_sm` (lemmatisation) |

---

## 3. Questions / Réponses probables du jury

### Q — Pourquoi du NLP non supervisé et pas un modèle supervisé ?
Il n'y a **aucun label** : le catalogue ne fournit pas de « bonne catégorie » cible. On veut **découvrir** des regroupements et des thèmes à partir du seul texte des descriptions. C'est la définition de l'apprentissage non supervisé appliqué au texte : clustering + topic modeling.

### Q — Décris ton préprocessing.
Deux étapes : (1) **nettoyage regex** — suppression des balises HTML, de la ponctuation et des chiffres, passage en minuscules, normalisation des espaces ; (2) **lemmatisation spaCy** (`en_core_web_sm`) — je ramène chaque mot à sa forme canonique (« running » → « run », « boxers » → « boxer »), en retirant les *stop words*, la ponctuation et les tokens trop courts. La lemmatisation **réduit la dimension** du vocabulaire et regroupe les variantes d'un même mot, ce qui améliore la qualité des vecteurs TF-IDF.

### Q — Pourquoi TF-IDF ? Comment ça marche ?
TF-IDF pondère chaque mot d'un document par sa **fréquence dans le document (TF)** multipliée par l'**inverse de sa fréquence dans le corpus (IDF)**. Les mots fréquents partout (peu discriminants) sont écrasés, les mots spécifiques à quelques produits ressortent. J'utilise des **unigrammes + bigrammes** (`ngram_range=(1,2)`) pour capter des expressions comme « organic cotton » ou « waterproof breathable », avec `min_df=2` (mot présent dans au moins 2 produits) et `max_df=0.9` pour retirer les mots ubiquitaires.

### Q — Pourquoi DBSCAN et pas KMeans ?
- **KMeans** impose de fixer **k** à l'avance et affecte **tous** les points, y compris des produits atypiques qui polluent les centroïdes.
- **DBSCAN** regroupe par **densité** : il trouve **automatiquement** le nombre de clusters et isole les produits atypiques comme **bruit** (label -1). C'est adapté à un catalogue hétérogène.
- L'énoncé recommande explicitement DBSCAN avec **distance cosinus** (et non euclidienne) : en NLP, deux descriptions sont proches par l'**angle** entre leurs vecteurs, pas par leur longueur. Deux fiches courtes et longues sur le même thème doivent être considérées similaires → cosinus.

### Q — Comment as-tu choisi `eps` et `min_samples` ?
Par **balayage** : je teste plusieurs `eps` (0,45 → 0,80) et `min_samples` (2, 3, 4) et je retiens la combinaison qui donne un nombre de clusters proche de la cible (10-20), **peu de bruit**, et pas de cluster géant qui absorbe tout. Le meilleur compromis est `eps = 0,75`, `min_samples = 3` → **14 clusters, ~6 % d'outliers**. C'est dans la fourchette demandée par l'énoncé.

### Q — Que représente la distance cosinus dans DBSCAN ?
La **similarité cosinus** = cosinus de l'angle entre deux vecteurs TF-IDF (1 = identiques, 0 = orthogonaux/aucun mot commun). La **distance cosinus = 1 − similarité**. `eps = 0,75` signifie qu'on relie deux produits si leur distance cosinus est ≤ 0,75, soit une similarité ≥ 0,25. J'utilise `algorithm='brute'` car la métrique cosinus n'est pas supportée par les arbres (ball_tree/kd_tree).

### Q — Comment fonctionne ton système de recommandation ?
Pour un `item_id` donné : je récupère son **cluster**, puis je classe les autres produits **du même cluster** par **similarité cosinus décroissante** et je renvoie le **top-5**. Rester dans le cluster garantit une cohérence thématique. **Cas particulier** : si le produit est un outlier (cluster -1), je bascule sur une similarité cosinus **globale** (sur tout le catalogue) pour éviter de recommander d'autres outliers sans rapport. Exemple : pour le produit #1 (« Active classic boxers »), le système propose des boxers/briefs Capilene et des sous-vêtements coton — recommandations parfaitement cohérentes.

### Q — La boucle interactive `input()` demandée par l'énoncé ?
Elle est **fournie** dans le notebook : une version **démo non bloquante** exécute `show_recommendations` sur quelques `id` (1, 42, 137) pour que le jury voie le résultat directement, et la **boucle `while True` avec `input()`** est fournie juste en dessous (en commentaire pour ne pas bloquer l'exécution automatique du notebook — il suffit de la décommenter dans une session Jupyter vivante).

### Q — Qu'est-ce que le topic modeling LSA / TruncatedSVD ?
La **LSA (Latent Semantic Analysis)** applique une **décomposition en valeurs singulières tronquée (TruncatedSVD)** à la matrice TF-IDF. Elle projette les 8 109 dimensions sur **12 « topics » latents** : chaque topic est une combinaison de mots co-occurrents (ex. « organic / cotton / recycle », ou « strap / shoulder / waterproof »). Contrairement au clustering, un document peut appartenir à **plusieurs topics** ; j'extrais donc le **topic principal** (composante de plus forte valeur absolue) pour ranger chaque produit. C'est une piste pour **repenser la taxonomie** du site.

### Q — DBSCAN vs LSA : quelle différence ?
DBSCAN **partitionne** les produits en groupes disjoints (chaque produit dans un seul cluster). LSA **décompose** le vocabulaire en thèmes et permet une appartenance **multiple / graduée**. Les deux sont complémentaires : le clustering sert la reco, le topic modeling sert la réorganisation du catalogue.

### Q — Comment évalues-tu la qualité sans labels ?
En non supervisé sans vérité terrain, l'évaluation est surtout **qualitative** : je lis les **top-termes** de chaque cluster/topic (via la moyenne TF-IDF) et je vérifie leur cohérence sémantique, et je visualise des **wordclouds** par cluster et par topic. Les clusters obtenus sont interprétables (sous-vêtements, coton bio, sacs/bretelles, etc.), ce qui valide qualitativement l'approche. Piste honnête d'amélioration : ajouter un score de silhouette (sur distance cosinus) pour objectiver.

### Q — Limites et perspectives ?
- Petit corpus (500 produits) → certains clusters sont fins (3-4 produits).
- DBSCAN laisse ~6 % d'outliers non recommandés via cluster (gérés par le fallback cosinus global).
- Améliorations : embeddings de phrases (Sentence-BERT) au lieu de TF-IDF pour capter la sémantique, exposer la reco derrière une petite API/UI, et objectiver le choix de `eps`/`n_components`.

### Q — Reproductibilité ?
`data/sample-data.csv` (500 lignes, dataset Kaggle `cclark/product-item-data`) + `notebooks/01_north_face_solution.ipynb`. `TruncatedSVD` a un `random_state=42`. `pip install -r requirements.txt` + `python -m spacy download en_core_web_sm`, puis exécution complète du notebook rejoue tout à l'identique (wordclouds inclus).
