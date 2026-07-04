# Dossier projet — The North Face e-commerce (NLP)

**Bloc 3 — Analyse prédictive de données structurées (apprentissage non supervisé)**
Certification Jedha CDSD — RNCP35288
Auteur : Aymeric Lahonde — promotion Jedha CDSD 2026

---

## 1. Contexte et problème métier

The North Face, marque américaine d'équipement outdoor, veut **augmenter ses ventes sur son site e-commerce**. Le département marketing a identifié deux leviers exploitant les **descriptions textuelles** du catalogue :

1. **Recommander des produits similaires** sur chaque fiche produit (« Vous pourriez aussi aimer… ») pour augmenter le panier moyen ;
2. **Challenger la taxonomie existante** du catalogue en découvrant automatiquement des thèmes plus pertinents pour la navigation.

**Problème data science.** Aucune variable cible : on ne prédit pas une valeur, on **structure un corpus de texte** non étiqueté. C'est de l'**apprentissage non supervisé appliqué au NLP** : clustering de documents + topic modeling.

---

## 2. Données

- **Source** : [eCommerce Item Data — cclark (Kaggle)](https://www.kaggle.com/datasets/cclark/product-item-data).
- **Volume** : **500 produits**, 2 colonnes : `id` (identifiant) et `description` (texte marketing).
- **Contenu** : descriptions de vêtements et équipement outdoor (sous-vêtements techniques, coton bio, sacs, chaussures…).
- Fichier fourni dans `data/sample-data.csv`. Le notebook le charge automatiquement depuis `data/`.

---

## 3. Préprocessing du texte

1. **Nettoyage regex** (`clean_text`) : suppression des balises HTML, des caractères non alphabétiques (chiffres, ponctuation), passage en minuscules, normalisation des espaces.
2. **Lemmatisation spaCy** (`en_core_web_sm`, pipeline sans NER ni parser pour la performance) : chaque token est ramené à son lemme, en retirant les *stop words*, la ponctuation et les tokens de moins de 3 caractères. Cela regroupe les variantes morphologiques (« boxers » → « boxer », « running » → « run ») et réduit le vocabulaire.
3. **Vectorisation TF-IDF** (`TfidfVectorizer`) : unigrammes + bigrammes, `min_df=2`, `max_df=0.9`, stop words anglais. Résultat : matrice creuse **500 × 8 109**.

---

## 4. Partie 1 — Clustering des produits similaires (DBSCAN)

- **Algorithme** : `DBSCAN`, métrique **cosine** (adaptée au texte — proximité par l'angle des vecteurs, pas par la norme), `algorithm='brute'` (requis pour la métrique cosinus).
- **Choix des hyperparamètres** : balayage de `eps ∈ [0,45 ; 0,80]` et `min_samples ∈ {2,3,4}`, avec un score maison favorisant ~12 clusters, peu de bruit et pas de cluster géant. Retenu : **`eps = 0,75`, `min_samples = 3`**.
- **Résultat** : **14 clusters** + un cluster de bruit (-1) contenant **31 produits (~6,2 %)**.
- **Interprétation** : chaque cluster est décrit par ses **top-termes** (moyenne TF-IDF). Exemples : cluster « coton bio / recyclé », cluster « sous-vêtements Capilene / contrôle des odeurs », cluster « sacs / bretelles / imperméable ». Le plus gros cluster (237 produits) regroupe le textile coton générique.
- **Visualisation** : **wordclouds** des 6 principaux clusters (générés et visibles dans le notebook).

---

## 5. Partie 2 — Système de recommandation

- On réutilise les `cluster_id` de la Partie 1 et une **matrice de similarité cosinus** entre tous les produits.
- `find_similar_items(item_id, top_k=5)` : renvoie les 5 produits **du même cluster** les plus proches en similarité cosinus. Si le produit est un **outlier** (cluster -1), bascule sur une similarité cosinus **globale**.
- `show_recommendations(item_id)` : affiche la description du produit et ses 5 suggestions.
- **Interface utilisateur** : démo non bloquante sur plusieurs `id`, + **boucle interactive `input()`** fournie (en commentaire, à décommenter dans une session Jupyter).
- **Exemple** : produit #1 « Active classic boxers » → suggestions #19 (Cap 1 boxer briefs), #494 (Active boxer briefs), #495 (Active briefs), #365 (Organic cotton boxers), #18 (Cap 1 bottoms). Cohérence thématique validée.

---

## 6. Partie 3 — Topic modeling (LSA)

- **Algorithme** : `TruncatedSVD` (LSA) sur la matrice TF-IDF, **`n_components = 12`**, `random_state=42`.
- Chaque produit reçoit un vecteur de 12 scores de topics ; on extrait le **topic principal** (composante de plus forte valeur absolue).
- **Interprétation** : chaque topic est décrit par ses mots de plus fort poids. Exemples : topic « organic / cotton / recycle / common thread », topic « spandex / nylon / coverage / tencel », topic « strap / shoulder / waterproof breathable ». Ces thèmes suggèrent une **réorganisation possible du catalogue**.
- **Visualisation** : **wordclouds** des 6 principaux topics (générés et visibles dans le notebook).

---

## 7. Résultats et recommandations métier

1. **Reco produit** — brancher `find_similar_items` sur les fiches produit pour un bloc « Vous pourriez aussi aimer ».
2. **Navigation** — s'appuyer sur les topics LSA (coton bio, techniques/spandex, sacs/portage…) pour repenser les rayons plutôt que les catégories historiques.
3. **Priorité** — le gros cluster textile coton (237 produits) mériterait une sous-segmentation plus fine.

---

## 8. Limites et perspectives

- Corpus réduit (500 produits) : quelques clusters très fins (3-4 produits).
- ~6 % de produits en bruit DBSCAN (traités par le fallback cosinus global côté reco).
- **Perspectives** : remplacer TF-IDF par des **embeddings de phrases** (Sentence-BERT) pour capter la sémantique ; objectiver le choix de `eps` / `n_components` (silhouette cosinus, cohérence des topics) ; exposer la reco derrière une **API** ou une petite **UI**.

---

## 9. Reproductibilité

- Données : `data/sample-data.csv` (dataset Kaggle `cclark/product-item-data`).
- Notebook : `notebooks/01_north_face_solution.ipynb` (exécuté de bout en bout, wordclouds inclus).
- Installation : `pip install -r requirements.txt` puis `python -m spacy download en_core_web_sm`.
- Déterminisme : `TruncatedSVD(random_state=42)` ; DBSCAN et TF-IDF sont déterministes.

**Stack** : Python, pandas, numpy, spaCy (`en_core_web_sm`), scikit-learn (TfidfVectorizer, DBSCAN, TruncatedSVD, cosine_similarity), wordcloud, matplotlib.
