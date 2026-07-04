# Bloc 3 — The North Face e-commerce (NLP)

> Projet de la **Certification Jedha — Concepteur Développeur en Science des Données (CDSD)**
> [RNCP35288 — France Compétences](https://www.francecompetences.fr/recherche/rncp/35288/)
> **Bloc 3** : NLP non supervisé — clustering de produits, recommander system et topic modeling

## Objectif du projet

Le département marketing de The North Face veut booster les ventes en ligne à partir
des descriptions textuelles du catalogue produits. À partir de ces descriptions, on doit :

1. **Regrouper** les produits aux descriptions similaires (clustering)
2. **Recommander** des produits similaires sur chaque fiche produit ("Vous pourriez aussi aimer…")
3. **Extraire des topics latents** pour challenger les catégories existantes du catalogue

C'est un projet de **NLP non supervisé** + **système de recommandation content-based**.

## Approche technique

| Étape | Méthode |
|-------|---------|
| Préprocessing | nettoyage regex + **lemmatisation spaCy** (`en_core_web_sm`) |
| Vectorisation | **TF-IDF** (`TfidfVectorizer`, unigrammes + bigrammes) |
| Clustering | **DBSCAN** avec distance **cosine** (adaptée au texte) |
| Recommander | similarité **cosinus** à l'intérieur du cluster (`find_similar_items`) |
| Topic modeling | **LSA / TruncatedSVD** (12 composantes) + topic principal |
| Visualisation | **wordclouds** par cluster et par topic |

> Note : DBSCAN (et non KMeans) est utilisé car il gère les outliers et ne demande pas
> de fixer le nombre de clusters à l'avance. Sur ce catalogue on obtient ~14 clusters
> avec ~6 % d'outliers.

## Données

**Dataset :** [eCommerce Item Data — cclark (Kaggle)](https://www.kaggle.com/datasets/cclark/product-item-data)
(500 produits, 2 colonnes : `id` et `description`).

Le fichier `sample-data.csv` est fourni dans `data/`. Pour le récupérer à nouveau :

1. Télécharger `sample-data.csv` depuis la page Kaggle ci-dessus (compte Kaggle requis).
2. Le placer dans le dossier `data/` à la racine du projet.

Le notebook cherche automatiquement le fichier dans `data/sample-data.csv`.

## Structure du projet

```
Bloc_3_North_Face_ecommerce/
├── data/
│   └── sample-data.csv                    (catalogue produits, 500 lignes)
├── notebooks/
│   ├── 00_north_face_enonce.ipynb         (énoncé Jedha)
│   ├── 01_north_face_solution.ipynb       (livrable : préprocessing + DBSCAN + reco + LSA)
│   └── 02_north_face_oral_support.ipynb   (support de soutenance)
├── requirements.txt
└── README.md
```

## Comment rejouer le projet

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Puis ouvrir `notebooks/01_north_face_solution.ipynb` dans Jupyter ou VS Code et exécuter
toutes les cellules (le notebook est déjà exécuté avec les wordclouds visibles).

## Livrables

- Notebook de modélisation : préprocessing spaCy + TF-IDF + DBSCAN + recommandations cosinus + LSA
- Visualisations : wordclouds par cluster et par topic
- Fonction `find_similar_items(item_id)` + démo (et boucle interactive `input()` fournie en commentaire)
- Notebook "oral support" pour la soutenance

## Auteur

[Aymeric Lahonde](https://github.com/aymericlahonde-dotcom) — promotion Jedha CDSD 2026.
