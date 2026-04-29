# Bloc 3 — The North Face e-commerce (NLP)

> Projet de la **Certification Jedha — Concepteur Développeur en Science des Données (CDSD)**
> [RNCP35288 — France Compétences](https://www.francecompetences.fr/recherche/rncp/35288/)
> **Bloc 3** : Analyse prédictive de données structurées — NLP / Recommender system

## Objectif du projet

The North Face veut **recommander des produits similaires** à ses clients e-commerce. À partir des descriptions textuelles du catalogue (titre + description marketing), on doit :

1. **Représenter** chaque produit par un vecteur (TF-IDF ou embeddings)
2. **Clusteriser** les produits par similarité textuelle (KMeans)
3. **Recommander** les N produits les plus proches d'un produit donné (similarité cosinus)

C'est un projet de **NLP non supervisé** + **système de recommandation simple**.

## Données

Le dataset (catalogue produits) est téléchargé directement depuis l'URL Jedha dans le notebook (S3 public Jedha) — pas besoin de gérer le fichier en local.

## Livrable

- Notebook de modélisation : TF-IDF + KMeans + recommandations cosinus
- Visualisations : wordclouds par cluster, top mots-clés, exemples de recommandations
- Notebook "oral support" pour la soutenance

## Structure du projet

```
Bloc_3_North_Face_ecommerce/
├── notebooks/
│   ├── 00_north_face_enonce.ipynb         (énoncé Jedha)
│   ├── 01_north_face_solution.ipynb       (modélisation TF-IDF + clustering — livrable)
│   └── 02_north_face_oral_support.ipynb   (support de soutenance)
├── requirements.txt
└── README.md
```

## Comment rejouer le projet

```bash
cd Bloc_3_North_Face_ecommerce
pip install -r requirements.txt
```

Puis ouvrir `notebooks/01_north_face_solution.ipynb` dans Jupyter ou VS Code.

## Auteur

[Aymeric Lahonde](https://github.com/aymericlahonde-dotcom) — promotion Jedha CDSD 2026.
