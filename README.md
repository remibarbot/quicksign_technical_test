- [Test Technique : Machine Learning Engineer](#test-technique--machine-learning-engineer)
  - [Disclaimer !](#disclaimer-)
  - [Introduction](#introduction)
  - [Bonnes pratiques](#bonnes-pratiques)
  - [Objectif](#objectif)
  - [Sujet](#sujet)
  - [Exigences techniques](#exigences-techniques)
  - [Déroulement](#déroulement)
  - [Livrables](#livrables)
  - [Durée estimée](#durée-estimée)
  - [Exécution](#exécution)
    - [Ubuntu](#ubuntu)
  - [Analyse du rendu](#analyse-du-rendu)

# Test Technique : Machine Learning Engineer

## Disclaimer !

Le test est conçu pour évaluer votre approche et votre capacité à résoudre des problèmes de manière pratique. Nous comprenons que le temps et les ressources sont limités, et il est normal de ne pas pouvoir compléter chaque partie du test. Ce qui nous intéresse avant tout, c'est votre raisonnement, la façon dont vous abordez les problèmes et les choix que vous faites pour résoudre les différentes étapes.

Nous ne recherchons pas de solution parfaite, ni à l'état de l'art, mais plutôt une solution fonctionnelle, claire et bien documentée. Il est important d’expliquer vos choix dans un fichier dédié ou directement dans les commentaires du code. Nous préférons un code simple, mais qui fonctionne, plutôt qu'un code plus complexe, mais difficile à exécuter.

N’hésitez pas à simplifier certaines parties du test si nécessaire (par exemple, gérer uniquement des fichiers spécifiques ou vous concentrer sur une approche plus rapide), mais veillez à bien documenter vos décisions.

Bonne chance, et surtout, amusez-vous avec ce test !

## Introduction

Chez QuickSign, nous traitons des dizaines de milliers de documents tous les mois. L'équipe Document Recognition (DocReco) a pour rôle de créer des algorithmes permettant de classifier et de lire ces documents. Ces différents algorithmes sont ensuite appelés via une API.

L'ensemble de nos librairies et nos services sont développés avec l'outil [poetry](https://python-poetry.org/). Cela nous permet de les versionner, de développer plusieurs services / librairies sur la même machine en utilisant des environnements virtuels et de gérer les dépendances de nos services / librairies en fixant les versions des packages dont ils dépendent.

## Bonnes pratiques

L'utilisation de poetry est primordiale et nous permet de tester le code reçu en quelques lignes. Un environnement à jour est indispensable afin de nous permettre de tester le code.

Certaines conventions sur le code sont établies au sein de notre équipe. Nous appliquons `ruff` pour formater le code et respecter les normes définies. Nous attendons la même rigueur pour votre rendu. Nous utilisons également `mypy` pour la gestion du typage.

Le script `lint_module.sh` permet d'automatiser l'application de ces outils.

Nous accordons par ailleurs une forte importance à la réalisation de tests unitaires, et au coverage de ceux-ci. Pour les lancer et avoir un rapport de coverage, utilisez le script `run_tests.sh`

Nous vous conseillons, avant même de poursuivre, de nous assurer que vous avez sur votre machine :

- [poetry](https://python-poetry.org/)
- [docker](https://docs.docker.com/engine/install/)

et que vous êtes en capacité de lancer les commandes dont vous aurez besoin tout au long du test

```
poetry install
./lint_module.sh
./run_tests.sh
docker-compose up --build
```

---

## Objectif

Ce test technique vise à évaluer votre capacité à :

- Trouver un jeu de données pertinent (en utiliser un ou plusieurs publics, ou en générer).
- Implémenter une pipeline complète :
  - Préparation des données,
  - Entraînement de modèles,
  - Évaluation
  - Déploiement via une API.
- Assurer la qualité de votre code via des tests et de la documentation.
- Conteneuriser votre solution à l'aide de Docker.

---

## Sujet

À partir de la base de code fournie, vous devez construire un système utilisant [FastAPI](https://fastapi.tiangolo.com/) et permettant de :

1. **Entraîner des algorithmes de classification supervisée** pour distinguer deux types de documents : **"Manuscrit"** ou **"Dactylographié"**.
2. **Tester ces algorithmes** sur un jeu de test.
3. **Exposer des routes FastAPI** pour utiliser votre système.
4. **S'assurer que le code est fonctionnel à l'aide de tests.**

---

## Exigences techniques

- Utiliser FastAPI pour exposer votre service.
- Utiliser Docker pour conteneuriser votre application.
- Vous pouvez utiliser n'importe quelle librairie de machine learning (scikit-learn, PyTorch, TensorFlow, etc.), mais votre `pyproject.toml` doit être à jour et fonctionnel sur un système linux (debian / ubuntu).
- Comparer les performances d'au moins 2 algorithmes.
- Implémenter des tests.

---

## Déroulement

1. **Récupération de la donnée** : Utilisez un ou plusieurs jeux de données publics ou générez-les si nécessaire.
2. **Entraînement & Test** : Implémentez un système qui permet d'entraîner un ou plusieurs algorithmes sur ces données, d'évaluer les performances, et de retourner les métriques d'évaluation.
3. **Service d'Inférence** : Implémentez une route permettant d'envoyer une requête avec des données à un algorithme et de récupérer un résultat.
4. **Conteneurisation** : Modifiez le Dockerfile si nécessaire pour inclure vos dépendances. Vérifiez que le service fonctionne avec `docker compose up --build`.
5. **Testing** : Vérifier que tout est fonctionnel à l'aide de tests.

---

## Livrables

Un dossier contenant :

- Votre code source dans le dossier `app/`.
- Le nécéssaire pour générer un environnement virtuel faisant tourner l'application à l'aide de poetry.
- Vos fichiers de configuration Docker.
- Des tests unitaires vérifiant les fonctionnalités de votre API.

---

## Durée estimée

Le test ne devrait pas prendre plus d'une demi-journée.

## Exécution

### Ubuntu

Ce setup a été pensé pour une machine GNU/Linux (Ubuntu par exemple). Si vous n'avez pas accès directement à cet OS, nous vous conseillons d'utiliser une VM ou Docker.

Les fichiers `Dockerfile` et `docker-compose.yml` permettent de lancer le web service sur le port `9000` via la commande :

```
docker compose up --build
```

## Analyse du rendu

Nous accorderons beaucoup d'importance aux respects des consignes et à la méthodologie scientifique, assez peu, voire aucune importance aux résultats statistiques.

Tout rendu doit impérativement permettre l'exécution des quatre actions suivantes sans erreur.

```
poetry install
./lint_module.sh
./run_tests.sh
docker-compose up --build
```
