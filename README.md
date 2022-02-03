# Projet Projet BigData NoSQL
## Sujet

Réaliser une architecture d'analyse et de prédiction de données.
Les données seront des annonces d'appartements airbnb a Bordeaux.
Ces données sont constituées de texte, valeurs numériques qu'il faudra traiter et analyser.
L'objectif sera donc d'analyser ces données et de proposer un modèle de prédiction qui à partir de ces données estimera le prix d'une nuit de manière la plus fiable possible.


## Etapes du projet

- Etape 0
    - [x] Prise en main de HADOOP
    - [x] Insertion des données
- Etape 1
    - [x] Récupération des données (.csv) en local
- Etape 2
    - [x] Création d'une instance AWS (EC2 & S3)
    - [x] Envoi des données sur AWS via SSH
    - [x] Gestion du chiffrement
- Etape 3
    - [x] Choix & mise en place des modèles de machine learning
    - [x] Envoi du script sur EC2 pour qu'il soit exécuté
- Etape 4
    - [x] Récupération du .csv des informations individus
    - [x] Concatenation du résultat & predict.csv
- Etape 5
    - [x] Création de la base MongoDB
    - [x] Affichage grapique via GraphViz

## Technologies

Utilisation d'un HDFS Hadoop (machine virtuelle HortonWorks) afin de récupérer nos données
Utilisation des services AWS EC2 pour créer une machine virtuelle linux qui executera notre traitement et d'un bucket S3 qui stockera nos données entrées/sorties
Création d'une base MongoDB qui contiendra les résultats de notre traitement des analyses et prédictions de nos données


## Developers

- GAGNAIRE Thomas thomas.gagnaire@telecom-st-etienne.fr
- BIRON Gregoire gregoire.biron@telecom-st-etienne.fr
- FELICIANO Ruben ruben.feliciano@telecom-st-etienne.fr
- LUVISUTTO Eva eva.luvisutto@telecom-st-etienne.fr

