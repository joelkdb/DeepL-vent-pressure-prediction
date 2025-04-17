# 🫁 DeepL-vent-pressure-prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-Google%20Brain%20Challenge-20BEFF.svg)](https://www.kaggle.com/c/ventilator-pressure-prediction)

Un projet d'apprentissage profond pour la prédiction de pression ventilatoire en soins intensifs, basé sur le jeu de données "Google Brain - Ventilator Pressure Prediction" https://research.google/blog/machine-learning-for-mechanical-ventilation-control/

## 📋 À propos du projet

Ce projet vise à développer des modèles de deep learning pour prédire avec précision la pression dans les voies respiratoires de patients sous ventilation mécanique. La prédiction précise de la pression ventilatoire est cruciale pour optimiser les soins aux patients en état critique et réduire les lésions pulmonaires potentielles liées à la ventilation.

### Objectifs
- Analyser en profondeur les données de séries temporelles de pression ventilatoire
- Implémenter et comparer différentes architectures de deep learning (MLP, GRU, LSTM, Transformer)
- Optimiser les modèles pour minimiser l'erreur de prédiction
- Visualiser et interpréter les résultats des modèles

## 🔢 Données

Nous utilisons le jeu de données "Google Brain - Ventilator Pressure Prediction" disponible sur Kaggle. Ce dataset contient des séries temporelles des paramètres de ventilateur et des pressions mesurées.

### Structure des données
- Paramètres d'entrée du ventilateur
- Mesures physiologiques des patients
- Pression mesurée dans les voies respiratoires (cible à prédire)

## 🧠 Modèles

Nous explorons plusieurs architectures de deep learning, notamment :

- **MLP (Multi-Layer Perceptron)** : Modèle de référence simple
- **GRU (Gated Recurrent Units)** : Pour capturer les dépendances temporelles
- **LSTM (Long Short-Term Memory)** : Pour les relations à long terme dans les séries temporelles
- **Transformer** : Architecture basée sur l'attention pour une modélisation avancée

## 👥 Équipe
Ce projet est réalisé par des étudiants de l'UTBM:

- [Membre 1]
- Membre 2]
- [Membre 3]
- [Membre 4]
- [Membre 5]

## 🚀 Installation et démarrage

```bash
# Cloner le repository
git clone https://github.com/votre-utilisateur/deepl-vent-pressure-prediction.git
cd deepl-vent-pressure-prediction-ai





