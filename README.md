# GPT — Un modèle GPT construit from scratch

> Assembled from the NeetCode ML course on [NeetCode.io](https://neetcode.io)
> Built by **Alexandre Ovu** on July 31, 2026

Chaque fichier de ce projet est du code que j'ai écrit et soumis en suivant le cours NeetCode Machine Learning.  
Les problèmes s'enchaînent progressivement, partant des bases de la descente de gradient jusqu'à un GPT opérationnel.

## Présentation

Mon projet est une implémentation complète d'un modèle GPT réalisée 
sans utiliser de bibliothèques haut niveau dédiées aux Transformers.

L'objectif est de comprendre et reconstruire les composants internes 
des Large Language Models (LLM), depuis les mathématiques fondamentales 
jusqu'à l'architecture Transformer utilisée par les modèles modernes.


## Compétences couvertes

### Fondations Machine Learning & Deep Learning

- Descente de gradient
- Fonctions de perte (loss functions)
- Fonctions d'activation
- Réseaux de neurones
- Rétropropagation (backpropagation)
- Perceptrons multicouches (MLP)
- Initialisation des poids
- Pytorch & Numpy basics
- Boucles d'entraînement
- Diagnostic et analyse d'entraînement


### Pipeline NLP (Natural Language Processing)

- Prétraitement du texte
- Construction de vocabulaire
- Tokenisation
- Préparation des datasets
- Chargement des données par batch
- Représentation numérique du langage


### Architecture Transformer

- Word embeddings
- Encodage positionnel
- Self-attention
- Multi-head attention
- Blocs Transformer
- Layer Normalization
- RMS Normalization
- KV Cache
- Grouped Query Attention


### Modèle GPT

- Architecture GPT autoregressive
- Entraînement d'un modèle de langage
- Prédiction du prochain token
- Génération de texte
- Échantillonnage (text sampling)


## Parcours d'apprentissage

Ce projet a été réalisé en suivant le parcours Machine Learning 
de NeetCode.

Le cours a fourni une progression pédagogique et les bases théoriques, 
mais l'implémentation, l'intégration des différentes architectures 
et l'organisation du projet sont mes propres réalisations.


## Structure du projet
```text
GPT/
│
├── foundations/
│   ├── gradient_descent.py
│   ├── activations.py
│   ├── softmax.py
│   ├── loss.py
│   ├── linear_regression.py
│   ├── linear_regression_training.py
│   ├── neuron.py
│   ├── backprop.py
│   ├── multi_layer_backprop.py
│   ├── mlp.py
│   ├── weight_init.py
│   ├── pytorch_basics.py
│   ├── digit_classifier.py
│   ├── sentiment.py
│   ├── training_loop.py
│   ├── training_diagnostics.py
│   └── dead_relu_detector.py
│
├── data/
│   ├── tokenizer.py
│   ├── vocab.py
│   ├── dataset.py
│   ├── loader.py
│   ├── nlp_preprocessing.py
│   └── tokenizer_utils.py
│
├── model/
│   ├── embeddings.py
│   ├── positional_encoding.py
│   ├── attention.py
│   ├── multi_head_attention.py
│   ├── transformer.py
│   ├── normalization.py
│   ├── batch_normalization.py
│   ├── rms_normalization.py
│   ├── kv_cache.py
│   ├── grouped_query_attention.py
│   └── gpt.py
│
├── train.py
├── generate.py
```

