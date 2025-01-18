# Leveraging BERT and Linguistic Features for Paraphrase Count Prediction

## Overview
This project predicts the number of paraphrases a sentence can generate using a custom BERT-based regression model enhanced with linguistic features. The project leverages advanced natural language processing (NLP) techniques for paraphrase generation, feature extraction, and model training.

## Features
- **Generate paraphrases** using the **PEGASUS** model.
- **Extract linguistic features** such as:
  - Part-of-Speech (POS) tags.
  - Dependency structures.
  - Readability scores.
- **Train a BERT-based regression model** to predict paraphrase counts for a given sentence.
- Evaluate model performance using:
  - **R²** (Coefficient of Determination)
  - **MSE** (Mean Squared Error)
  - **Precision**, **Recall**, and **F1-score**.

## Folder Structure
```plaintext
├── data/
│   ├── raw/
│   │   └── Enriched_Dataset.xlsx          # Dataset used for training
├── notebooks/
│   ├── dataset_creation_small.ipynb       # Generates a small paraphrase dataset
│   ├── enriching_dataset_linguistic_features.ipynb  # Adds linguistic features
│   ├── model_training_evaluation.ipynb    # Model training and evaluation
├── src/
│   ├── models.py                          # Model architecture and dataset classes
│   ├── train_model.py                     # Main script for training and evaluation
├── requirements.txt                       # Python dependencies for the project
├── LICENSE                                # License for the project
├── README.md                              # Project description and instructions
.gitignore                                 # Files and folders to exclude from Git

