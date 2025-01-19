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
```
## 🚀Installation

### Prerequisites
 - Python 3.7 or higher.
 - GPU (optional but recommended for faster model training).

### Steps to run
 - Clone this repository:
   ```plaintext
   git clone https://github.com/satvik-vinoth/Leveraging-BERT-and-Linguistic-Features-for-Paraphrase-Count-Prediction.git
   cd Leveraging-BERT-and-Linguistic-Features-for-Paraphrase-Count-Prediction
   ```
 - Install the required dependencies:
   ```plaintext
   pip install -r requirements.txt
   ```
 - Running the Training Script
   ```plaintext
   python src/train_model.py
   ```
## Outputs
The project generates the following outputs:
 - Training and Validation Loss Curves: Visualized to track model performance over epochs.
 - Feature Importance: Visualized as a bar chart to show the impact of linguistic features.
 - Predicted vs. Actual Paraphrase Counts: Scatter plot to compare predictions with ground truth.
 - Metrics: Final evaluation metrics (R², MSE, Precision, Recall, and F1-score).

## Contact
For questions, feel free to reach out via GitHub.
  
