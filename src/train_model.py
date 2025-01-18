import pandas as pd
import torch
from transformers import BertTokenizer, Trainer, TrainingArguments, TrainerCallback
from sklearn.metrics import r2_score, mean_squared_error, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from models import set_seed, BertRegressionWithFeatures, ParaphraseRegressionDataset

# Set seed
set_seed(42)

# Load the dataset
file_path = 'data/raw/Enriched_Dataset.xlsx'
data = pd.read_excel(file_path)

# Handle missing values and ensure data integrity
data = data.dropna(subset=['Sentence', 'Total Paraphrases'])

# Duplicate data to balance the dataset
duplicated_data = pd.concat([data, data], ignore_index=True)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Select the feature columns (excluding Sentence and Total Paraphrases)
feature_columns = list(data.columns)
feature_columns.remove('Sentence')
feature_columns.remove('Total Paraphrases')

# Ensure feature columns contain only numerical values
for col in feature_columns:
    duplicated_data[col] = pd.to_numeric(duplicated_data[col], errors='coerce')

# Drop rows with missing or invalid numerical values in feature columns
duplicated_data = duplicated_data.dropna(subset=feature_columns)

# Split data into training and validation sets
train_data, val_data = train_test_split(duplicated_data, test_size=0.1, random_state=42)
train_dataset = ParaphraseRegressionDataset(train_data, tokenizer, feature_columns)
val_dataset = ParaphraseRegressionDataset(val_data, tokenizer, feature_columns)

# Load modified BERT model
num_features = len(feature_columns)
model = BertRegressionWithFeatures('bert-base-uncased', num_features)

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize lists to store metrics
training_losses = []
validation_losses = []
r2_scores = []
mse_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    report_to="none",
)

# Define compute_metrics function
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.squeeze()

    r2 = r2_score(labels, preds)
    mse = mean_squared_error(labels, preds)
    precision = precision_score((labels >= 0.5), (preds >= 0.5), zero_division=0)
    recall = recall_score((labels >= 0.5), (preds >= 0.5), zero_division=0)
    f1 = f1_score((labels >= 0.5), (preds >= 0.5), zero_division=0)

    return {"r2": r2, "mse": mse, "precision": precision, "recall": recall, "f1": f1}

# Custom callback
class CustomCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        training_losses.append(state.log_history[-1]['loss'])

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        validation_losses.append(metrics['eval_loss'])
        r2_scores.append(metrics['eval_r2'])
        mse_scores.append(metrics['eval_mse'])
        precision_scores.append(metrics['eval_precision'])
        recall_scores.append(metrics['eval_recall'])
        f1_scores.append(metrics['eval_f1'])

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[CustomCallback()],
)

# Train the model
trainer.train()

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(training_losses) + 1), training_losses, label='Training Loss')
plt.plot(range(1, len(validation_losses) + 1), validation_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot: R² Score vs Epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(r2_scores) + 1), r2_scores, label='R² Score')
plt.title('R² Score over Epochs')
plt.xlabel('Epochs')
plt.ylabel('R² Score')
plt.legend()
plt.show()

# Plot: Mean Squared Error (MSE) over Epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(mse_scores) + 1), mse_scores, label='MSE')
plt.title('Mean Squared Error over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()

# Plot: Precision, Recall, and F1-Score over Epochs
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(precision_scores) + 1), precision_scores, label='Precision', color='orange')
plt.plot(range(1, len(recall_scores) + 1), recall_scores, label='Recall', color='green')
plt.plot(range(1, len(f1_scores) + 1), f1_scores, label='F1 Score', color='red')
plt.title('Precision, Recall, and F1-Score over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()
plt.show()

# Predict on the validation set
predictions = trainer.predict(val_dataset).predictions.squeeze()
true_values = val_dataset.data['Total Paraphrases'].values

# Plot: Predicted vs Actual Paraphrase Counts
plt.figure(figsize=(10, 6))
plt.scatter(true_values, predictions, alpha=0.5)
plt.title('Predicted vs Actual Paraphrase Counts')
plt.xlabel('Actual Paraphrase Counts')
plt.ylabel('Predicted Paraphrase Counts')
plt.show()

# Feature importance
def plot_feature_importance(model, feature_columns):
    weights = model.feature_fc.weight.cpu().detach().numpy()
    feature_importance = np.mean(np.abs(weights), axis=0)
    feature_importance_series = pd.Series(feature_importance, index=feature_columns).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    feature_importance_series.plot(kind='bar')
    plt.title('Feature Importance')
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.show()

plot_feature_importance(model, feature_columns)