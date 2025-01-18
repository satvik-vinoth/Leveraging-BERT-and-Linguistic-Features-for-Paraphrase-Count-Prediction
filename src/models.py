import torch
import random
import numpy as np
from transformers import BertForSequenceClassification
from torch.utils.data import Dataset

# Set seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Custom Dataset for Regression with additional features
class ParaphraseRegressionDataset(Dataset):
    def __init__(self, data, tokenizer, feature_columns):
        self.data = data
        self.tokenizer = tokenizer
        self.feature_columns = feature_columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data.iloc[idx]['Sentence']
        label = self.data.iloc[idx]['Total Paraphrases']
        additional_features = self.data.iloc[idx][self.feature_columns].values.astype(float)

        encoding = self.tokenizer(sentence, truncation=True, padding='max_length', max_length=128, return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['features'] = torch.tensor(additional_features, dtype=torch.float)
        item['labels'] = torch.tensor(label, dtype=torch.float)
        return item

# Custom BERT Model with additional layers for linguistic features
class BertRegressionWithFeatures(torch.nn.Module):
    def __init__(self, model_name, num_features, dropout_rate=0.1):
        super(BertRegressionWithFeatures, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)
        self.dropout_bert = torch.nn.Dropout(dropout_rate)
        self.feature_fc = torch.nn.Linear(num_features, 128)
        self.dropout_features = torch.nn.Dropout(dropout_rate)
        self.custom_fc = torch.nn.Linear(896, 1024)  # 768 from BERT + 128 from features
        self.hidden_fc = torch.nn.Linear(1024, 1024)
        self.output_fc = torch.nn.Linear(1024, 1)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, features=None, labels=None):
        outputs = self.bert.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout_bert(pooled_output)

        feature_output = torch.relu(self.feature_fc(features))
        feature_output = self.dropout_features(feature_output)

        combined_output = torch.cat((pooled_output, feature_output), dim=1)

        custom_output = torch.relu(self.custom_fc(combined_output))
        hidden_output = torch.relu(self.hidden_fc(custom_output))

        logits = self.output_fc(hidden_output)

        loss = None
        if labels is not None:
            loss = torch.nn.functional.mse_loss(logits.view(-1), labels.view(-1))

        return {"loss": loss, "logits": logits}
