import json
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def load_data(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

train_data = load_data('train.json')
val_data = load_data('val.json')
test_data = load_data('test.json')

# define dataset
class TweetDataset(Dataset):
    def __init__(self, data, tokenizer, max_length, label_mapping):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label_mapping = label_mapping

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tweet = item['tweet']
        
        labels = [0] * len(self.label_mapping)  
        for label in item.get('labels', {}).keys():
            if label in self.label_mapping:
                labels[self.label_mapping[label]] = 1  
        
        encoding = self.tokenizer(
            tweet,
            padding='max_length',
            max_length=self.max_length,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(labels, dtype=torch.float) 
        }

# label_mapping
label_mapping = {
    'ineffective': 0, 'unnecessary': 1, 'pharma': 2, 'rushed': 3,
    'side-effect': 4, 'mandatory': 5, 'country': 6, 'ingredients': 7,
    'political': 8, 'none': 9, 'conspiracy': 10, 'religious': 11
}

#  BERT_tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased', 
    num_labels=len(label_mapping),
    problem_type="multi_label_classification"
)

train_dataset = TweetDataset(train_data, tokenizer, max_length=128, label_mapping=label_mapping)
val_dataset = TweetDataset(val_data, tokenizer, max_length=128, label_mapping=label_mapping)

# Parameter
training_args = TrainingArguments(
    output_dir="./results_bce",
    evaluation_strategy='epoch',
    learning_rate=4e-5,
    per_device_train_batch_size=8,  
    per_device_eval_batch_size=8,
    num_train_epochs=8, 
    weight_decay=0.01, # prevent overfitting
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

# evaluate metric
def compute_metrics(pred):
    labels = pred.label_ids
    preds = (pred.predictions > 0.5).astype(int)  
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

# testing
test_dataset = TweetDataset(test_data, tokenizer, max_length=128, label_mapping=label_mapping)
test_results = trainer.predict(test_dataset)
print(test_results.metrics)

predictions = (test_results.predictions > 0.5).astype(int)

df = pd.DataFrame(predictions, columns=label_mapping.keys())
df.insert(0, 'index', range(len(df))) 

# save result as .csv
output_path = "test_bce_12.csv"
df.to_csv(output_path, index=False)
