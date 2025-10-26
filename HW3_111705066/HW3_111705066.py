import json
import torch
import pandas as pd
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score
from torch import nn

# Load data
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

train_data = load_json('train.json')
valid_data = load_json('val.json')
test_data = load_json('test.json')

# Preprocess data
def preprocess(data, is_train=True):
    rows = []
    for item in data:
        u = item['u']  
        s = item['s']  
        r = item['r']  

        # 交替合併 u 和 s，保持原本的對話順序
        context = []
        for i, statement in enumerate(s):
            context.append(u if i % 2 == 0 else statement)  # 偶數位置放 u，奇數位置放 s

        context = " [SEP] ".join(context)  # 合併成單一字串，使用 [SEP] 分隔
        input_text = f"{context} [SEP] {r}"

        label = item['r.label'] if is_train else None
        rows.append({
            'input_text': input_text,
            'label': label
        })
    return pd.DataFrame(rows)

train_df = preprocess(train_data)
valid_df = preprocess(valid_data)
test_df = preprocess(test_data, is_train=False)

# Tokenizer and Dataset
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

def tokenize_data(df, is_train=True):
    encodings = tokenizer(
        list(df['input_text']),
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )
    labels = torch.tensor(df['label'].tolist()) if is_train else None
    return encodings, labels

train_encodings, train_labels = tokenize_data(train_df)
valid_encodings, valid_labels = tokenize_data(valid_df)
test_encodings, _ = tokenize_data(test_df, is_train=False)

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = self.labels[idx]
        return item

train_dataset = CustomDataset(train_encodings, train_labels)
valid_dataset = CustomDataset(valid_encodings, valid_labels)
test_dataset = CustomDataset(test_encodings)

# Load model
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
#model = RobertaForSequenceClassification.from_pretrained('textattack/roberta-base-imdb', num_labels=2)
# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=7,
    weight_decay=0.01,
    logging_dir='./logs',
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = torch.tensor(logits)
    preds = torch.argmax(logits, axis=-1)
    accuracy = (preds == labels).float().mean().item()
    return {"accuracy": accuracy}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

# Train model
trainer.train()

# Evaluate and predict
trainer.evaluate()
predictions = trainer.predict(test_dataset)

# Save predictions
test_df['response_quality'] = torch.argmax(torch.tensor(predictions.predictions), axis=-1).numpy()

# Save the result to CSV with 'index' and 'request_quality' columns
output = pd.DataFrame({
    'index': test_df.index,
    'response_quality': test_df['response_quality']
})

output.to_csv('robert_usr.csv', index=False)
