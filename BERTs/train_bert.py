import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

# ===== CONFIGURATION: Update these paths before running =====
# Path to the multiclass bias CSV file created by dataset_utils/create_multiclass_bias.py
DATA_FILE = "multiclass-bias.csv"  # EDIT: Update to your data file path
MODEL_SAVE_DIR = "bert_bias_classifier"  # EDIT: Where to save the trained model

# Model hyperparameters
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 100
MODEL_NAME = "bert-base-uncased"
# ===== END CONFIGURATION =====

# Load data
base_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(base_path, DATA_FILE)
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found at: {data_path}. Please update DATA_FILE path in configuration.")
df = pd.read_csv(data_path)

# Encode labels
le = LabelEncoder()
df['label_id'] = le.fit_transform(df['bias_label'])
num_labels = len(le.classes_)

# Simple train/test split
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)

# Dataset class
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
class BiasDataset(Dataset):
    def __init__(self, df):
        self.encodings = tokenizer(df['text'].astype(str).tolist(), truncation=True, padding=True, max_length=MAX_LEN)
        self.labels = df['label_id'].tolist()
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    def __len__(self):
        return len(self.labels)

train_dataset = BiasDataset(train_df)
test_dataset = BiasDataset(test_df)

# Model
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)



training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps=10,
    disable_tqdm=False
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    report = classification_report(labels, preds, target_names=le.classes_, output_dict=True, zero_division=0)
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "micro_f1": report["weighted avg"]["f1-score"]
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
results = trainer.evaluate()
print("\nEvaluation Results:")
for k, v in results.items():
    print(f"{k}: {v}")

# Save model and label encoder
save_dir = os.path.join(base_path, "bert_bias_classifier")
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
import joblib
joblib.dump(le, os.path.join(save_dir, "label_encoder.joblib"))
print(f"\nModel and label encoder saved to {save_dir}")
