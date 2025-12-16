import os
import sys
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import wandb

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import WANDB_API_KEY

# Parameters
DATA_FILE = "multiclass-bias.csv"  # Use balanced set
MAX_LEN = 128
BATCH_SIZE = 8
EPOCHS = 100
MODEL_NAME = "YituTech/conv-bert-base"
SAVE_MODEL_NAME = "convbert_bias_classifier"

# Initialize wandb
if WANDB_API_KEY:
    wandb.login(key=WANDB_API_KEY, relogin=True)
    wandb.init(project="mbib-multiclass", name="convbert-classifier", reinit=True)
    print("Weights & Biases logging enabled")
else:
    print("Warning: WANDB_API_KEY not set in config.py, logging disabled")

# Load data
base_path = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base_path, DATA_FILE))
print(f"Loaded {len(df)} samples from {DATA_FILE}")

# Encode labels
le = LabelEncoder()
df['label_id'] = le.fit_transform(df['bias_label'])
num_labels = len(le.classes_)
print(f"Number of classes: {num_labels}")
print(f"Classes: {le.classes_}")

# Simple train/test split
train_df = df.sample(frac=0.8, random_state=42)
test_df = df.drop(train_df.index)
print(f"Train set: {len(train_df)}, Test set: {len(test_df)}")

# Load tokenizer and model
print(f"\nLoading tokenizer and model from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Dataset class
class BiasDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.encodings = tokenizer(
            df['text'].astype(str).tolist(),
            truncation=True,
            padding=True,
            max_length=MAX_LEN
        )
        self.labels = df['label_id'].tolist()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = BiasDataset(train_df, tokenizer)
test_dataset = BiasDataset(test_df, tokenizer)

# Model
print(f"Loading model {MODEL_NAME}...")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)

# Training arguments with wandb integration
training_args = TrainingArguments(
    output_dir="./convbert_results",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    disable_tqdm=False,
    report_to=["wandb"] if WANDB_API_KEY else [],
    run_name="convbert-classifier",
    seed=42,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=1)
    report = classification_report(
        labels,
        preds,
        target_names=le.classes_,
        output_dict=True,
        zero_division=0
    )
    return {
        "accuracy": report["accuracy"],
        "macro_f1": report["macro avg"]["f1-score"],
        "micro_f1": report["weighted avg"]["f1-score"],
        "precision": report["weighted avg"]["precision"],
        "recall": report["weighted avg"]["recall"],
    }

print("\nStarting training...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()

print("\nEvaluating on test set...")
results = trainer.evaluate()
print("\nEvaluation Results:")
for k, v in results.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# Log final metrics to wandb
if WANDB_API_KEY:
    wandb.log({"final_metrics": results})
    wandb.finish()

# Save model and label encoder
save_dir = os.path.join(base_path, SAVE_MODEL_NAME)
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)
import joblib
joblib.dump(le, os.path.join(save_dir, "label_encoder.joblib"))
print(f"\nModel, tokenizer, and label encoder saved to {save_dir}")
