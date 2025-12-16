import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
from tqdm import tqdm

# --- Config ---
DATA_FILE = "multiclass-bias.csv"
MODEL_DIR = "/home/jacob/CSE595/final_project/MBIB/convbert_results/checkpoint-236210"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 128
BATCH_SIZE = 32

print(f"Using device: {DEVICE}")

# Load data
base_path = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(base_path, DATA_FILE))
print(f"Loaded {len(df)} samples from {DATA_FILE}")

# Load model and tokenizer
model_path = "/home/jacob/CSE595/final_project/MBIB/convbert_results/checkpoint-236210"
print(f"Loading ConvBERT model from {model_path}")
model = AutoModelForSequenceClassification.from_pretrained(model_path).to(DEVICE)
# Load tokenizer from HuggingFace (ConvBERT base)
tokenizer = AutoTokenizer.from_pretrained("YituTech/conv-bert-base")
# Load label encoder from checkpoint directory if it exists, otherwise create default
label_encoder_path = os.path.join(model_path, "label_encoder.joblib")
if os.path.exists(label_encoder_path):
    label_encoder = joblib.load(label_encoder_path)
else:
    # Fallback: load from bert_bias_classifier directory
    label_encoder = joblib.load(os.path.join(base_path, "bert_bias_classifier", "label_encoder.joblib"))

model.eval()

# Tokenize data
print("Tokenizing data...")
input_ids = []
attention_masks = []
labels = []

for text, label in tqdm(zip(df['text'], df['bias_label']), total=len(df)):
    encoded = tokenizer(
        str(text),
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids.append(encoded['input_ids'].squeeze())
    attention_masks.append(encoded['attention_mask'].squeeze())
    labels.append(label_encoder.transform([label])[0])

input_ids = torch.stack(input_ids)
attention_masks = torch.stack(attention_masks)
labels_tensor = torch.tensor(labels, dtype=torch.long)

print(f"Tokenized {len(input_ids)} samples")

# Create DataLoader
dataset = TensorDataset(input_ids, attention_masks, labels_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# Evaluate
print("\nEvaluating model...")
all_preds = []
all_true = []
total_loss = 0.0

with torch.no_grad():
    for batch_idx, (batch_input_ids, batch_attention_masks, batch_labels) in enumerate(tqdm(dataloader, total=len(dataloader))):
        batch_input_ids = batch_input_ids.to(DEVICE)
        batch_attention_masks = batch_attention_masks.to(DEVICE)
        batch_labels = batch_labels.to(DEVICE)

        outputs = model(
            input_ids=batch_input_ids,
            attention_mask=batch_attention_masks,
            labels=batch_labels
        )

        loss = outputs.loss
        logits = outputs.logits
        total_loss += loss.item()

        predictions = torch.argmax(logits, dim=1)
        all_preds.extend(predictions.cpu().numpy())
        all_true.extend(batch_labels.cpu().numpy())

# Convert predictions back to labels
all_preds_labels = label_encoder.inverse_transform(all_preds)
all_true_labels = label_encoder.inverse_transform(all_true)

# Calculate metrics
avg_loss = total_loss / len(dataloader)
accuracy = accuracy_score(all_true_labels, all_preds_labels)
macro_f1 = f1_score(all_true_labels, all_preds_labels, average='macro', zero_division=0)
micro_f1 = f1_score(all_true_labels, all_preds_labels, average='micro', zero_division=0)

print("\n" + "="*60)
print("CONVBERT MULTICLASS BIAS CLASSIFIER EVALUATION")
print("="*60)
print(f"Average Loss: {avg_loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Micro F1: {micro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(all_true_labels, all_preds_labels, digits=4))

print("\nConfusion Matrix:")
print(confusion_matrix(all_true_labels, all_preds_labels))

# Per-class accuracy
print("\nPer-Class Metrics:")
class_labels = label_encoder.classes_
for label_idx, label_name in enumerate(class_labels):
    mask = np.array(all_true) == label_idx
    if mask.sum() > 0:
        class_accuracy = accuracy_score(
            np.array(all_true)[mask],
            np.array(all_preds)[mask]
        )
        count = mask.sum()
        print(f"  {label_name}: {class_accuracy:.4f} ({count} samples)")

print("\nClass Distribution in Dataset:")
print(df['bias_label'].value_counts())

print("\n" + "="*60)
print("Evaluation complete!")
