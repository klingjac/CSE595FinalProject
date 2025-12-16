import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

base_path = os.path.dirname(os.path.abspath(__file__))
files = [
    ("multiclass-bias.csv", "Balanced"),
    ("multiclass-bias-unbalanced.csv", "Unbalanced")
]

for fname, label in files:
    fpath = os.path.join(base_path, fname)
    if not os.path.exists(fpath):
        print(f"{fname} not found, skipping.")
        continue
    df = pd.read_csv(fpath)
    y_true = df['bias_label'].values
    classes, counts = np.unique(y_true, return_counts=True)
    class_probs = counts / counts.sum()
    class_to_prob = dict(zip(classes, class_probs))
    print(f"\n{label} set class distribution:")
    for c, p in class_to_prob.items():
        print(f"  {c}: {p:.3f}")
    # Simulate random predictions weighted by class distribution
    y_pred = np.random.choice(classes, size=len(y_true), p=class_probs)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')
    print(f"{label} set results:")
    print(f"  Macro-F1: {macro_f1:.4f}")
    print(f"  Micro-F1: {micro_f1:.4f}")
