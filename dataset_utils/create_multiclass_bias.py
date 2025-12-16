import os
import pandas as pd

"""
This file combines the various bias CSV files from the MBIB dataset into a unified dataset.

IMPORTANT: Before running this script, copy the following CSV files from the MBIB dataset
to this directory (dataset_utils/):
    - cognitive-bias.csv
    - political-bias.csv
    - fake-news.csv
    - gender-bias.csv
    - racial-bias.csv
    - hate-speech.csv

This script will create:
    - multiclass-bias-unbalanced.csv (all bias types combined)
    - multiclass-bias.csv (balanced version for training)
"""

# ===== CONFIGURATION: Update input paths if your CSV files are elsewhere =====
bias_files = {
    'political': ['cognitive-bias.csv', 'political-bias.csv'],
    'fake_news': ['fake-news.csv'],
    'gender': ['gender-bias.csv'],
    'racial': ['racial-bias.csv'],
    'hate_speech': ['hate-speech.csv'],
    'linguistic': ['linguistic-bias.csv']
}

# EDIT: Set this to the directory containing the MBIB CSV files
base_path = os.path.dirname(os.path.abspath(__file__))


# Load and label each bias class, and collect unbiased examples
bias_frames = []
unbiased_frames = []
for label, files in bias_files.items():
    for fname in files:
        fpath = os.path.join(base_path, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            if 'label' in df.columns:
                # Biased rows (label==1)
                biased = df[df['label'] == 1].copy()
                biased['bias_label'] = label
                bias_frames.append(biased)
                # Unbiased rows (label==0)
                unbiased = df[df['label'] == 0].copy()
                unbiased['bias_label'] = 'no_bias'
                unbiased_frames.append(unbiased)
            else:
                print(f"Warning: {fname} missing 'label' column.")
        else:
            print(f"Warning: {fname} not found.")

# Combine all bias and unbiased data (unbalanced)
all_bias_df = pd.concat(bias_frames + unbiased_frames, ignore_index=True)

# Save unbalanced version
all_bias_df.to_csv(os.path.join(base_path, 'multiclass-bias-unbalanced2.csv'), index=False)
print('Unbalanced multiclass bias dataset created as multiclass-bias-unbalanced.csv')

# Create balanced version
min_count = all_bias_df['bias_label'].value_counts().min()
balanced_df = (
    all_bias_df.groupby('bias_label', group_keys=False)
    .apply(lambda x: x.sample(n=min_count, random_state=42))
    .reset_index(drop=True)
)

# Save balanced version
balanced_df.to_csv(os.path.join(base_path, 'multiclass-bias2.csv'), index=False)
print('Balanced multiclass bias dataset created as multiclass-bias.csv')
