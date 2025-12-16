# BERT Classification and Bias Rewriting

This project provides implementations for detecting media bias using BERT-based classifiers and includes tools for bias rewriting using reinforcement learning.

## Project Structure

```
.
├── BERTs/                    # BERT classifier implementations
│   ├── train_bert.py         # Train standard BERT classifier
│   └── train_convbert.py     # Train ConvBERT classifier
│
├── baselines/                # Baseline model implementations
│   ├── baseline_logistic.py  # Logistic regression baseline
│   └── baseline_random.py    # Random classification baseline
│
├── dataset_utils/            # Data processing utilities
│   └── create_multiclass_bias.py  # Create multiclass bias dataset
│
├── eval_scripts/             # Evaluation scripts
│   ├── eval_bert.py          # Evaluate BERT classifier
│   ├── eval_convbert.py      # Evaluate ConvBERT classifier
│   └── eval_bias_rewriter.py # Evaluate bias rewriting system
│
└── rl_bias/                  # Reinforcement learning bias rewriting
    ├── rl_bias_gpt2.py       # RL-based bias rewriting with GPT-2
    └── rl_bias_qwen.py       # RL-based bias rewriting with Qwen

```

## Data Preparation

### Step 1: Obtain the Dataset

The dataset used in this project comes from the MBIB (Media Bias In the News) repository:

- **Source**: https://github.com/Media-Bias-Group/MBIB
- **Instructions**: Clone or download the repository following the instructions provided in the MBIB repository

### Step 2: Run Corpus Generation

After obtaining the MBIB dataset, run the corpus generation script to create the various task-specific CSV files. Follow the instructions in the MBIB repository for corpus generation.

### Step 3: Generate Multiclass Bias Labels

Once the corpus has been generated and the task CSVs are created, run the multiclass bias label generation:

```bash
python dataset_utils/create_multiclass_bias.py
```

This script processes the task CSVs and creates multiclass bias labels for training and evaluation.

## Usage

### Training BERT Classifiers

**Standard BERT:**
```bash
python BERTs/train_bert.py
```

**ConvBERT:**
```bash
python BERTs/train_convbert.py
```

### Baseline Models

**Logistic Regression Baseline:**
```bash
python baselines/baseline_logistic.py
```

**Random Classification Baseline:**
```bash
python baselines/baseline_random.py
```

### Evaluation

**Evaluate BERT:**
```bash
python eval_scripts/eval_bert.py
```

**Evaluate ConvBERT:**
```bash
python eval_scripts/eval_convbert.py
```

**Evaluate Bias Rewriting:**
```bash
python eval_scripts/eval_bias_rewriter.py
```

### Reinforcement Learning Bias Rewriting

**GPT-2 Based Rewriting:**
```bash
python rl_bias/rl_bias_gpt2.py
```

**Qwen Based Rewriting:**
```bash
python rl_bias/rl_bias_qwen.py
```

## Important Notes

**Data Paths and Checkpoints**: All scripts require data paths and model checkpoint paths to be configured. Before running any script, please edit the relevant configuration variables (typically at the top of each script) to point to:
- The correct location of your MBIB dataset
- The correct paths for training data CSVs
- The correct paths for model checkpoints and output directories

Ensure that all paths are absolute or correctly relative to the script execution directory.

## Requirements

This project requires Python 3.7+. Install all dependencies with:

```bash
pip install -r requirements.txt
```

This will install all required packages including PyTorch, Transformers (Hugging Face), and other dependencies.

## License

Please refer to the MBIB repository for information about dataset licensing.
