
"""
Evaluation script comparing baseline GPT-2 vs PPO-trained GPT-2 for bias rewriting.

Computes metrics:
- Bias removal rate: % of outputs classified as 'no_bias' by BERT
- Semantic preservation: Average cosine similarity between input and output (via SBERT)
- Output diversity: Distinct outputs per input
"""

import os
import sys
import argparse
from typing import List, Tuple

import torch
import pandas as pd
import joblib
import numpy as np

from transformers import AutoTokenizer, BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import WANDB_API_KEY

# ----------------------------- Config -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_MAX_LEN = 128
GEN_MAX_TOKENS = 64

GEN_KWARGS = dict(
    max_new_tokens=GEN_MAX_TOKENS,
    do_sample=True,
    temperature=1.3,
    top_p=0.9,
    top_k=50,
    num_return_sequences=1,
)

# ----------------------------- Utilities ----------------------------

def load_bert_reward_model(bert_dir: str):
    """Load BERT classifier and label encoder."""
    bert_model = BertForSequenceClassification.from_pretrained(bert_dir).to(DEVICE)
    bert_tokenizer = BertTokenizer.from_pretrained(bert_dir)
    le = joblib.load(os.path.join(bert_dir, "label_encoder.joblib"))
    bert_model.eval()
    return bert_model, bert_tokenizer, le

def load_sbert_model(device=DEVICE):
    """Load Sentence-BERT for semantic similarity."""
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"⚠ Error loading SBERT: {e}")
        return None

@torch.no_grad()
def get_bias_label(text: str, bert_model, bert_tokenizer, le) -> str:
    """Get bias label from BERT classifier."""
    tokens = bert_tokenizer(
        text,
        add_special_tokens=True,
        max_length=BERT_MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(DEVICE)
    
    outputs = bert_model(**tokens)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    pred_id = int(logits.argmax(dim=1).item())
    return le.inverse_transform([pred_id])[0]

@torch.no_grad()
def get_bias_probability(text: str, bert_model, bert_tokenizer, le, target_class: str = 'no_bias') -> float:
    """Get probability of target bias class from BERT."""
    tokens = bert_tokenizer(
        text,
        add_special_tokens=True,
        max_length=BERT_MAX_LEN,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    ).to(DEVICE)
    
    outputs = bert_model(**tokens)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    target_idx = list(le.classes_).index(target_class)
    return float(probs[0, target_idx].item())

@torch.no_grad()
def compute_semantic_similarity(text1: str, text2: str, sbert_model) -> float:
    """Compute cosine similarity between two texts."""
    if sbert_model is None:
        return 0.0
    
    try:
        emb1 = sbert_model.encode(text1, convert_to_tensor=True)
        emb2 = sbert_model.encode(text2, convert_to_tensor=True)
        similarity = float(torch.nn.functional.cosine_similarity(
            emb1.unsqueeze(0),
            emb2.unsqueeze(0)
        ).item())
        return similarity
    except Exception as e:
        print(f"Warning: Error computing similarity: {e}")
        return 0.0

def generate_rewrite(text: str, bias_label: str, model, tokenizer) -> str:
    """Generate bias removal rewrite using model."""
    prompt = (
        f"Rewrite the sentence to remove {bias_label.replace('_', ' ')} "
        f"while preserving meaning and grammar. Output only the revised sentence.\n\n"
        f"Sentence: {text}\nRewritten:"
    )
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    ).to(DEVICE)
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            **GEN_KWARGS,
        )
    
    # Extract only the generated part (after prompt)
    prompt_len = inputs['input_ids'].shape[1]
    generated_ids = output_ids[0, prompt_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    return generated_text

# ----------------------------- Main Evaluation ----------------------------

def evaluate_model(model, tokenizer, test_df: pd.DataFrame, bert_model, bert_tokenizer, le, sbert_model, model_name: str = "Model"):
    """
    Evaluate a model on bias rewriting task.
    
    Returns dict with metrics:
    - bias_removal_rate: % of outputs classified as 'no_bias'
    - no_bias_probability: Average P(no_bias) from BERT
    - semantic_similarity: Average cosine similarity with inputs
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}")
    
    bias_removal_rate = 0.0
    avg_no_bias_prob = 0.0
    avg_semantic_sim = 0.0
    count = 0
    
    results = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"{model_name}"):
        orig_text = str(row['text'])
        orig_bias = str(row['bias_label'])
        
        # Generate rewrite
        try:
            rewritten = generate_rewrite(orig_text, orig_bias, model, tokenizer)
        except Exception as e:
            print(f"\nWarning: Generation failed for sample {idx}: {e}")
            continue
        
        # Get BERT predictions
        predicted_label = get_bias_label(rewritten, bert_model, bert_tokenizer, le)
        no_bias_prob = get_bias_probability(rewritten, bert_model, bert_tokenizer, le, 'no_bias')
        
        # Compute semantic similarity
        sem_sim = compute_semantic_similarity(orig_text, rewritten, sbert_model)
        
        # Update metrics
        if predicted_label == 'no_bias':
            bias_removal_rate += 1.0
        avg_no_bias_prob += no_bias_prob
        avg_semantic_sim += sem_sim
        count += 1
        
        results.append({
            'original': orig_text,
            'original_bias': orig_bias,
            'rewritten': rewritten,
            'predicted_label': predicted_label,
            'no_bias_probability': no_bias_prob,
            'semantic_similarity': sem_sim,
            'is_no_bias': predicted_label == 'no_bias'
        })
    
    # Compute averages
    if count > 0:
        bias_removal_rate /= count
        avg_no_bias_prob /= count
        avg_semantic_sim /= count
    
    results_df = pd.DataFrame(results)
    
    print(f"\nResults for {model_name}:")
    print(f"  Bias Removal Rate (% no_bias):    {bias_removal_rate*100:.2f}%")
    print(f"  Avg P(no_bias) from BERT:         {avg_no_bias_prob:.4f}")
    print(f"  Avg Semantic Similarity (SBERT):  {avg_semantic_sim:.4f}")
    print(f"  Samples evaluated:                {count}")
    
    return {
        'bias_removal_rate': bias_removal_rate,
        'avg_no_bias_prob': avg_no_bias_prob,
        'avg_semantic_similarity': avg_semantic_sim,
        'count': count,
        'results_df': results_df
    }

def print_example_results(all_results: dict, num_examples: int = 3):
    """Print example rewrites from each model."""
    print(f"\n\n{'='*100}")
    print(f"EXAMPLE REWRITES")
    print(f"{'='*100}")
    
    # Get a sample of results with improvements
    for model_name, results in all_results.items():
        results_df = results['results_df']
        
        print(f"\n{'─'*100}")
        print(f"{model_name.upper()}")
        print(f"{'─'*100}")
        
        # Show examples where model removes bias
        improved = results_df[results_df['is_no_bias'] == True]
        if len(improved) > 0:
            print(f"\n✓ Examples where {model_name} removes bias:\n")
            for i, (idx, row) in enumerate(improved.head(num_examples).iterrows()):
                print(f"Example {i+1}:")
                print(f"  Original ({row['original_bias']}): {row['original']}")
                print(f"  Rewritten: {row['rewritten']}")
                print(f"  BERT Prediction: {row['predicted_label']} (confidence: {row['no_bias_probability']:.3f})")
                print(f"  Semantic Similarity: {row['semantic_similarity']:.3f}")
                print()
        else:
            print(f"\n✗ No examples where {model_name} successfully removes bias")
            print()
            # Show first example anyway
            if len(results_df) > 0:
                row = results_df.iloc[0]
                print(f"First example (not removed):")
                print(f"  Original ({row['original_bias']}): {row['original']}")
                print(f"  Rewritten: {row['rewritten']}")
                print(f"  BERT Prediction: {row['predicted_label']} (confidence: {row['no_bias_probability']:.3f})")
                print()

def main():
    parser = argparse.ArgumentParser(description="Evaluate bias rewriter models")
    parser.add_argument('--data', type=str, default='multiclass-bias.csv', help='Test data CSV (EDIT: update if needed)')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples to evaluate (default: 100)')
    parser.add_argument('--baseline_only', action='store_true', help='Only evaluate baseline GPT-2')
    parser.add_argument('--ppo_only', action='store_true', help='Only evaluate PPO model')
    parser.add_argument('--checkpoint', type=str, default=None, help='Optional checkpoint path to evaluate instead of final PPO model (EDIT: update if needed)')
    parser.add_argument('--output', type=str, default='eval_results.csv', help='Output CSV for detailed results')
    
    args = parser.parse_args()
    
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, args.data)
    
    # ===== CONFIGURATION: Update these paths if needed =====
    # Path to the trained BERT bias classifier model saved by BERTs/train_bert.py
    bert_dir = os.path.join(base_path, "bert_bias_classifier")  # EDIT: Update to your BERT model directory
    
    # Path to the trained PPO bias rewriting model (saved by rl_bias/rl_bias_gpt2.py)
    # Or provide via --checkpoint argument
    ppo_model_base_dir = os.path.join(base_path, "gpt2_bias_rewriter_ppo")  # EDIT: Update if needed
    # ===== END CONFIGURATION =====
    
    baseline_model_dir = None  # Will use HuggingFace GPT-2
    
    # Use checkpoint if provided, otherwise use final PPO model
    if args.checkpoint:
        ppo_model_dir = args.checkpoint
        model_name = f"PPO Checkpoint ({os.path.basename(args.checkpoint)})"
    else:
        ppo_model_dir = ppo_model_base_dir
        model_name = "PPO-trained GPT-2"
    
    print(f"Device: {DEVICE}")
    
    # Load test data
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    biased_df = df[df["bias_label"] != "no_bias"].reset_index(drop=True)
    
    if len(biased_df) == 0:
        print("No biased samples found!")
        return
    
    # Sample subset if requested
    if args.num_samples > 0 and len(biased_df) > args.num_samples:
        test_df = biased_df.sample(n=args.num_samples, random_state=42).reset_index(drop=True)
        print(f"Using {args.num_samples} samples for evaluation")
    else:
        test_df = biased_df
        print(f"Using all {len(test_df)} biased samples for evaluation")
    
    # Load BERT reward model
    print(f"\nLoading BERT classifier from {bert_dir}...")
    bert_model, bert_tokenizer, le = load_bert_reward_model(bert_dir)
    
    # Load SBERT for semantic similarity
    print(f"Loading SBERT for semantic similarity...")
    sbert_model = load_sbert_model(DEVICE)
    
    # Storage for results
    all_results = {}
    
    # Evaluate baseline GPT-2
    if not args.ppo_only:
        print(f"\nLoading baseline GPT-2...")
        baseline_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if baseline_tokenizer.pad_token is None:
            baseline_tokenizer.pad_token = baseline_tokenizer.eos_token
        baseline_tokenizer.padding_side = "left"
        baseline_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        baseline_model.eval()
        
        baseline_results = evaluate_model(
            baseline_model, baseline_tokenizer, test_df,
            bert_model, bert_tokenizer, le, sbert_model,
            model_name="Baseline GPT-2"
        )
        all_results['baseline'] = baseline_results
    
    # Evaluate PPO-trained model
    if not args.baseline_only:
        ppo_checkpoint_dir = args.checkpoint if args.checkpoint else os.path.join(base_path, "gpt2_bias_rewriter_ppo_checkpoint_batch_2000")
        print(f"\nLoading {model_name} from {ppo_checkpoint_dir}...")
        try:
            ppo_tokenizer = AutoTokenizer.from_pretrained(ppo_checkpoint_dir)
            ppo_model = GPT2LMHeadModel.from_pretrained(ppo_checkpoint_dir).to(DEVICE)
        except Exception as e:
            print(f"⚠️  Could not load PPO model from {ppo_checkpoint_dir}: {e}")
            print(f"Using base gpt2 tokenizer and model as fallback...")
            ppo_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            ppo_model = GPT2LMHeadModel.from_pretrained("gpt2").to(DEVICE)
        
        if ppo_tokenizer.pad_token is None:
            ppo_tokenizer.pad_token = ppo_tokenizer.eos_token
        ppo_tokenizer.padding_side = "left"
        ppo_model.eval()
        
        ppo_results = evaluate_model(
            ppo_model, ppo_tokenizer, test_df,
            bert_model, bert_tokenizer, le, sbert_model,
            model_name=model_name
        )
        all_results['ppo'] = ppo_results
    
    # Print example results
    print_example_results(all_results, num_examples=3)
    
    # Summary comparison
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        baseline_removal = all_results['baseline']['bias_removal_rate'] * 100
        ppo_removal = all_results['ppo']['bias_removal_rate'] * 100
        improvement = ppo_removal - baseline_removal
        
        print(f"\nBias Removal Rate (% no_bias):")
        print(f"  Baseline:  {baseline_removal:.2f}%")
        print(f"  PPO:       {ppo_removal:.2f}%")
        print(f"  Improvement: {improvement:+.2f}%")
        
        print(f"\nAvg P(no_bias) from BERT:")
        print(f"  Baseline:  {all_results['baseline']['avg_no_bias_prob']:.4f}")
        print(f"  PPO:       {all_results['ppo']['avg_no_bias_prob']:.4f}")
        
        print(f"\nAvg Semantic Similarity (SBERT):")
        print(f"  Baseline:  {all_results['baseline']['avg_semantic_similarity']:.4f}")
        print(f"  PPO:       {all_results['ppo']['avg_semantic_similarity']:.4f}")
    
    # Save detailed results
    if 'ppo' in all_results:
        output_path = os.path.join(base_path, args.output)
        all_results['ppo']['results_df'].to_csv(output_path, index=False)
        print(f"\n✓ Detailed results saved to: {output_path}")

if __name__ == "__main__":
    main()
