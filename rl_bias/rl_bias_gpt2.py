

"""
This file contains an implementation of the PPO reinforcement learning setup using
the GPT-2 model.
"""

import os
import sys
import warnings
from typing import List

import torch
import pandas as pd
import joblib
import wandb
import numpy as np

from transformers import AutoTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from sentence_transformers import SentenceTransformer

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import WANDB_API_KEY

# ----------------------------- CONFIGURATION: Update these paths before running ----------------------------
# Path to the multiclass bias CSV file created by dataset_utils/create_multiclass_bias.py
DATA_FILE = "multiclass-bias.csv"  # EDIT: Update to your data file path

# Path to the trained BERT bias classifier model saved by BERTs/train_bert.py
BERT_MODEL_DIR = "bert_bias_classifier"  # EDIT: Update to your BERT model directory

# Directory where trained RL model checkpoints will be saved
SAVE_DIR = "gpt2_large_bias_rewriter_ppo"  # EDIT: Where to save trained models

# ===== Model and Training Configuration =====
LLM_MODEL = "gpt2-large"  # 774M params - 6.2x larger than gpt2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Lengths
PROMPT_MAX_LEN = 256   # prompt/token budget to model
BERT_MAX_LEN = 128     # BERT classifier max len

# Training schedule
EPOCHS = 3             # Fewer epochs to prevent divergence
PPO_BATCH_SIZE = 4     # Smaller batches for more stable learning with sparse rewards
PPO_MINI_BATCH_SIZE = 2   # REDUCED: Smaller mini-batches prevent reward exploitation
LEARNING_RATE = 5.0e-8   # MUCH more conservative - prevent policy divergence

# Reward function weights (from paper: Section 5.2)
# R(ŷ|x) = r_unbiased(ŷ) only - removed similarity to simplify signal
# ALPHA and BETA kept for backward compatibility but BETA=0 disables similarity
ALPHA = 1.0            # Full weight on bias removal
BETA = 0.0             # DISABLED: Remove similarity component for clarity

# Generation kwargs (stochastic sampling prevents mode collapse)
GEN_KWARGS = dict(
    max_new_tokens=64,
    do_sample=True,    # CRITICAL: Use stochastic sampling for diversity
    temperature=1.3,   # INCREASED: Higher temperature forces more exploration
    top_p=0.9,         # Reduced top_p to prevent extreme tokens
    top_k=50,          # Add top_k filtering for more controlled diversity
)

# Weights & Biases
PROJECT_NAME = "mbib-rl-bias-rewriting"
RUN_NAME = "gpt2-large-ppo-bias-removal"

# --------------------------- Utilities ----------------------------
def set_seeds(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_tokenizer_padding_for_gpt2(tok):
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # safer for causal masks

# Initialize wandb
def init_wandb():
    if WANDB_API_KEY:
        wandb.login(key=WANDB_API_KEY, relogin=True)
        wandb.init(
            project=PROJECT_NAME,
            name=RUN_NAME,
            config={
                "epochs": EPOCHS,
                "batch_size": PPO_BATCH_SIZE,
                "mini_batch_size": PPO_MINI_BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "llm_model": LLM_MODEL,
                "seed": SEED,
            }
        )
        print("✓ Weights & Biases logging enabled")
        return True
    else:
        print("⚠ WANDB_API_KEY not set in config.py, logging disabled")
        return False

# ---------------------------- Script ------------------------------
def main():
    set_seeds(SEED)
    wandb_enabled = init_wandb()

    # Paths
    base_path = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_path, DATA_FILE)
    bert_dir = os.path.join(base_path, BERT_MODEL_DIR)
    save_path = os.path.join(base_path, SAVE_DIR)
    os.makedirs(save_path, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Output directory: {save_path}")

    # --------- Load data ---------
    print(f"\nLoading data from {data_path}...")
    df = pd.read_csv(data_path)
    biased_df = df[df["bias_label"] != "no_bias"].reset_index(drop=True)
    if biased_df.empty:
        raise ValueError("No biased rows found (bias_label != 'no_bias'). Check your dataset.")
    
    print(f"Found {len(biased_df)} biased sentences for training")

    # --------- Load BERT reward model & label encoder ---------
    print(f"\nLoading BERT classifier from {bert_dir}...")
    bert_model = BertForSequenceClassification.from_pretrained(bert_dir).to(DEVICE)
    bert_tokenizer = BertTokenizer.from_pretrained(bert_dir)
    le = joblib.load(os.path.join(bert_dir, "label_encoder.joblib"))
    bert_model.eval()

    MAX_REWARD = 1.0
    MIN_REWARD = 0.0

    @torch.no_grad()
    def classify_bias_label(text: str) -> str:
        enc = bert_tokenizer(
            text, truncation=True, padding=True, max_length=BERT_MAX_LEN, return_tensors="pt"
        ).to(DEVICE)
        logits = bert_model(**enc).logits
        pred_id = int(logits.argmax(dim=1).item())
        return le.inverse_transform([pred_id])[0]

    @torch.no_grad()
    def compute_reward(orig_text: str, rewritten_text: str, sbert_model) -> float:
        """
        Reward function from paper (Section 5.2):
        R(ŷ|x) = α * r_unbiased(ŷ) + β * r_similarity(x, ŷ)
        
        Where:
        - r_unbiased: Probability of 'no_bias' class from BERT classifier
        - r_similarity: Cosine similarity between original and rewritten text (SBERT)
        
        Args:
            orig_text: Original biased text
            rewritten_text: LLM-generated rewritten text
            sbert_model: Sentence-BERT model for semantic similarity
        
        Returns:
            Combined reward score in range [0, 1]
        """
        # 1. Compute r_unbiased: BERT classifier probability for 'no_bias' class
        try:
            tokens = bert_tokenizer(
                rewritten_text,
                add_special_tokens=True,
                max_length=BERT_MAX_LEN,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(DEVICE)
            
            with torch.no_grad():
                outputs = bert_model(**tokens)
                logits = outputs.logits
                # Get softmax probabilities
                probs = torch.softmax(logits, dim=-1)
                # Get probability of 'no_bias' class
                no_bias_idx = list(le.classes_).index('no_bias')
                r_unbiased = float(probs[0, no_bias_idx].item())
        except Exception as e:
            print(f"Warning: Error computing BERT reward: {e}")
            r_unbiased = 0.0
        
        # 2. Compute r_similarity: Cosine similarity from SBERT embeddings
        # Use as a soft penalty: penalize divergence but allow some paraphrasing
        if sbert_model is not None:
            try:
                with torch.no_grad():
                    orig_embedding = sbert_model.encode(orig_text, convert_to_tensor=True)
                    rewritten_embedding = sbert_model.encode(rewritten_text, convert_to_tensor=True)
                    
                    # Compute cosine similarity (range: [-1, 1])
                    similarity = float(torch.nn.functional.cosine_similarity(
                        orig_embedding.unsqueeze(0),
                        rewritten_embedding.unsqueeze(0)
                    ).item())
                    # Soft penalty: apply a threshold at 0.5
                    # If similarity >= 0.5, reward is 1.0 (preserve meaning)
                    # If similarity < 0.5, linearly scale down (penalize divergence)
                    if similarity >= 0.5:
                        r_similarity = 1.0
                    else:
                        # For similarity in [0, 0.5), scale from 0 to 1
                        # At similarity=0: r_similarity=0
                        # At similarity=0.5: r_similarity=1
                        r_similarity = similarity / 0.5  # Maps [0, 0.5] to [0, 1]
            except Exception as e:
                print(f"Warning: Error computing SBERT similarity: {e}")
                r_similarity = 0.5
        else:
            r_similarity = 0.5
        
        # 3. Combine rewards as per paper formula
        # For now, use only r_unbiased (BETA=0 disables similarity)
        combined_reward = ALPHA * r_unbiased
        
        # Validate and clamp reward to [0, 1]
        combined_reward = max(0.0, min(1.0, combined_reward))
        
        return combined_reward

    # --------- Load models ---------
    print(f"\nLoading LLM model: {LLM_MODEL}...")
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    ensure_tokenizer_padding_for_gpt2(llm_tokenizer)

    policy = AutoModelForCausalLMWithValueHead.from_pretrained(LLM_MODEL).to(DEVICE)
    policy.config.pad_token_id = llm_tokenizer.pad_token_id

    ref_policy = AutoModelForCausalLMWithValueHead.from_pretrained(LLM_MODEL).to(DEVICE)
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    ref_policy.config.pad_token_id = llm_tokenizer.pad_token_id
    
    # Load SBERT for semantic similarity scoring
    print(f"Loading SBERT model for semantic similarity computation...")
    try:
        sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(DEVICE)
        sbert_model.eval()
        print("✓ SBERT model loaded successfully")
    except Exception as e:
        print(f"⚠ Error loading SBERT: {e}")
        print("Falling back to reduced similarity computation")
        sbert_model = None

    # --------- PPO config/trainer ---------
    print(f"\nInitializing PPO trainer...")
    ppo_config = PPOConfig(
        learning_rate=LEARNING_RATE,
        batch_size=PPO_BATCH_SIZE,
        mini_batch_size=PPO_MINI_BATCH_SIZE,
        seed=SEED,
        cliprange=0.1,       # REDUCED: Tighter clip range prevents aggressive updates
        cliprange_value=0.1, # REDUCED: Tighter value clipping
        vf_coef=0.1,        # Lower value coeff - focus more on policy than value
        kl_penalty="kl",    # Standard KL penalty for stability
        target_kl=0.05,     # MUCH TIGHTER: Early stop if KL > 0.05
        gamma=0.99,         # Discount factor
        lam=0.95,           # GAE lambda
        whiten_rewards=False,  # DISABLE: destroys signal with sparse rewards
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy,
        ref_model=ref_policy,
        tokenizer=llm_tokenizer,
    )

    # --------- Train loop ---------
    print(f"\nStarting PPO training for {EPOCHS} epochs...")
    print(f"Reward Function (Paper Section 5.2): R(ŷ|x) = {ALPHA}*r_unbiased(ŷ) + {BETA}*r_similarity(x,ŷ)")
    total_rows = len(biased_df)
    total_batches = (total_rows + PPO_BATCH_SIZE - 1) // PPO_BATCH_SIZE
    global_step = 0
    print(f"Total samples: {total_rows}, Total batches per epoch: {total_batches}\n")
    sys.stdout.flush()

    for epoch in range(EPOCHS):
        epoch_rewards = []
        epoch_loss = None
        consecutive_negative_kl = 0  # Track negative KL divergence
        global_batch_count = 0  # Track batches across epochs
        
        for batch_num, start in enumerate(range(0, total_rows, PPO_BATCH_SIZE)):
            batch_df = biased_df.iloc[start:start + PPO_BATCH_SIZE]

            # 1) Build prompts
            prompts: List[str] = []
            orig_biases: List[str] = []
            for row in batch_df.itertuples(index=False):
                text = str(row.text)
                orig_bias = str(row.bias_label)
                prompt = (
                    f"Rewrite the sentence to remove {orig_bias.replace('_', ' ')} "
                    f"while preserving meaning and grammar. Output only the revised sentence.\n\n"
                    f"Sentence: {text}\nRewritten:"
                )
                prompts.append(prompt)
                orig_biases.append(orig_bias)

            # 2) Tokenize queries
            query_tensors = []
            attn_masks = []
            for prompt in prompts:
                enc = llm_tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=PROMPT_MAX_LEN,
                )
                input_ids = enc["input_ids"][0].to(DEVICE)
                attention_mask = enc["attention_mask"][0].to(DEVICE)
                query_tensors.append(input_ids)
                attn_masks.append(attention_mask)

            # 3) Generate responses
            response_tensors = []
            with torch.no_grad():
                for q, m in zip(query_tensors, attn_masks):
                    m2 = m.unsqueeze(0)
                    gen_out = ppo_trainer.generate(
                        q,
                        attention_mask=m2,
                        eos_token_id=llm_tokenizer.eos_token_id,
                        pad_token_id=llm_tokenizer.pad_token_id,
                        **GEN_KWARGS,
                    )
                    if gen_out.dim() == 1:
                        resp_only = gen_out[q.shape[0]:]
                    else:
                        resp_only = gen_out[0, q.shape[0]:]
                    response_tensors.append(resp_only)

            # 4) Decode for reward computation
            responses_text = [
                llm_tokenizer.decode(r_ids, skip_special_tokens=True)
                for r_ids in response_tensors
            ]

            # 5) Compute rewards using paper's formula: R(ŷ|x) = α*r_unbiased + β*r_similarity
            raw_rewards = [
                compute_reward(orig_text, resp_text, sbert_model)
                for orig_text, resp_text in zip(prompts, responses_text)
            ]
            
            # Convert to numpy for analysis
            raw_rewards_np = np.array(raw_rewards)
            epoch_rewards.extend(raw_rewards_np.tolist())  # Track raw rewards for logging
            
            # IMPORTANT: Keep raw rewards but center them around 0.5
            # This maintains the reward signal while reducing KL divergence risk
            # Subtract 0.5 so rewards are roughly centered around 0
            centered_rewards = raw_rewards_np - 0.5  # Now roughly in range [-0.5, 0.5]
            centered_rewards = np.clip(centered_rewards, -0.3, 0.3)  # Hard cap to prevent extremes
            
            rewards = [
                torch.tensor(r, device=DEVICE, dtype=torch.float32)
                for r in centered_rewards
            ]

            # 6) PPO update
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            epoch_loss = stats.get("ppo/loss", None)
            global_batch_count += 1

            # 6b) Save checkpoint every 1000 batches
            if global_batch_count % 1000 == 0:
                checkpoint_dir = os.path.join(base_path, f"{SAVE_DIR}_checkpoint_batch_{global_batch_count}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                ppo_trainer.model.save_pretrained(checkpoint_dir)
                llm_tokenizer.save_pretrained(checkpoint_dir)
                print(f"✓ Checkpoint saved: {checkpoint_dir}")
                sys.stdout.flush()

            # 7) Log to wandb
            if wandb_enabled:
                log_dict = {
                    "epoch": epoch + 1,
                    "batch": batch_num + 1,
                    "avg_reward": sum(epoch_rewards) / len(epoch_rewards),
                    "max_reward": max(epoch_rewards),
                    "min_reward": min(epoch_rewards),
                    "batch_avg_reward": sum([float(r.item()) for r in rewards]) / len(rewards),
                }
                if epoch_loss is not None:
                    log_dict["ppo_loss"] = epoch_loss
                for key in ["policy/approxkl", "policy/clipfrac", "policy/entropy", "objective/kl"]:
                    if key in stats:
                        log_dict[key] = stats[key]
                wandb.log(log_dict)

            # 8) Console feedback
            rlist_centered = [round(r, 3) for r in centered_rewards]
            rlist_raw = [round(r, 3) for r in raw_rewards_np]
            kl = stats.get("policy/kl", None)
            kl_str = f"KL={kl:.4f} | " if kl is not None else ""
            
            # Check for negative KL (sign of divergence)
            if kl is not None and kl < 0:
                consecutive_negative_kl += 1
                print(f"⚠️  WARNING: Negative KL detected! ({consecutive_negative_kl} consecutive)")
                if consecutive_negative_kl >= 3:
                    print(f"❌ CRITICAL: KL divergence failure after {consecutive_negative_kl} consecutive negative values!")
                    print(f"Stopping training to prevent model degradation.")
                    sys.exit(1)
            else:
                consecutive_negative_kl = 0
            
            avg_reward = sum(epoch_rewards) / len(epoch_rewards)
            num_improved = sum(1 for r in raw_rewards_np if r > 0.5)
            print(
                f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_num+1}/{total_batches} | "
                f"{kl_str}AvgRaw={np.mean(raw_rewards_np):.3f} | Improved={num_improved}/{len(rewards)} | RawRewards={rlist_raw}"
            )

    # --------- Save ---------
    print(f"\nSaving model to {save_path}...")
    ppo_trainer.model.save_pretrained(save_path)
    llm_tokenizer.save_pretrained(save_path)
    print(f"✓ Model and tokenizer saved to: {save_path}")

    if wandb_enabled:
        wandb.log({"training_complete": True})
        wandb.finish()

    print("✓ Training complete!")

if __name__ == "__main__":
    main()
