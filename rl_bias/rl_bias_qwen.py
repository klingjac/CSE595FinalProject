

"""
This file contains an attempted implementation of the PPO rienforcement learning
setup with Qwen3-4B in place of GPT-2. This setup however is incomplete
and requires further work to be functional.
"""

import os
import sys
import warnings
from typing import List
import re

import torch
import pandas as pd
import joblib
import wandb
import numpy as np

from transformers import AutoTokenizer
from transformers import BertTokenizer, BertForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from sentence_transformers import SentenceTransformer
from peft import LoraConfig

# ===================== CONFIGURATION: Update these paths before running =====================
# Path to the multiclass bias CSV file created by dataset_utils/create_multiclass_bias.py
DATA_FILE = "multiclass-bias.csv"  # EDIT: Update to your data file path

# Path to the trained BERT bias classifier model saved by BERTs/train_bert.py
BERT_MODEL_DIR = "bert_bias_classifier"  # EDIT: Update to your BERT model directory

# ===== Model and Training Configuration =====
# Qwen model from Hugging Face
LLM_MODEL = "Qwen/Qwen3-4B"
# If you want, you can try the instruct version:
# LLM_MODEL = "Qwen/Qwen3-4B-Instruct-2507"

# Directory where trained RL model checkpoints will be saved
SAVE_DIR = "qwen3_4b_qllora_bias_rewriter_ppo"  # EDIT: Where to save trained models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

# Lengths
PROMPT_MAX_LEN = 256   # prompt/token budget to model
BERT_MAX_LEN = 128     # BERT classifier max len

# Training schedule
EPOCHS = 3
PPO_BATCH_SIZE = 2
PPO_MINI_BATCH_SIZE = 1
LEARNING_RATE = 5.0e-8   # conservative to avoid divergence

# Reward function weights
ALPHA = 1.0   # weight on discrete bias reward
BETA = 0.1    # weight on semantic similarity (SBERT-based)

# Generation kwargs (safer sampling)
GEN_KWARGS = dict(
    max_new_tokens=64,
    do_sample=True,
    temperature=0.7,   # lower = more stable
    top_p=0.9,
    top_k=40,
)

# Weights & Biases
PROJECT_NAME = "mbib-rl-bias-rewriting"
RUN_NAME = "qwen3-4b-qllora-ppo-bias-removal"

# =============================================================

# Enable TF32 for speed on Ampere+ GPUs (safe even if CPU-only)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import WANDB_API_KEY


# --------------------------- Utilities ----------------------------
def set_seeds(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_tokenizer_padding(tok) -> bool:
    """
    Ensure tokenizer has a pad token and left padding.

    Returns:
        added_new_token (bool): True if we added a new PAD token to the vocab.
    """
    added_new_token = False
    if tok.pad_token_id is None:
        if tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token
        else:
            # Add a dedicated PAD token if nothing else exists
            tok.add_special_tokens({"pad_token": "[PAD]"})
            added_new_token = True
    tok.padding_side = "left"
    return added_new_token


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
                "alpha": ALPHA,
                "beta": BETA,
            }
        )
        print("✓ Weights & Biases logging enabled")
        return True
    else:
        print("⚠ WANDB_API_KEY not set in config.py, logging disabled")
        return False


def clean_response(text: str) -> str:
    """
    Clean the raw model output before reward:
    - Strip whitespace.
    - Keep only the first line (drop meta chatter like "Okay, so I need to...").
    """
    text = text.strip()
    if "\n" in text:
        text = text.split("\n", 1)[0].strip()
    return text


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

    print(f"Device for reward models: {DEVICE}")
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

    @torch.no_grad()
    def classify_bias_label(text: str) -> str:
        enc = bert_tokenizer(
            text, truncation=True, padding=True, max_length=BERT_MAX_LEN, return_tensors="pt"
        ).to(DEVICE)
        logits = bert_model(**enc).logits
        pred_id = int(logits.argmax(dim=1).item())
        return le.inverse_transform([pred_id])[0]

    @torch.no_grad()
    def debug_reward_example(orig_bias: str, text: str):
        """
        Debug helper: print BERT's classification and reward components for a given text.
        """
        tokens = bert_tokenizer(
            text,
            add_special_tokens=True,
            max_length=BERT_MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(DEVICE)
        with torch.no_grad():
            outputs = bert_model(**tokens)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
        classes = list(le.classes_)
        pred_idx = int(probs.argmax())
        pred_label = classes[pred_idx]
        no_bias_idx = classes.index("no_bias")
        no_bias_prob = float(probs[no_bias_idx])

        # discrete bias reward
        if pred_label == "no_bias":
            r_unbiased = 1.0
        elif pred_label != orig_bias:
            r_unbiased = 0.5
        else:
            r_unbiased = 0.0

        print("---- BERT reward debug ----")
        print(f"Orig bias label: {orig_bias}")
        print(f"Pred label:      {pred_label}")
        print(f"P(no_bias):      {no_bias_prob:.3f}")
        print(f"r_unbiased:      {r_unbiased:.3f}")
        print("---------------------------")

    @torch.no_grad()
    def compute_reward(orig_text: str, orig_bias: str, rewritten_text: str, sbert_model) -> float:
        """
        Reward function:
        - Discrete bias reward from BERT labels:
            r_unbiased =
                1.0 if pred_label == "no_bias"
                0.5 if pred_label != orig_bias (but not "no_bias")
                0.0 if pred_label == orig_bias

        - Similarity reward r_similarity in [0, 1] from SBERT cosine similarity.

        Final: R(ŷ|x) = α * r_unbiased + β * r_similarity

        Also:
        - If the text looks like garbage (too few letters), reward = 0.
        """
        # 0. Garbage-output guard: if too few alphabetic characters, treat as nonsense
        letters = re.findall(r"[A-Za-z]", rewritten_text)
        if len(letters) < 10:
            return 0.0

        # 1. BERT classification -> discrete bias reward
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
                probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            classes = list(le.classes_)
            pred_idx = int(probs.argmax())
            pred_label = classes[pred_idx]

            if pred_label == "no_bias":
                r_unbiased = 1.0
            elif pred_label != orig_bias:
                r_unbiased = 0.5
            else:
                r_unbiased = 0.0
        except Exception as e:
            print(f"Warning: Error computing BERT reward: {e}")
            r_unbiased = 0.0
        
        # 2. Similarity component r_similarity in [0, 1]
        if sbert_model is not None:
            try:
                with torch.no_grad():
                    orig_embedding = sbert_model.encode(orig_text, convert_to_tensor=True)
                    rewritten_embedding = sbert_model.encode(rewritten_text, convert_to_tensor=True)
                    # cosine_similarity in [-1, 1]
                    similarity = float(torch.nn.functional.cosine_similarity(
                        orig_embedding.unsqueeze(0),
                        rewritten_embedding.unsqueeze(0)
                    ).item())
                    # Map [-1, 1] -> [0, 1]
                    r_similarity = (similarity + 1.0) / 2.0
                    r_similarity = max(0.0, min(1.0, r_similarity))
            except Exception as e:
                print(f"Warning: Error computing SBERT similarity: {e}")
                r_similarity = 0.5
        else:
            r_similarity = 0.5
        
        combined_reward = ALPHA * r_unbiased + BETA * r_similarity
        combined_reward = max(0.0, min(1.0, combined_reward))
        return combined_reward

    # --------- Load Qwen3-4B model & tokenizer (QLoRA) ---------
    print(f"\nLoading LLM model (QLoRA): {LLM_MODEL}...")
    llm_tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL,
        trust_remote_code=True
    )

    added_new_token = ensure_tokenizer_padding(llm_tokenizer)

    # LoRA config for Qwen3 (attention + MLP projection layers)
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )

    # Policy model: Qwen3-4B in 4-bit with LoRA adapters (trainable)
    policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        LLM_MODEL,
        trust_remote_code=True,
        load_in_4bit=True,      # 4-bit base weights (bitsandbytes)
        peft_config=lora_config # LoRA adapters on top
    )

    # Reference model: same base Qwen3-4B in 4-bit, no LoRA (frozen)
    ref_policy = AutoModelForCausalLMWithValueHead.from_pretrained(
        LLM_MODEL,
        trust_remote_code=True,
        load_in_4bit=True,
    )

    # If we added a new pad token, resize embeddings
    if added_new_token:
        policy.resize_token_embeddings(len(llm_tokenizer))
        ref_policy.resize_token_embeddings(len(llm_tokenizer))

    # Disable cache and enable grad checkpointing on policy for memory savings
    if hasattr(policy, "config"):
        policy.config.use_cache = False
    if hasattr(policy, "gradient_checkpointing_enable"):
        policy.gradient_checkpointing_enable()

    policy.config.pad_token_id = llm_tokenizer.pad_token_id
    ref_policy.config.pad_token_id = llm_tokenizer.pad_token_id

    # Freeze reference model entirely
    ref_policy.eval()
    for p in ref_policy.parameters():
        p.requires_grad_(False)
    
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
        cliprange=0.1,
        cliprange_value=0.1,
        vf_coef=0.1,
        kl_penalty="kl",
        target_kl=0.05,
        gamma=0.99,
        lam=0.95,
        whiten_rewards=True,    # let TRL normalize rewards
        max_grad_norm=0.3,      # extra safety on update size
    )

    ppo_trainer = PPOTrainer(
        config=ppo_config,
        model=policy,
        ref_model=ref_policy,
        tokenizer=llm_tokenizer,
    )

    # --------- Train loop ---------
    print(f"\nStarting PPO training for {EPOCHS} epochs...")
    print(f"Reward Function: R(ŷ|x) = {ALPHA}*r_unbiased(ŷ) + {BETA}*r_similarity(x,ŷ)")
    total_rows = len(biased_df)
    total_batches = (total_rows + PPO_BATCH_SIZE - 1) // PPO_BATCH_SIZE
    print(f"Total samples: {total_rows}, Total batches per epoch: {total_batches}\n")
    sys.stdout.flush()

    for epoch in range(EPOCHS):
        epoch_rewards = []
        epoch_loss = None
        consecutive_negative_kl = 0
        global_batch_count = 0
        
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
                input_ids = enc["input_ids"][0].to(ppo_trainer.model.current_device)
                attention_mask = enc["attention_mask"][0].to(ppo_trainer.model.current_device)
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

            # 4) Decode and clean for reward computation
            responses_text = [
                clean_response(llm_tokenizer.decode(r_ids, skip_special_tokens=True))
                for r_ids in response_tensors
            ]

            # ---- Optional logging of one example per batch ----
            if LOG_EXAMPLES and len(prompts) > 0 and len(responses_text) > 0:
                example_idx = 0
                example_prompt = prompts[example_idx]
                example_resp = responses_text[example_idx]
                example_orig_bias = orig_biases[example_idx]
                print("\n================ Example from batch ================")
                print(f"Epoch {epoch+1}, Batch {batch_num+1}")
                print("PROMPT:")
                print(example_prompt)
                print("\nRESPONSE (cleaned):")
                print(example_resp)
                print("===================================================\n")
                debug_reward_example(example_orig_bias, example_resp)
                sys.stdout.flush()
            # ---------------------------------------------------

            # 5) Compute rewards
            raw_rewards = [
                compute_reward(orig_text, orig_bias, resp_text, sbert_model)
                for orig_text, orig_bias, resp_text in zip(prompts, orig_biases, responses_text)
            ]
            
            raw_rewards_np = np.array(raw_rewards, dtype=np.float32)
            epoch_rewards.extend(raw_rewards_np.tolist())
            
            rewards = [
                torch.tensor(r, device=ppo_trainer.model.current_device, dtype=torch.float32)
                for r in raw_rewards_np
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
                    "avg_reward": float(sum(epoch_rewards) / len(epoch_rewards)),
                    "max_reward": float(max(epoch_rewards)),
                    "min_reward": float(min(epoch_rewards)),
                    "batch_avg_reward": float(raw_rewards_np.mean()),
                }
                if epoch_loss is not None:
                    log_dict["ppo_loss"] = epoch_loss
                for key in ["policy/approxkl", "policy/clipfrac", "policy/entropy", "objective/kl"]:
                    if key in stats:
                        log_dict[key] = stats[key]
                wandb.log(log_dict)

            # 8) Console feedback
            rlist_raw = [round(float(r), 3) for r in raw_rewards_np]
            kl = stats.get("policy/kl", None)
            kl_str = f"KL={kl:.4f} | " if kl is not None else ""
            
            if kl is not None and kl < 0:
                consecutive_negative_kl += 1
                print(f"⚠️  WARNING: Negative KL detected! ({consecutive_negative_kl} consecutive)")
                if consecutive_negative_kl >= 3:
                    print(f"❌ CRITICAL: KL divergence failure after {consecutive_negative_kl} consecutive negative values!")
                    print(f"Stopping training to prevent model degradation.")
                    sys.exit(1)
            else:
                consecutive_negative_kl = 0
            
            num_improved = sum(1 for r in raw_rewards_np if r > 0.5)
            print(
                f"Epoch {epoch+1}/{EPOCHS} | Batch {batch_num+1}/{total_batches} | "
                f"{kl_str}AvgRaw={raw_rewards_np.mean():.3f} | "
                f"Improved={num_improved}/{len(rewards)} | RawRewards={rlist_raw}"
            )

    # --------- Save ---------
    print(f"\nSaving model (LoRA adapters) to {save_path}...")
    ppo_trainer.model.save_pretrained(save_path)
    llm_tokenizer.save_pretrained(save_path)
    print(f"✓ Model and tokenizer saved to: {save_path}")

    if wandb_enabled:
        wandb.log({"training_complete": True})
        wandb.finish()

    print("✓ Training complete!")

if __name__ == "__main__":
    main()
