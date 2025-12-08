import json
import os

notebook_path = "/Users/vedaangchopra/all_data/complete_technical_work/all_projects_implemented/Which_VLM_Router/artemis_final/router_train/notebooks/05_multitask_reward_router.ipynb"

cells = []

def add_markdown(source):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True)
    })

def add_code(source):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True)
    })

# ---------------------------------------------------------
# 1. Overview & Configuration
# ---------------------------------------------------------
add_markdown("""# Multi-Task Reward Router Training

This notebook trains a multi-task Artemis Router that predicts:
(a) **utility-based rewards** per (sample, model, mode), and
(b) the **router_task** (task_type) from prompt text.

It uses the new `utility_*` columns computed by the utility scoring module (assumed to be in `data/router_profiles_with_utility.parquet`).

## Approach

1.  **Load Data**: Validated profiles with pre-computed utilities.
2.  **Multi-Task Model**:
    *   **Shared Encoder**: DistilBERT (or config default).
    *   **Utility Head**: `(text + model + mode) -> scalar utility`.
    *   **Task Head**: `(text) -> task logits`.
3.  **Training**: Joint loss `L = L_utility + λ * L_task`.
4.  **Evaluation**: Routing accuracy vs Oracle, Task classification accuracy.

## Benefits
*   **Efficiency**: One model handles both task detection and routing.
*   **Aligned Objectives**: Learning task features helps routing; learning utility helps task separation.
""")

add_code("""# === Path Setup ===
import sys
from pathlib import Path

# Notebook paths
NOTEBOOK_DIR = Path.cwd()
ARTEMIS_DIR = NOTEBOOK_DIR.parent.parent  # artemis_final/
ROOT_DIR = ARTEMIS_DIR.parent             # Which_VLM_Router/
ROUTER_TRAIN_DIR = ARTEMIS_DIR / 'router_train'

# Add to sys.path for imports
for p in [str(ARTEMIS_DIR), str(ROOT_DIR), str(ROUTER_TRAIN_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

print(f"📁 Setup complete")

# Add artemis_final to path
# Also add router_train for local imports

import logging
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup
from tqdm.auto import tqdm

# Local imports
from config import Config, RouterModelConfig, TrainingConfig
from training.dataset import RewardRouterDataset, split_by_sample
from models.reward_router import RewardRouterModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("✓ Imports successful")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
""")

add_code("""# Configuration
# -----------------------------------------------------------------------------
CONFIG = Config.default()

# Dataset settings
DATASET_FILE = "router_profiles_with_utility.parquet" # Assumed pre-computed
DATA_PATH = CONFIG.paths.get_data_path(DATASET_FILE)

# Model settings
MODEL_NAME = "distilbert-base-uncased"
MAX_SEQ_LENGTH = 256

# Training settings
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 3e-5
LAMBDA_TASK = 0.5  # Weight for task classification loss
NV_WORKERS = 0     # 0 for notebook compatibility

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Config overrides
CONFIG.training.batch_size = BATCH_SIZE
CONFIG.training.num_epochs = NUM_EPOCHS
CONFIG.training.learning_rate = LEARNING_RATE
CONFIG.model.text_encoder_name = MODEL_NAME

print("Configuration:")
print(f"  DATA_PATH: {DATA_PATH}")
print(f"  DEVICE: {DEVICE}")
print(f"  BATCH_SIZE: {BATCH_SIZE}")
print(f"  NUM_EPOCHS: {NUM_EPOCHS}")
print(f"  LAMBDA_TASK: {LAMBDA_TASK}")
""")

# ---------------------------------------------------------
# 2. Load and Inspect Data
# ---------------------------------------------------------
add_markdown("""## 2. Load and Inspect Data

We load the parquet file containing `utility_*` columns.
""")

add_code("""print(f"Loading data from: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    print(f"⚠️ WARNING: File not found at {DATA_PATH}.")
    print("Please ensure the utility scoring module has run and produced this file.")
    # Fallback for notebook runnable state if file missing: create dummy
    print("Creating DUMMY dataframe for demonstration purposes...")
    df_profiles = pd.DataFrame({
        "sample_id": [f"sample_{i}" for i in range(100) for _ in range(3)],
        "model_name": ["model_a", "model_b", "model_c"] * 100,
        "router_task": ["ocr", "vqa", "reasoning"] * 100,
        "data_split": ["train"]*240 + ["val"]*30 + ["test"]*30,
        "prompt_text": ["describe this image"] * 300,
        "utility_accuracy": np.random.rand(300),
        "utility_cheap": np.random.rand(300),
        "utility_fast": np.random.rand(300),
        "utility_balanced": np.random.rand(300),
        "ok": [True] * 300
    })
else:
    df_profiles = pd.read_parquet(DATA_PATH)
    print(f"✓ Loaded {len(df_profiles)} rows.")

# Basic filtering
if "ok" in df_profiles.columns:
    df_profiles = df_profiles[df_profiles["ok"] == True]

# Check for utility columns
util_cols = [c for c in df_profiles.columns if c.startswith("utility_")]
print(f"Utility columns found: {util_cols}")

if not util_cols:
    print("⚠️ No utility_* columns found! Falling back to 'reward_' columns if available...")
    util_cols = [c for c in df_profiles.columns if c.startswith("reward_")]
    # Rename for consistency
    rename_map = {c: c.replace("reward_", "utility_") for c in util_cols}
    df_profiles.rename(columns=rename_map, inplace=True)
    util_cols = list(rename_map.values())

print(f"✓ Final DataFrame shape: {df_profiles.shape}")
print(f"  Unique samples: {df_profiles['sample_id'].nunique()}")
print(f"  Models: {df_profiles['model_name'].unique()}")

# Show sample
display(df_profiles.head())
""")

# ---------------------------------------------------------
# 3. Build Long-Form Router Training Dataset
# ---------------------------------------------------------
add_markdown("""## 3. Build Long-Form Dataset

Convert to `(sample, model, mode)` format.
Modes: `accuracy`, `cheap`, `fast`, `balanced`.
""")

add_code("""MODES = ["accuracy", "cheap", "fast", "balanced"]

def build_long_form(df_in):
    rows = []
    
    # Pre-fetch columns to avoid repetitive access
    # Using itertuples for speed or just simple iteration if size allows
    # For creating a new dataframe, usually list of dicts is easiest to reason about
    
    for idx, row in tqdm(df_in.iterrows(), total=len(df_in), desc="Building long-form"):
        # Common fields
        sample_id = row["sample_id"]
        router_task = row.get("router_task", "unknown")
        data_split = row.get("data_split", "train")
        model_name = row["model_name"]
        
        # Metadata for prompt building (some might be missing, use safe gets)
        prompt_raw = row.get("vlm_samples.prompt_text", row.get("prompt_text", ""))
        prompt_len = row.get("prompt_len_words", 0)
        img_w = row.get("img_width", 0)
        img_h = row.get("img_height", 0)
        img_ar = row.get("img_aspect_ratio", 1.0)
        source_dataset = row.get("source_dataset", "unknown")
        
        # Create row for each mode
        for mode in MODES:
            # Map mode to utility column
            util_col = f"utility_{mode}"
            
            # Check if we have a valid target
            if util_col not in row or pd.isna(row[util_col]):
                continue
                
            val = row[util_col]
            
            rows.append({
                "sample_id": sample_id,
                "router_task": router_task,
                "data_split": data_split,
                "model_name": model_name,
                "mode_name": mode,
                "utility_target": float(val),
                
                # Metadata columns for Dataset class
                "prompt_raw": prompt_raw,
                "prompt_len_words": prompt_len,
                "img_width": img_w,
                "img_height": img_h,
                "img_aspect_ratio": img_ar,
                "source_dataset": source_dataset,
                "source_config": "default" # placeholder
            })
            
    return pd.DataFrame(rows)

df_long = build_long_form(df_profiles)
print(f"Long-form shape: {df_long.shape}")
print("Rows per mode:")
print(df_long["mode_name"].value_counts())
""")

# ---------------------------------------------------------
# 4. Prepare Mappings
# ---------------------------------------------------------
add_markdown("""## 4. Prepare Mappings (Models, Modes, Tasks)

We need integer IDs for:
*   Models (for embedding)
*   Modes (for embedding)
*   Tasks (for classification head)
""")

add_code("""# 1. Model ID
unique_models = sorted(df_long["model_name"].unique())
model_to_id = {name: i for i, name in enumerate(unique_models)}
df_long["model_id"] = df_long["model_name"].map(model_to_id)

# 2. Mode ID
mode_to_id = {name: i for i, name in enumerate(MODES)}
df_long["mode_id"] = df_long["mode_name"].map(mode_to_id)

# 3. Task ID
unique_tasks = sorted(df_long["router_task"].unique())
task_to_id = {name: i for i, name in enumerate(unique_tasks)}
df_long["task_id"] = df_long["router_task"].map(task_to_id)

print("Models:", model_to_id)
print("Modes:", mode_to_id)
print("Tasks:", task_to_id)

# Save indices
def save_json(data, filename):
    path = CONFIG.paths.get_data_path(filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved {path}")

save_json(model_to_id, "model_index.json")
save_json(mode_to_id, "mode_index.json")
save_json(task_to_id, "task_index.json")
""")

# ---------------------------------------------------------
# 5. Define Multi-Task Router Model
# ---------------------------------------------------------
add_markdown("""## 5. Define Multi-Task Router Model

Extending the architecture to include a `task_head`.
""")

add_code("""class MultiTaskRouterModel(nn.Module):
    def __init__(
        self,
        config: RouterModelConfig,
        num_models: int,
        num_modes: int,
        num_tasks: int,
    ):
        super().__init__()
        self.config = config
        
        # 1. Text Encoder (Shared)
        self.text_encoder = AutoModel.from_pretrained(config.text_encoder_name)
        self.text_hidden_size = self.text_encoder.config.hidden_size
        
        if config.freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
                
        # 2. Embeddings
        self.model_embedding = nn.Embedding(num_models, config.model_emb_dim)
        self.mode_embedding = nn.Embedding(num_modes, config.mode_emb_dim)
        
        # 3. Utility Head (Routing)
        # Input: Text + Model + Mode
        router_input_dim = self.text_hidden_size + config.model_emb_dim + config.mode_emb_dim
        
        self.utility_mlp = nn.Sequential(
            nn.Linear(router_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1)
        )
        
        # 4. Task Head (Classification)
        # Input: Text only
        self.task_head = nn.Sequential(
            nn.Linear(self.text_hidden_size, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, num_tasks)
        )
        
    def forward(self, input_ids, attention_mask, model_id, mode_id, **kwargs):
        # Encode text
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Pooling (CLS)
        last_hidden = outputs.last_hidden_state
        h_text = last_hidden[:, 0, :] # [batch, hidden]
        
        # --- Task Classification ---
        task_logits = self.task_head(h_text) # [batch, num_tasks]
        
        # --- Utility Prediction ---
        h_model = self.model_embedding(model_id) # [batch, model_emb_dim]
        h_mode = self.mode_embedding(mode_id)    # [batch, mode_emb_dim]
        
        # Concat
        h_router = torch.cat([h_text, h_model, h_mode], dim=-1)
        
        # Predict
        utility_hat = self.utility_mlp(h_router).squeeze(-1) # [batch]
        
        return {
            "utility_hat": utility_hat,
            "task_logits": task_logits
        }

""")

# ---------------------------------------------------------
# 6. Build Dataset & DataLoaders
# ---------------------------------------------------------
add_markdown("""## 6. Build Dataset & DataLoaders

We extend `RewardRouterDataset` or just reuse it and let it return what it returns, but we NEED `task_id`.
The existing `RewardRouterDataset` returns:
`input_ids`, `attention_mask`, `model_id`, `mode_id`, `reward`, `sample_id`.

It does NOT return `task_id` in `__getitem__`.
We will implement a custom `MultiTaskDataset`.
""")

add_code("""class MultiTaskDataset(RewardRouterDataset):
    def __getitem__(self, idx):
        # reuse parent to get base items or row
        row = self.df.iloc[idx]
        
        # Get base item using parent logic (text building + tokenization)
        # But wait, parent __getitem__ does tokenization inside.
        # We can either duplicate logic or call parent.
        # Calling parent is easier if we can extract task_id from row safely.
        
        item = super().__getitem__(idx)
        
        # Add task_id
        # We ensured df has task_id column in step 4
        item["task_id"] = torch.tensor(row["task_id"], dtype=torch.long)
        
        # Rename 'reward' to 'utility_target' for clarity if preferred, 
        # but 'reward' key is fine.
        item["utility_target"] = item["reward"] # alias
        
        return item

def multitask_collate_fn(batch):
    base_batch = {}
    
    # Stack tensors
    for key in ["input_ids", "attention_mask", "model_id", "mode_id", "utility_target", "task_id", "reward"]:
        if key in batch[0]:
            base_batch[key] = torch.stack([item[key] for item in batch])
            
    # List sample_ids
    if "sample_id" in batch[0]:
        base_batch["sample_ids"] = [item["sample_id"] for item in batch]
        
    return base_batch

# Split by sample_id
train_df, val_df, test_df = split_by_sample(
    df_long,
    train_ratio=CONFIG.training.train_ratio,
    val_ratio=CONFIG.training.val_ratio,
    test_ratio=CONFIG.training.test_ratio,
    seed=CONFIG.training.seed
)

tokenizer = AutoTokenizer.from_pretrained(CONFIG.model.text_encoder_name)

train_dataset = MultiTaskDataset(train_df, tokenizer, max_seq_length=CONFIG.model.max_seq_length, split="train")
val_dataset = MultiTaskDataset(val_df, tokenizer, max_seq_length=CONFIG.model.max_seq_length, split="val", enable_augmentation=False)
test_dataset = MultiTaskDataset(test_df, tokenizer, max_seq_length=CONFIG.model.max_seq_length, split="test", enable_augmentation=False)

train_loader = DataLoader(
    train_dataset, 
    batch_size=CONFIG.training.batch_size, 
    shuffle=True, 
    collate_fn=multitask_collate_fn, 
    num_workers=NV_WORKERS
) 
val_loader = DataLoader(
    val_dataset, 
    batch_size=CONFIG.training.batch_size, 
    shuffle=False, 
    collate_fn=multitask_collate_fn,
    num_workers=NV_WORKERS
)
test_loader = DataLoader(
    test_dataset, 
    batch_size=CONFIG.training.batch_size, 
    shuffle=False, 
    collate_fn=multitask_collate_fn,
    num_workers=NV_WORKERS
)

print(f"Train batches: {len(train_loader)}")
print(f"Val batches:   {len(val_loader)}")
""")

# ---------------------------------------------------------
# 7. Training Loop
# ---------------------------------------------------------
add_markdown("""## 7. Training Loop

Multi-task loss: $L = L_{routing} + \lambda \cdot L_{task}$
""")

add_code("""num_models = len(model_to_id)
num_modes = len(mode_to_id)
num_tasks = len(task_to_id)

model = MultiTaskRouterModel(CONFIG.model, num_models=num_models, num_modes=num_modes, num_tasks=num_tasks)
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=CONFIG.training.learning_rate, weight_decay=CONFIG.training.weight_decay)
training_steps = len(train_loader) * CONFIG.training.num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1*training_steps), num_training_steps=training_steps)

loss_fn_utility = nn.MSELoss()
loss_fn_task = nn.CrossEntropyLoss()

print(f"Starting training on {DEVICE}...")

best_val_loss = float("inf")
best_model_state = None

for epoch in range(CONFIG.training.num_epochs):
    model.train()
    train_losses = []
    
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG.training.num_epochs}"):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        model_id = batch["model_id"].to(DEVICE)
        mode_id = batch["mode_id"].to(DEVICE)
        task_id = batch["task_id"].to(DEVICE)
        utility_target = batch["utility_target"].to(DEVICE)
        
        optimizer.zero_grad()
        
        outputs = model(input_ids, attention_mask, model_id, mode_id)
        
        loss_u = loss_fn_utility(outputs["utility_hat"], utility_target)
        loss_t = loss_fn_task(outputs["task_logits"], task_id)
        
        loss = loss_u + LAMBDA_TASK * loss_t
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG.training.gradient_clip_norm)
        optimizer.step()
        scheduler.step()
        
        train_losses.append(loss.item())
        
    avg_train_loss = np.mean(train_losses)
    
    # Validation
    model.eval()
    val_losses = []
    val_u_losses = []
    val_task_correct = 0
    val_task_total = 0
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            model_id = batch["model_id"].to(DEVICE)
            mode_id = batch["mode_id"].to(DEVICE)
            task_id = batch["task_id"].to(DEVICE)
            utility_target = batch["utility_target"].to(DEVICE)
            
            outputs = model(input_ids, attention_mask, model_id, mode_id)
            
            l_u = loss_fn_utility(outputs["utility_hat"], utility_target)
            l_t = loss_fn_task(outputs["task_logits"], task_id)
            
            val_loss = l_u + LAMBDA_TASK * l_t
            val_losses.append(val_loss.item())
            val_u_losses.append(l_u.item())
            
            # Task acc
            preds = torch.argmax(outputs["task_logits"], dim=1)
            val_task_correct += (preds == task_id).sum().item()
            val_task_total += len(task_id)
            
    avg_val_loss = np.mean(val_losses)
    avg_val_u_loss = np.mean(val_u_losses)
    val_task_acc = val_task_correct / val_task_total if val_task_total > 0 else 0.0
    
    print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f} (Util={avg_val_u_loss:.4f}, TaskAcc={val_task_acc:.2%})")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_model_state = model.state_dict()
        torch.save(best_model_state, CONFIG.paths.get_checkpoint_path("multitask_best.pt"))
        print("  -> Saved best model")

print("Training complete.")
""")

# ---------------------------------------------------------
# 8. Evaluation
# ---------------------------------------------------------
add_markdown("""## 8. Evaluation

Evaluate routing accuracy vs Oracle for each mode.
""")

add_code("""# Load best model
if best_model_state:
    model.load_state_dict(best_model_state)

model.eval()
results = []

# To compute routing accuracy, we need all predictions for a sample.
# We'll iterate the validation set and collect everything into a dataframe.
all_preds = []

with torch.no_grad():
    for batch in val_loader:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        model_id = batch["model_id"].to(DEVICE)
        mode_id = batch["mode_id"].to(DEVICE)
        
        outputs = model(input_ids, attention_mask, model_id, mode_id)
        
        util_hat = outputs["utility_hat"].cpu().numpy()
        task_logits = outputs["task_logits"].cpu().numpy()
        task_preds = np.argmax(task_logits, axis=1)
        
        batch_size = len(util_hat)
        for i in range(batch_size):
            all_preds.append({
                "sample_id": batch["sample_ids"][i],
                "model_id": batch["model_id"][i].item(),
                "mode_id": batch["mode_id"][i].item(),
                "utility_target": batch["utility_target"][i].item(),
                "utility_hat": util_hat[i],
                "task_id": batch["task_id"][i].item(),
                "task_pred": task_preds[i]
            })

df_res = pd.DataFrame(all_preds)

# Task Accuracy
task_acc = (df_res["task_id"] == df_res["task_pred"]).mean()
print(f"Overall Task Accuracy: {task_acc:.2%}")

# Routing Analysis per Mode
id_to_model = {v: k for k, v in model_to_id.items()}
id_to_mode = {v: k for k, v in mode_to_id.items()}

metric_rows = []

for mid in sorted(df_res["mode_id"].unique()):
    mode_name = id_to_mode[mid]
    sub = df_res[df_res["mode_id"] == mid]
    
    mse = np.mean((sub["utility_target"] - sub["utility_hat"])**2)
    
    # Needs grouping by sample_id to do routing
    # This is slow in pandas iteration but fine for evaluation
    
    total = 0
    correct = 0
    oracle_sum = 0
    router_sum = 0
    
    # We need to make sure we have all models for each sample
    # Since we did long-form expansion, we should (unless filtered)
    
    for sid, grp in sub.groupby("sample_id"):
        if len(grp) < 2: continue
        
        # Oracle
        best_row = grp.loc[grp["utility_target"].idxmax()]
        oracle_model = best_row["model_id"]
        oracle_util = best_row["utility_target"]
        
        # Router
        pred_best_row = grp.loc[grp["utility_hat"].idxmax()]
        router_model = pred_best_row["model_id"]
        router_util_actual = pred_best_row["utility_target"] # The ACTUAL utility of the chosen model
        
        total += 1
        if oracle_model == router_model:
            correct += 1
            
        oracle_sum += oracle_util
        router_sum += router_util_actual
        
    acc = correct / total if total > 0 else 0
    gap = (oracle_sum - router_sum) / total if total > 0 else 0
    
    metric_rows.append({
        "mode": mode_name,
        "mse": mse,
        "routing_acc": acc,
        "avg_gap": gap,
        "samples": total
    })

df_metrics = pd.DataFrame(metric_rows)
print(df_metrics)
""")

# ---------------------------------------------------------
# 9. Qualitative Inspection
# ---------------------------------------------------------
add_markdown("""## 9. Qualitative Inspection
""")

add_code("""# Sampling a few validation examples
samples = df_res["sample_id"].unique()
import random
random.shuffle(samples)

print("--- Qualitative Checks ---")
for sid in samples[:3]:
    sub = df_res[df_res["sample_id"] == sid]
    
    # Metadata
    # We need to look up text back from df_profiles or df_long if we want prompt text
    # Assuming df_long is available
    meta = df_long[df_long["sample_id"] == sid].iloc[0]
    prompt = meta.get("prompt_raw", "N/A")[:100] + "..."
    task_true = id_to_mode.get(sub.iloc[0]["task_id"], "Unknown") # careful, this is task_id not mode_id
    task_pred = id_to_mode.get(sub.iloc[0]["task_pred"], "Unknown") # wait, task_to_id mapping needed
    
    id_to_task = {v: k for k, v in task_to_id.items()}
    t_true = id_to_task.get(sub.iloc[0]["task_id"], "Unknown")
    t_pred = id_to_task.get(sub.iloc[0]["task_pred"], "Unknown")
    
    print(f"\\nSample: {sid}")
    print(f"Prompt: {prompt}")
    print(f"Task: True={t_true}, Pred={t_pred}")
    
    # Show predictions for 'balanced' mode (or other)
    # Check if we have balanced mode data
    bal_id = mode_to_id.get("balanced")
    if bal_id is not None:
        rows = sub[sub["mode_id"] == bal_id]
        if not rows.empty:
            print("Balanced Mode Utilities:")
            for _, r in rows.iterrows():
                mname = id_to_model[r["model_id"]]
                print(f"  {mname}: True={r['utility_target']:.4f}, Pred={r['utility_hat']:.4f}")
""")

# ---------------------------------------------------------
# 10. Save Artifacts
# ---------------------------------------------------------
add_markdown("""## 10. Save Artifacts
""")

add_code("""# Config is already saved or defined.
# We saved index jsons earlier.
# Model is saved.

print("All artifacts saved.")
print(f"Checkpoint: {CONFIG.paths.get_checkpoint_path('multitask_best.pt')}")
""")


# Write file
notebook_json = {
 "cells": cells,
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

with open(notebook_path, "w") as f:
    json.dump(notebook_json, f, indent=1)

print(f"Generated notebook at {notebook_path}")
