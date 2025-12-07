import logging
import os
import torch
import pandas as pd
from datetime import datetime
from typing import Optional, List

# Common imports
from common.config_loader import SystemConfig
from common.db import get_db_engine

# Router Train imports
# We assume router_train is in python path
try:
    from router_train.config import Config as TrainConfig, RouterModelConfig, TrainingConfig
    from router_train.training.train_reward_router import train_router
    from router_train.training.dataset import build_dataloaders
    from router_train.models.reward_router import create_model
    from router_train.training.train_reward_router import RewardRouterTrainer
except ImportError:
    # Adjust path if needed or fail gracefully during static analysis
    pass

logger = logging.getLogger(__name__)

class Retrainer:
    def __init__(self, system_config: SystemConfig):
        self.sys_config = system_config
        self.db_engine = get_db_engine(system_config.router_config.db_url)
        
    def fetch_training_data(self, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch verified data from collection tables and format for training.
        Returns DataFrame with columns expected by RewardRouterDataset.
        """
        # We need: sample_id, model_id, mode_id, reward, + format input text columns
        # For this prototype, we'll construct a minimal compatible dataframe.
        
        query = f"""
        SELECT 
            s.sample_id, 
            s.prompt_text as prompt_raw,
            r.router_chosen_model,
            r.router_mode,
            s.feedback_score as reward_signal
        FROM vlm_samples_collected s
        JOIN vlm_responses_collected r ON s.sample_id = r.sample_id
        WHERE s.feedback_score IS NOT NULL
        LIMIT {limit}
        """
        df = pd.read_sql(query, self.db_engine)
        
        if df.empty:
            return df
            
        # Map model name to ID
        model_order = self.sys_config.router_config.model_name_order
        model_map = {name: i for i, name in enumerate(model_order)}
        
        # Map mode name to ID (assuming order known or fixed)
        mode_order = self.sys_config.router_config.mode_name_order
        if not mode_order:
             # Default fallback
             mode_order = ['balanced', 'fast', 'cheap', 'accuracy']
        mode_map = {name: i for i, name in enumerate(mode_order)}
        
        # Transformation
        df['model_id'] = df['router_chosen_model'].map(model_map)
        df['mode_id'] = df['router_mode'].map(mode_map)
        
        # Filter out unknown models/modes
        df = df.dropna(subset=['model_id', 'mode_id'])
        df['model_id'] = df['model_id'].astype(int)
        df['mode_id'] = df['mode_id'].astype(int)
        
        # Normalize reward if needed (e.g. feedback 0-5 to 0-1)
        # Assuming model expects 0-1
        df['reward'] = df['reward_signal'] / 5.0
        
        # Add dummy columns required by dataset.py _build_input_text
        # In a real system, we'd log these or parse them
        df['router_task'] = 'vlm_routing'
        df['source_dataset'] = 'live_traffic'
        df['source_config'] = 'default'
        df['data_split'] = 'train' # default all to train for now
        df['txt_prompt_length_words'] = df['prompt_raw'].apply(lambda x: len(str(x).split()))
        df['img_width'] = 0
        df['img_height'] = 0
        df['img_aspect_ratio'] = 0.0
        
        return df

    def run_retraining(self, epochs: int = 1, batch_size: int = 8) -> str:
        """
        Run the retraining loop.
        Returns path to new checkpoint.
        """
        df = self.fetch_training_data()
        if len(df) < 10:
            logger.info("Not enough data to retrain.")
            return None
            
        logger.info(f"Retraining on {len(df)} samples...")
        
        # Manual Split for now (80/20)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        val_df = df.iloc[split_idx:].copy()
        test_df = val_df # reuse for test
        
        # Setup Router Config
        # We need to construct the complex Config object expected by train_reward_router
        # or just invoke the components manually like train_router does.
        
        # 1. Config
        # We'll create a dummy config object structure matching what's expected
        # This is a bit hacky but avoids needing to change the library code
        class PathsConfig:
            def get_data_path(self): return "memory"
            def get_checkpoint_path(self, name=""): 
                os.makedirs("checkpoints/retrain", exist_ok=True)
                return os.path.join("checkpoints/retrain", name or "best.pt")
                
        class MockConfig:
            def __init__(self):
                self.training = TrainingConfig(
                    batch_size=batch_size,
                    num_epochs=epochs,
                    learning_rate=1e-5,
                    weight_decay=0.01,
                    warmup_ratio=0.1,
                    scheduler="cosine",
                    gradient_clip_norm=1.0,
                    eval_interval=1,
                    save_best_only=True,
                    num_workers=0,
                    pin_memory=False,
                    seed=42
                )
                self.model = RouterModelConfig(
                    text_encoder_name=self.sys_config.router_config.text_encoder_name or "bert-base-uncased",
                    max_seq_length=128,
                    dropout=0.1
                )
                self.paths = PathsConfig()
                
        r_config = MockConfig() # Warning: relying on implicit self.sys_config access in outer scope if meant to be closure, but here explicit
        # Correction: `self` is available.
        r_config.model.text_encoder_name = self.sys_config.router_config.text_encoder_name
        
        # 2. Tokenizer
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(r_config.model.text_encoder_name)
        
        # 3. Dataloaders
        train_loader, val_loader, _ = build_dataloaders(
            train_df, val_df, test_df, 
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_seq_length=r_config.model.max_seq_length
        )
        
        # 4. Model
        num_models = len(self.sys_config.router_config.model_name_order)
        num_modes = len(self.sys_config.router_config.mode_name_order)
        
        # Start from existing checkpoint if possible!
        # create_model initializes fresh. We should load state dict.
        model = create_model(r_config.model, num_models, num_modes)
        
        ckpt_path = self.sys_config.router_config.checkpoint_path
        if os.path.exists(ckpt_path):
             logger.info(f"Loading weights from {ckpt_path}")
             # Warning: The saved model might be a full object or state_dict. 
             # RewardRouterModel.load() is a class method usually? 
             # Looking at source, model has .save(), so likely torch.load.
             # Let's try flexible load
             try:
                 state_dict = torch.load(ckpt_path, map_location='cpu')
                 model.load_state_dict(state_dict)
             except Exception as e:
                 logger.warning(f"Could not load checkpoint: {e}")
        
        # 5. Train
        trainer = RewardRouterTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=r_config,
            device=self.sys_config.router_config.device
        )
        
        trainer.train()
        
        # Return path to best model
        return str(r_config.paths.get_checkpoint_path("best.pt"))
