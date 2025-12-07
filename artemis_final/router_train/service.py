"""
Retrainer: Service for continuous retraining of the router model.
Uses collected data from DataCollector to fine-tune the reward router.
"""
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import torch
from transformers import AutoTokenizer

from common.config_loader import GlobalConfig
from data_loop.collector import DataCollector

logger = logging.getLogger(__name__)

class Retrainer:
    """
    Handles retraining of the router model using collected data.
    Integrates with existing router_train code.
    """
    
    def __init__(self, cfg: GlobalConfig, data_collector: DataCollector):
        """
        Initialize the Retrainer.
        
        Args:
            cfg: GlobalConfig from common.config_loader
            data_collector: DataCollector instance for fetching training data
        """
        self.cfg = cfg
        self.collector = data_collector
        self.base_dir = Path(cfg._base_dir)
        self.output_checkpoint = self.base_dir / cfg.retraining.output_checkpoint
        self.epochs = cfg.retraining.epochs
        self.batch_size = cfg.retraining.batch_size
        
    def build_dataset(self) -> Optional[pd.DataFrame]:
        """
        Build a training dataset from collected data.
        
        Returns:
            DataFrame suitable for router training, or None if insufficient data.
        """
        logger.info("Fetching collected data for training...")
        raw_data = self.collector.fetch_training_data(limit=1000)
        
        if len(raw_data) < 10:
            logger.warning(f"Insufficient data for training: {len(raw_data)} samples")
            return None
        
        logger.info(f"Building dataset from {len(raw_data)} samples")
        
        # Transform to router training format
        # Expected columns: sample_id, model_id, mode_id, reward, prompt_raw, + metadata
        records = []
        for row in raw_data:
            # Extract feedback score as reward signal
            feedback = row.get("feedback_params", {})
            if isinstance(feedback, str):
                import json
                feedback = json.loads(feedback)
            
            score = feedback.get("score", 3.0)  # Default neutral score
            reward = score / 5.0  # Normalize to 0-1
            
            # Extract prompt from input_messages
            messages = row.get("input_messages", [])
            if isinstance(messages, str):
                import json
                messages = json.loads(messages)
            
            prompt = ""
            for m in messages:
                if m.get("role") == "user":
                    prompt += str(m.get("content", "")) + " "
            prompt = prompt.strip()
            
            if not prompt:
                continue
            
            records.append({
                "sample_id": row["request_id"],
                "model_name": row.get("chosen_model", "unknown"),
                "mode_name": row.get("router_mode", "balanced"),
                "reward": reward,
                "prompt_raw": prompt,
                # Metadata for _build_input_text
                "router_task": "vlm_routing",
                "source_dataset": "collected",
                "source_config": "default",
                "data_split": "train",
                "txt_prompt_length_words": len(prompt.split()),
                "img_width": 0,
                "img_height": 0,
                "img_aspect_ratio": 0.0
            })
        
        if not records:
            return None
        
        df = pd.DataFrame(records)
        
        # Map model/mode names to IDs
        # Load from router config
        model_order = ["deepseek_ocr", "qwen2_5_vl_3b", "qwen2_5_vl_7b", "qwen3_vl_8b_thinking", "gemma_3_27b"]
        mode_order = ["accuracy", "cheap", "fast", "balanced"]
        
        model_map = {name: i for i, name in enumerate(model_order)}
        mode_map = {name: i for i, name in enumerate(mode_order)}
        
        df["model_id"] = df["model_name"].map(model_map).fillna(0).astype(int)
        df["mode_id"] = df["mode_name"].map(mode_map).fillna(3).astype(int)  # default balanced
        
        return df

    def retrain_once(self) -> Optional[str]:
        """
        Run one retraining cycle.
        
        Returns:
            Path to new checkpoint, or None if training was skipped.
        """
        df = self.build_dataset()
        if df is None:
            logger.info("Skipping retraining - insufficient data")
            return None
        
        logger.info(f"Starting retraining with {len(df)} samples, {self.epochs} epochs")
        
        try:
            # Import router_train components
            from router_train.training.dataset import build_dataloaders
            from router_train.models.reward_router import create_model
            from router_train.training.train_reward_router import RewardRouterTrainer
            from router_train.config import Config as TrainConfig
            
            # Split data
            split_idx = int(len(df) * 0.8)
            train_df = df.iloc[:split_idx].copy()
            val_df = df.iloc[split_idx:].copy()
            test_df = val_df.copy()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            
            # Build dataloaders
            train_loader, val_loader, _ = build_dataloaders(
                train_df, val_df, test_df,
                tokenizer=tokenizer,
                batch_size=self.batch_size,
                max_seq_length=256
            )
            
            # Create or load model
            num_models = 5
            num_modes = 4
            
            # Try to load existing checkpoint for fine-tuning
            existing_checkpoint = self.base_dir / self.cfg.router.checkpoint_path
            
            config = TrainConfig.default()
            model = create_model(config.model, num_models, num_modes)
            
            if existing_checkpoint.exists():
                logger.info(f"Loading existing weights from {existing_checkpoint}")
                try:
                    state = torch.load(existing_checkpoint, map_location="cpu")
                    if isinstance(state, dict) and "state_dict" in state:
                        model.load_state_dict(state["state_dict"])
                    else:
                        model.load_state_dict(state)
                except Exception as e:
                    logger.warning(f"Could not load checkpoint: {e}")
            
            # Create trainer and train
            trainer = RewardRouterTrainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                config=config,
                device=self.cfg.router.device
            )
            
            # Override epochs
            trainer.train_config.num_epochs = self.epochs
            
            # Train
            trainer.train()
            
            # Save new checkpoint
            self.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(self.output_checkpoint))
            
            logger.info(f"Retraining complete. New checkpoint: {self.output_checkpoint}")
            return str(self.output_checkpoint)
            
        except ImportError as e:
            logger.error(f"Missing router_train dependencies: {e}")
            return None
        except Exception as e:
            logger.error(f"Retraining failed: {e}")
            raise e
