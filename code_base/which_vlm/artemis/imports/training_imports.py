
import os, sys
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
import pandas as pd
import torch
from PIL import Image

# adjust this import path to wherever check_data_utils.py lives
from imports.check_data_utils import fetch_cauldron_image


class RouterDataset(Dataset):
    """Dataset for router training."""
    
    def __init__(
        self,
        df: pd.DataFrame,
        image_root: Path,
        image_processor,
        tokenizer,
        max_text_length: int = 256,
        model_names: List[str] = None,
        use_soft_labels: bool = True,
        use_image: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.image_root = image_root          # used by fetch_cauldron_image as cache root
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.use_image = use_image
        self.max_text_length = max_text_length
        self.model_names = model_names or model_names
        self.use_soft_labels = use_soft_labels
        
    def __len__(self):
        return len(self.df)
    
    def _construct_router_text(self, row) -> str:
        """Construct the text input for the router."""
        parts = []
        
        # Task information
        if pd.notna(row.get('router_task')):
            parts.append(f"Task: {row['router_task']}")
        
        # Dataset information
        if pd.notna(row.get('source_dataset')):
            parts.append(f"Dataset: {row['source_dataset']}")
        
        # Question type
        if pd.notna(row.get('txt_question_type')):
            parts.append(f"QType: {row['txt_question_type']}")
        
        # Has multiple choice
        if pd.notna(row.get('txt_has_mc_options')):
            has_mc = "Yes" if row['txt_has_mc_options'] else "No"
            parts.append(f"HasMC: {has_mc}")
        
        # Image dimensions
        if pd.notna(row.get('img_width')) and pd.notna(row.get('img_height')):
            parts.append(f"ImgSize: {int(row['img_width'])}x{int(row['img_height'])}")
        
        # Aspect ratio
        if pd.notna(row.get('img_aspect_ratio')):
            parts.append(f"AR: {row['img_aspect_ratio']:.2f}")
        
        # Prompt length
        if pd.notna(row.get('txt_prompt_length_words')):
            parts.append(f"PromptWords: {int(row['txt_prompt_length_words'])}")
        
        # Add the actual prompt
        if pd.notna(row.get('prompt_raw')):
            parts.append(f"Prompt: {row['prompt_raw']}")
        
        return " | ".join(parts)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # -----------------------------
        # 1. Load / fetch the image
        # -----------------------------
        if self.use_image:
            try:
                source_config = row.get("source_config")
                source_index = row.get("source_index")
                image_hash   = row.get("image_bytes_hash", None)

                if pd.isna(source_config) or pd.isna(source_index):
                    # Missing metadata → fallback
                    raise ValueError("Missing source_config/source_index for Cauldron fetch")

                # fetch_cauldron_image will:
                #  - try local cache at image_root / source_config / {hash}.png
                #  - otherwise stream from HuggingFace Cauldron
                img, _sample = fetch_cauldron_image(
                    source_config=str(source_config),
                    source_index=int(source_index),
                    image_hash=image_hash if pd.notna(image_hash) else None,
                    prefer_local_cache=True,
                    image_root=self.image_root,
                )
                image = img.convert("RGB")
            except Exception as e:
                print(f"[RouterDataset] Error fetching image for idx={idx}: {e}")
                image = Image.new("RGB", (224, 224), color="gray")
            
            pixel_values = self.image_processor(
                images=image,
                return_tensors="pt",
            )["pixel_values"].squeeze(0)
        else:
            # Image is not used by the model
            pixel_values = torch.zeros(3, 224, 224, dtype=torch.float32)
        
        # -----------------------------
        # 2. Construct and tokenize text
        # -----------------------------
        router_text = self._construct_router_text(row)
        text_encoding = self.tokenizer(
            router_text,
            max_length=self.max_text_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # -----------------------------
        # 3. Hard label
        # -----------------------------
        hard_label = row["router_best_model_id"]
        
        # -----------------------------
        # 4. Soft labels
        # -----------------------------
        soft_labels = torch.zeros(len(self.model_names), dtype=torch.float32)
        if self.use_soft_labels:
            for i, model_name in enumerate(self.model_names):
                col = f"router_soft_p_{model_name}"
                if col in row.index and pd.notna(row[col]):
                    soft_labels[i] = row[col]
            # Normalize to ensure sum=1
            if soft_labels.sum() > 0:
                soft_labels = soft_labels / soft_labels.sum()
            else:
                # Fallback to one-hot if soft labels missing
                soft_labels[hard_label] = 1.0
        else:
            soft_labels[hard_label] = 1.0
        
        return {
            "pixel_values": pixel_values,
            "input_ids": text_encoding["input_ids"].squeeze(0),
            "attention_mask": text_encoding["attention_mask"].squeeze(0),
            "hard_label": torch.tensor(hard_label, dtype=torch.long),
            "soft_labels": soft_labels,
            "sample_id": row.get("sample_id", f"sample_{idx}"),
        }

print("RouterDataset class defined (with Cauldron image fetch).")
