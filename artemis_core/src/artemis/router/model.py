import torch
import torch.nn as nn
from transformers import AutoModel
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class RouterModelConfig:
    text_encoder_name: str = "distilbert-base-uncased"
    freeze_text_encoder: bool = True
    model_emb_dim: int = 32
    mode_emb_dim: int = 16
    hidden_dim: int = 512
    dropout: float = 0.1
    max_seq_length: int = 256

class RewardRouterModel(nn.Module):
    def __init__(
        self,
        config: RouterModelConfig,
        num_models: int,
        num_modes: int,
        num_tasks: int,
        text_encoder_hidden_size: Optional[int] = None,
    ):
        super().__init__()
        self.config = config
        self.text_encoder = AutoModel.from_pretrained(config.text_encoder_name)
        
        # Auto-detect hidden size if not provided
        if text_encoder_hidden_size is None:
            if hasattr(self.text_encoder.config, "hidden_size"):
                text_encoder_hidden_size = self.text_encoder.config.hidden_size
            elif hasattr(self.text_encoder.config, "dim"):
                text_encoder_hidden_size = self.text_encoder.config.dim
            else:
                text_encoder_hidden_size = 768 # Default fallback

        if config.freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.model_embedding = nn.Embedding(num_models, config.model_emb_dim)
        self.mode_embedding = nn.Embedding(num_modes, config.mode_emb_dim)

        input_dim = text_encoder_hidden_size + config.model_emb_dim + config.mode_emb_dim
        
        self.routing_mlp = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        self.task_head = nn.Sequential(
            nn.Linear(text_encoder_hidden_size, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, num_tasks)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        model_id: torch.Tensor,
        mode_id: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        if hasattr(outputs, "last_hidden_state"):
            h_text = outputs.last_hidden_state[:, 0, :]
        else:
            h_text = outputs[0][:, 0, :]
        
        task_logits = self.task_head(h_text)
        
        h_model = self.model_embedding(model_id)
        h_mode = self.mode_embedding(mode_id)
        
        h_combined = torch.cat([h_text, h_model, h_mode], dim=-1)
        utility_hat = self.routing_mlp(h_combined).squeeze(-1)
        
        return {
            "utility_hat": utility_hat,
            "task_logits": task_logits
        }
