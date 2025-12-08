"""
Inference wrapper for trained Pairwise Router (margin ranking).

This module provides a simple interface to load and use the trained
pairwise ranking-based router.
"""

import torch
import torch.nn.functional as F
import sys
from pathlib import Path
from typing import Dict, List, Optional
import time

# Add router_train to path
router_train_path = Path(__file__).parent.parent.parent / "router_train"
if str(router_train_path) not in sys.path:
    sys.path.insert(0, str(router_train_path))

from router_train.models.pairwise_router import PairwiseRouterModel


class PairwiseRouterInference:
    """
    Inference wrapper for trained PairwiseRouterModel.

    This class loads a trained pairwise router checkpoint and provides
    a clean API for routing samples. It scores each (sample, model, mode)
    triple and selects the model with the highest score.

    Example:
        ```python
        router = PairwiseRouterInference(
            checkpoint_path='checkpoints/best_pairwise_router.pt',
            device='cuda:0'
        )

        result = router.route(
            prompt="What is in this image?",
            mode="accuracy",
            metadata={'router_task': 'vqa', 'source_dataset': 'test'}
        )

        print(f"Best model: {result['chosen_model']}")
        print(f"Scores: {result['scores']}")
        ```
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cpu',
        verbose: bool = True
    ):
        """
        Initialize the pairwise router for inference.

        Args:
            checkpoint_path: Path to trained .pt checkpoint
            device: Device for inference ('cpu', 'cuda:0', 'mps', etc.)
            verbose: Print initialization messages
        """
        self.device = device
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

        # Get model-to-id and mode-to-id mappings from checkpoint
        self.model_to_id = checkpoint['model_to_id']
        self.mode_to_id = checkpoint['mode_to_id']

        # Create reverse mappings
        self.id_to_model = {v: k for k, v in self.model_to_id.items()}
        self.id_to_mode = {v: k for k, v in self.mode_to_id.items()}

        self.num_models = len(self.model_to_id)
        self.num_modes = len(self.mode_to_id)

        self.model_names = [self.id_to_model[i] for i in range(self.num_models)]
        self.mode_names = [self.id_to_mode[i] for i in range(self.num_modes)]

        if self.verbose:
            print(f"[INFO] Loading PairwiseRouter from: {checkpoint_path}")

        # Initialize model
        self.model = PairwiseRouterModel(
            num_models=self.num_models,
            num_modes=self.num_modes,
        )

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        if self.verbose:
            print(f"[INFO] Model loaded on device: {device}")
            print(f"[INFO] Models: {self.model_names}")
            print(f"[INFO] Modes: {self.mode_names}")

    def format_sample_text(
        self,
        prompt: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Format sample text with metadata (matches training format).

        Args:
            prompt: Question/prompt text
            metadata: Optional metadata dict

        Returns:
            Formatted text string
        """
        if metadata is None:
            metadata = {}

        task = metadata.get('router_task', 'unknown')
        dataset = metadata.get('source_dataset', 'unknown')

        return f"[ROUTER] Task: {task}. Dataset: {dataset}. Question: {prompt}"

    def route(
        self,
        prompt: str,
        mode: str = "accuracy",
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Route a sample to the best VLM model.

        Scores all (sample, model, mode) triples and returns the best.

        Args:
            prompt: Question/prompt text
            mode: Routing mode, one of: "accuracy", "cheap", "fast", "balanced"
            metadata: Optional metadata dict

        Returns:
            Dictionary with routing results:
            {
                'chosen_model': str,
                'chosen_model_id': int,
                'scores': dict,       # {model_name: raw_score}
                'probs': dict,        # {model_name: softmax_probability}
                'mode': str,
                'inference_ms': float,
            }
        """
        if mode not in self.mode_to_id:
            raise ValueError(f"Unknown mode: {mode}. Must be one of {self.mode_names}")

        sample_text = self.format_sample_text(prompt, metadata)
        mode_id = self.mode_to_id[mode]

        start_time = time.time()

        with torch.no_grad():
            # Score all models for this sample
            sample_texts = [sample_text] * self.num_models
            model_ids = torch.arange(self.num_models, device=self.device)
            mode_ids = torch.tensor([mode_id] * self.num_models, device=self.device)

            scores = self.model(sample_texts, model_ids, mode_ids)  # [num_models]
            scores_np = scores.cpu().numpy()

            # Convert to probabilities via softmax
            probs = F.softmax(scores, dim=0).cpu().numpy()

        inference_time = (time.time() - start_time) * 1000

        best_idx = int(scores_np.argmax())
        best_model = self.id_to_model[best_idx]

        score_dict = {
            self.id_to_model[i]: float(scores_np[i])
            for i in range(self.num_models)
        }

        prob_dict = {
            self.id_to_model[i]: float(probs[i])
            for i in range(self.num_models)
        }

        return {
            'chosen_model': best_model,
            'chosen_model_id': best_idx,
            'scores': score_dict,
            'probs': prob_dict,
            'mode': mode,
            'inference_ms': inference_time,
        }

    def rank_models(
        self,
        prompt: str,
        mode: str = "accuracy",
        metadata: Optional[Dict] = None
    ) -> List[str]:
        """
        Rank all models for a given prompt, from best to worst.

        Args:
            prompt: Question/prompt text
            mode: Routing mode
            metadata: Optional metadata dict

        Returns:
            List of model names, sorted from best to worst
        """
        result = self.route(prompt, mode, metadata)
        sorted_models = sorted(
            result['scores'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [model for model, score in sorted_models]

    def route_batch(
        self,
        prompts: List[str],
        modes: List[str],
        metadata_list: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Route multiple samples.

        Args:
            prompts: List of question texts
            modes: List of routing modes
            metadata_list: Optional list of metadata dicts

        Returns:
            List of routing result dictionaries
        """
        if metadata_list is None:
            metadata_list = [None] * len(prompts)

        results = []
        for prompt, mode, metadata in zip(prompts, modes, metadata_list):
            result = self.route(prompt, mode, metadata)
            results.append(result)

        return results

    def get_stats(self) -> Dict:
        """Get router statistics and configuration."""
        return {
            'router_type': 'pairwise',
            'device': str(self.device),
            'checkpoint_path': self.checkpoint_path,
            'num_models': self.num_models,
            'num_modes': self.num_modes,
            'model_names': self.model_names,
            'mode_names': self.mode_names,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Pairwise Router Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../checkpoints/best_pairwise_router.pt",
        help="Path to checkpoint file"
    )
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    print("=" * 60)
    print("PAIRWISE ROUTER INFERENCE TEST")
    print("=" * 60)

    router = PairwiseRouterInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        verbose=True
    )

    test_cases = [
        {"prompt": "What is the capital of France?", "mode": "cheap"},
        {"prompt": "Analyze this complex diagram in detail.", "mode": "accuracy"},
        {"prompt": "Quick OCR on this receipt.", "mode": "fast"},
        {"prompt": "Balanced analysis of this chart.", "mode": "balanced"},
    ]

    print("\n" + "=" * 60)
    print("TEST CASES")
    print("=" * 60)

    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}]")
        print(f"  Prompt: {test['prompt'][:50]}...")
        print(f"  Mode: {test['mode']}")

        result = router.route(**test)

        print(f"  → Chosen: {result['chosen_model']}")
        print(f"  → Latency: {result['inference_ms']:.2f}ms")
        print(f"  → Scores:")
        for model, score in sorted(result['scores'].items(), key=lambda x: x[1], reverse=True):
            marker = "★" if model == result['chosen_model'] else " "
            print(f"      {marker} {model:30s}: {score:.4f}")

        print(f"  → Ranking: {router.rank_models(**test)}")

    print("\n" + "=" * 60)
    print("✓ All tests passed")
    print("=" * 60)
