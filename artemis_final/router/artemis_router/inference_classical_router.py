"""
Inference wrapper for trained Classical Router (CE+KL).

This module provides a simple interface to load and use the trained
classical classification-based router.
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

from models.classical_router import ClassicalRouterModel


class ClassicalRouterInference:
    """
    Inference wrapper for trained ClassicalRouterModel.

    This class loads a trained classical router checkpoint and provides
    a clean API for routing samples to the best VLM model using
    softmax probabilities over model classes.

    Example:
        ```python
        router = ClassicalRouterInference(
            checkpoint_path='checkpoints/best_classical_router.pt',
            device='cuda:0'
        )

        result = router.route(
            prompt="What is in this image?",
            mode="accuracy",
            metadata={'router_task': 'vqa', 'source_dataset': 'test'}
        )

        print(f"Best model: {result['chosen_model']}")
        print(f"Probabilities: {result['probs']}")
        ```
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cpu',
        verbose: bool = True
    ):
        """
        Initialize the classical router for inference.

        Args:
            checkpoint_path: Path to trained .pt checkpoint
            device: Device for inference ('cpu', 'cuda:0', 'mps', etc.)
            verbose: Print initialization messages
        """
        self.device = device
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

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
            print(f"[INFO] Loading ClassicalRouter from: {checkpoint_path}")

        # Initialize model
        self.model = ClassicalRouterModel(
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

        Args:
            prompt: Question/prompt text
            mode: Routing mode, one of: "accuracy", "cheap", "fast", "balanced"
            metadata: Optional metadata dict

        Returns:
            Dictionary with routing results:
            {
                'chosen_model': str,
                'chosen_model_id': int,
                'probs': dict,        # {model_name: probability}
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
            mode_ids = torch.tensor([mode_id], device=self.device)
            logits = self.model([sample_text], mode_ids)  # [1, num_models]
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        inference_time = (time.time() - start_time) * 1000

        best_idx = int(probs.argmax())
        best_model = self.id_to_model[best_idx]

        prob_dict = {
            self.id_to_model[i]: float(probs[i])
            for i in range(self.num_models)
        }

        return {
            'chosen_model': best_model,
            'chosen_model_id': best_idx,
            'probs': prob_dict,
            'mode': mode,
            'inference_ms': inference_time,
        }

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
            'router_type': 'classical',
            'device': str(self.device),
            'checkpoint_path': self.checkpoint_path,
            'num_models': self.num_models,
            'num_modes': self.num_modes,
            'model_names': self.model_names,
            'mode_names': self.mode_names,
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Classical Router Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../checkpoints/best_classical_router.pt",
        help="Path to checkpoint file"
    )
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    print("=" * 60)
    print("CLASSICAL ROUTER INFERENCE TEST")
    print("=" * 60)

    router = ClassicalRouterInference(
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
        print(f"  → Probabilities:")
        for model, prob in sorted(result['probs'].items(), key=lambda x: x[1], reverse=True):
            marker = "★" if model == result['chosen_model'] else " "
            print(f"      {marker} {model:30s}: {prob:.4f}")

    print("\n" + "=" * 60)
    print("✓ All tests passed")
    print("=" * 60)
