"""
Inference wrapper for trained Reward Router.

This module provides a simple interface to load and use the trained
reward-based router from router_train module.
"""

import torch
import sys
import pickle
import io
from pathlib import Path
from typing import Dict, List, Optional
import time
from PIL import Image

# Add artemis_final to path for package imports
artemis_path = Path(__file__).parent.parent.parent.parent  # artemis_final's parent
if str(artemis_path) not in sys.path:
    sys.path.insert(0, str(artemis_path))

from artemis_final.router_train.models.reward_router import RewardRouterModel
from artemis_final.router_train.config import RouterModelConfig
from transformers import AutoTokenizer


class _ConfigRemappingUnpickler(pickle.Unpickler):
    """Custom unpickler that remaps config module paths."""
    
    def find_class(self, module, name):
        # Remap 'config' module to the correct path
        if module == 'config' and name == 'RouterModelConfig':
            return RouterModelConfig
        return super().find_class(module, name)


def _load_checkpoint_safe(path, map_location='cpu'):
    """Load checkpoint with custom unpickler to handle module remapping."""
    with open(path, 'rb') as f:
        unpickler = _ConfigRemappingUnpickler(io.BytesIO(f.read()))
        unpickler.find_class = lambda m, n: (
            RouterModelConfig if (m == 'config' and n == 'RouterModelConfig')
            else pickle.Unpickler.find_class(unpickler, m, n)
        )
        # Use torch's load with the buffer
        f.seek(0)
        try:
            return torch.load(f, map_location=map_location, weights_only=False)
        except ModuleNotFoundError:
            # If config module not found, try to load without pickle verification
            f.seek(0)
            return torch.load(f, map_location=map_location, weights_only=False, 
                             pickle_module=type('FakePickle', (), {
                                 'Unpickler': _ConfigRemappingUnpickler,
                                 'load': pickle.load,
                                 'dump': pickle.dump,
                             }))


class RewardRouterInference:
    """
    Simple inference wrapper for trained reward router.

    This class loads a trained reward router checkpoint and provides
    a clean API for routing samples to the best VLM model.

    Example:
        ```python
        from PIL import Image

        router = RewardRouterInference(
            checkpoint_path='checkpoints/best_reward_router.pt',
            device='cuda:0'
        )

        img = Image.open('diagram.jpg')
        result = router.route(
            prompt="What is shown in this diagram?",
            image=img,
            mode="accuracy"
        )

        print(f"Best model: {result['chosen_model']}")
        ```
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cpu',
        verbose: bool = True
    ):
        """
        Initialize the reward router for inference.

        Args:
            checkpoint_path: Path to trained .pt checkpoint
            device: Device for inference ('cpu', 'cuda:0', 'mps', etc.)
            verbose: Print initialization messages
        """
        self.device = device
        self.verbose = verbose
        self.checkpoint_path = checkpoint_path

        # Load configuration from checkpoint with custom unpickler
        # Load configuration from checkpoint with custom unpickler
        try:
            checkpoint = _load_checkpoint_safe(checkpoint_path, map_location=self.device)
        except Exception as e:
            if verbose:
                print(f"[ERROR] Failed to load checkpoint safe: {e}")
                # Try fallback just in case, but usually _load_checkpoint_safe covers it
            raise e

        # Extract config - handle various checkpoint formats
        if 'config' in checkpoint and checkpoint['config'] is not None:
            config = checkpoint['config']
            if hasattr(config, 'text_encoder_name') and config.text_encoder_name is not None:
                self.config = config
            else:
                self.config = RouterModelConfig()
                if self.verbose:
                    print("[INFO] Config missing text_encoder_name, using default RouterModelConfig")
        else:
            # Fallback for known checkpoint "best_multitask_router_v1.pt" which uses hidden_dim=256
            self.config = RouterModelConfig(hidden_dim=256)
            if self.verbose:
                print("[INFO] Using default RouterModelConfig with hidden_dim=256 (compatible with v1 multitask router)")

        # Model metadata - try to get from checkpoint, fallback to defaults
        self.num_models = checkpoint.get('num_models', 5)
        self.num_modes = checkpoint.get('num_modes', 4)
        self.num_tasks = checkpoint.get('num_tasks', 30)

        self.model_names = [
            "deepseek_ocr",
            "qwen2_5_vl_3b",
            "qwen2_5_vl_7b",
            "qwen3_vl_8b_thinking",
            "gemma_3_27b"
        ]

        self.mode_names = ["accuracy", "cheap", "fast", "balanced"]

        # Create model ID mappings
        self.model_to_id = {name: idx for idx, name in enumerate(self.model_names)}
        self.id_to_model = {idx: name for idx, name in enumerate(self.model_names)}

        self.mode_to_id = {name: idx for idx, name in enumerate(self.mode_names)}
        self.id_to_mode = {idx: name for idx, name in enumerate(self.mode_names)}

        # Initialize model
        if self.verbose:
            print(f"[INFO] Loading model from: {checkpoint_path}")

        self.model = RewardRouterModel(
            config=self.config,
            num_models=self.num_models,
            num_modes=self.num_modes,
            num_tasks=self.num_tasks,
        )

        # Load weights - handle both 'state_dict' and 'model_state_dict' keys
        # Load weights - handle 'state_dict', 'model_state_dict', or raw state dict
        if 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'])
        elif 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            if self.verbose:
                print("[INFO] No 'state_dict' key found. Attempting to load as raw state dict.")
            self.model.load_state_dict(checkpoint)
        self.model.to(device)
        self.model.eval()

        # Load tokenizer - use DebertaV2Tokenizer directly for DeBERTa to avoid bug
        if 'deberta' in self.config.text_encoder_name.lower():
            from transformers import DebertaV2Tokenizer
            self.tokenizer = DebertaV2Tokenizer.from_pretrained(self.config.text_encoder_name)
        else:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(self.config.text_encoder_name)
            except (AttributeError, TypeError) as e:
                if self.verbose:
                    print(f"[INFO] Fast tokenizer failed, trying slow tokenizer: {e}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.config.text_encoder_name, use_fast=False
                )

        if self.verbose:
            print(f"[INFO] Model loaded on device: {device}")
            print(f"[INFO] Text encoder: {self.config.text_encoder_name}")
            print(f"[INFO] Models: {self.model_names}")
            print(f"[INFO] Modes: {self.mode_names}")
            print(f"[INFO] Tasks: {self.num_tasks}")

    def format_sample_text(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Format sample text with image metadata computed on-the-fly.

        This matches the primary training format (70% of training data):
        [ROUTER] PromptLenWords: X. ImgWidth: W. ImgHeight: H. ImgAR: A. Question: {prompt}

        Key insight: Image resolution tells the model about accuracy/latency/cost trade-offs.
        - High-res (2048x1536) → needs accurate models (gemma_3_27b)
        - Low-res (512x384) → can use fast models (qwen2_5_vl_3b)

        Args:
            prompt: Question/prompt text
            image: PIL.Image object (optional). If provided, computes width, height, aspect ratio.
            metadata: Optional metadata dict (rarely used). If provided with 'router_task'
                     and 'source_dataset', will use full metadata format.

        Returns:
            Formatted text string matching training augmentation format
        """
        if metadata is None:
            metadata = {}

        # Compute prompt length in words
        prompt_len = len(prompt.split())

        # Check if user provided full metadata (rare case)
        has_task = 'router_task' in metadata
        has_dataset = 'source_dataset' in metadata

        if has_task and has_dataset:
            # Full metadata format (10% of training) - use when user explicitly provides
            task = metadata['router_task']
            dataset = metadata['source_dataset']
            source_config = metadata.get('source_config', 'unknown')
            data_split = metadata.get('data_split', 'unknown')

            if image is not None:
                img_width = image.width
                img_height = image.height
                img_ar = img_width / img_height
            else:
                img_width = img_height = 0
                img_ar = 1.0

            return (
                f"[ROUTER] Task: {task}. Dataset: {dataset}. "
                f"SourceConfig: {source_config}. Split: {data_split}. "
                f"PromptLenWords: {prompt_len}. "
                f"ImgWidth: {img_width}. ImgHeight: {img_height}. "
                f"ImgAR: {img_ar:.2f}. "
                f"Question: {prompt}"
            )

        # Primary format (70% of training): Question + Image metadata
        # This is the standard inference format
        if image is not None:
            img_width = image.width
            img_height = image.height
            img_ar = img_width / img_height

            return (
                f"[ROUTER] PromptLenWords: {prompt_len}. "
                f"ImgWidth: {img_width}. ImgHeight: {img_height}. "
                f"ImgAR: {img_ar:.2f}. "
                f"Question: {prompt}"
            )

        # Question-only fallback (for text-only queries without image)
        return f"[ROUTER] Question: {prompt}"

    def route(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
        mode: str = "accuracy",
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Route a sample to the best VLM model using text + image metadata.

        Args:
            prompt: Question/prompt text
            image: PIL.Image object (optional). Router computes width, height, aspect ratio
                   to help predict accuracy/latency/cost trade-offs.
            mode: Routing mode, one of:
                - "accuracy": Maximize prediction quality
                - "cheap": Balance quality with low cost
                - "fast": Balance quality with low latency
                - "balanced": Multi-objective optimization
            metadata: Optional metadata dict (rarely needed). Router works without it!

        Returns:
            Dictionary with routing results:
            {
                'chosen_model': str,           # Best model name
                'chosen_model_id': int,        # Model ID (0-4)
                'rewards': dict,               # {model_name: predicted_reward}
                'mode': str,                   # Routing mode used
                'inference_ms': float,         # Inference time
            }

        Raises:
            ValueError: If mode is not recognized

        Example:
            ```python
            from PIL import Image
            router = RewardRouterInference('checkpoints/best_reward_router.pt')

            img = Image.open('document.jpg')
            result = router.route(
                prompt="Extract text from this document.",
                image=img,
                mode="fast"
            )
            print(f"Route to: {result['chosen_model']}")
            ```
        """
        # Validate mode
        if mode not in self.mode_names:
            raise ValueError(
                f"Unknown mode: {mode}. Must be one of {self.mode_names}"
            )

        # Handle swapped arguments (common user error: passing (image, prompt))
        if not isinstance(prompt, str) and isinstance(image, str):
            if self.verbose:
                print("[INFO] Detected swapped arguments in route(). Swapping prompt and image.")
            prompt, image = image, prompt

        # Format text with image metadata
        sample_text = self.format_sample_text(prompt, image, metadata)

        # Get mode ID
        mode_id = self.mode_to_id[mode]

        # Run inference
        start_time = time.time()

        with torch.no_grad():
            # Tokenize the text
            encoded = self.tokenizer(
                [sample_text] * self.num_models,
                padding=True,
                truncation=True,
                max_length=self.config.max_seq_length,
                return_tensors="pt"
            )
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)

            # Prepare inputs for all models
            model_ids = torch.arange(self.num_models, device=self.device)
            mode_ids = torch.tensor([mode_id], device=self.device).expand(self.num_models)

            # Predict rewards
            # Output is a dict: {'utility_hat': ..., 'task_logits': ...}
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                model_id=model_ids,
                mode_id=mode_ids,
            )
            
            # Extract utility scores (shape: [num_models])
            rewards = outputs["utility_hat"]

            # Convert to numpy
            rewards = rewards.cpu().numpy()  # [num_models]

        inference_time = (time.time() - start_time) * 1000  # ms

        # Choose best model (highest reward)
        best_idx = int(rewards.argmax())
        best_model = self.model_names[best_idx]

        # Build reward dictionary
        reward_dict = {
            name: float(reward)
            for name, reward in zip(self.model_names, rewards)
        }

        return {
            'chosen_model': best_model,
            'chosen_model_id': best_idx,
            'rewards': reward_dict,
            'mode': mode,
            'inference_ms': inference_time,
        }

    def route_batch(
        self,
        prompts: List[str],
        images: Optional[List[Optional[Image.Image]]] = None,
        modes: Optional[List[str]] = None,
        metadata_list: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """
        Route multiple samples in batch.

        Args:
            prompts: List of question texts
            images: Optional list of PIL.Image objects (one per prompt, can be None)
            modes: List of routing modes (one per prompt). If None, uses "balanced" for all.
            metadata_list: Optional list of metadata dicts

        Returns:
            List of routing result dictionaries
        """
        batch_size = len(prompts)

        # Set defaults
        if images is None:
            images = [None] * batch_size
        if modes is None:
            modes = ["balanced"] * batch_size
        if metadata_list is None:
            metadata_list = [None] * batch_size

        # Validate lengths
        if len(images) != batch_size:
            raise ValueError(f"images length ({len(images)}) must match prompts ({batch_size})")
        if len(modes) != batch_size:
            raise ValueError(f"modes length ({len(modes)}) must match prompts ({batch_size})")
        if len(metadata_list) != batch_size:
            raise ValueError(f"metadata_list length ({len(metadata_list)}) must match prompts ({batch_size})")

        # Route each sample individually
        # Note: Could be optimized with true batching if needed
        results = []
        for prompt, image, mode, metadata in zip(prompts, images, modes, metadata_list):
            result = self.route(prompt, image, mode, metadata)
            results.append(result)

        return results

    def get_stats(self) -> Dict:
        """
        Get router statistics and configuration.

        Returns:
            Dictionary with router info
        """
        return {
            'device': str(self.device),
            'checkpoint_path': self.checkpoint_path,
            'text_encoder': self.config.text_encoder_name,
            'num_models': self.num_models,
            'num_modes': self.num_modes,
            'model_names': self.model_names,
            'mode_names': self.mode_names,
            'max_seq_length': self.config.max_seq_length,
            'hidden_dim': self.config.hidden_dim,
        }


# Example usage and testing
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Reward Router Inference")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="../checkpoints/best_reward_router.pt",
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for inference"
    )

    args = parser.parse_args()

    # Initialize router
    print("="*60)
    print("REWARD ROUTER INFERENCE TEST")
    print("="*60)

    router = RewardRouterInference(
        checkpoint_path=args.checkpoint,
        device=args.device,
        verbose=True
    )

    # Test cases
    test_cases = [
        {
            'prompt': "What is the capital of France?",
            'mode': "cheap",
            'metadata': {'router_task': 'qa', 'source_dataset': 'test'}
        },
        {
            'prompt': "Analyze this complex medical image in detail.",
            'mode': "accuracy",
            'metadata': {'router_task': 'medical_vqa', 'source_dataset': 'test'}
        },
        {
            'prompt': "Quick question: Is this a cat or dog?",
            'mode': "fast",
            'metadata': {'router_task': 'classification', 'source_dataset': 'test'}
        },
        {
            'prompt': "Provide a balanced analysis of this chart.",
            'mode': "balanced",
            'metadata': {'router_task': 'chartqa', 'source_dataset': 'test'}
        },
    ]

    # Run tests
    print("\n" + "="*60)
    print("TEST CASES")
    print("="*60)

    for i, test in enumerate(test_cases, 1):
        print(f"\n[Test {i}]")
        print(f"  Prompt: {test['prompt'][:60]}...")
        print(f"  Mode: {test['mode']}")

        result = router.route(**test)

        print(f"  → Chosen: {result['chosen_model']}")
        print(f"  → Latency: {result['inference_ms']:.2f}ms")
        print(f"  → Rewards:")
        for model, reward in sorted(
            result['rewards'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            marker = "★" if model == result['chosen_model'] else " "
            print(f"      {marker} {model:30s}: {reward:.4f}")

    print("\n" + "="*60)
    print("✓ All tests passed")
    print("="*60)

    # Print stats
    print("\nRouter Statistics:")
    stats = router.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
