"""
Feature extraction for router inference.

Converts Sample objects into tensors suitable for the MultimodalRouter.
"""

import torch
import numpy as np
from typing import List, Dict
from PIL import Image
from transformers import AutoTokenizer, CLIPImageProcessor

from .schemas import Sample
from .config import FeatureConfig


class FeatureExtractor:
    """
    Extracts and processes features from Sample objects for router inference.

    Handles:
    - Text tokenization with metadata injection
    - Image preprocessing
    - Text-only, image-only, and multimodal samples
    - Batching and padding
    """

    def __init__(self, cfg: FeatureConfig, device: str, router_config):
        """
        Initialize feature extractor.

        Args:
            cfg: Feature configuration
            device: Target device for tensors
            router_config: Router config (for encoder names)
        """
        self.cfg = cfg
        self.device = device
        self.router_config = router_config

        # Load tokenizer
        tokenizer_name = getattr(router_config, 'text_tokenizer_name', 'distilbert-base-uncased')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Load image processor
        vision_encoder_name = getattr(router_config, 'vision_encoder_name', 'openai/clip-vit-base-patch32')
        self.image_processor = CLIPImageProcessor.from_pretrained(vision_encoder_name)

    def _build_router_text(self, sample: Sample) -> str:
        """
        Build router input text by combining metadata and question.

        This matches the format used during training (see 09_inference_router_analysis.ipynb).

        Args:
            sample: Input sample

        Returns:
            Formatted text string
        """
        meta_parts = []

        # Extract metadata
        meta = sample.metadata

        # Text length metadata
        text_len_words = len(sample.text.split())
        text_len_chars = len(sample.text)
        meta_parts.append(f"PromptLenWords: {text_len_words}.")
        meta_parts.append(f"PromptLenChars: {text_len_chars}.")

        # Image metadata (if available)
        if sample.image is not None:
            w, h = sample.image.size
            ar = w / h if h > 0 else 1.0
            meta_parts.append(f"ImageWidth: {w}. ImageHeight: {h}.")
            meta_parts.append(f"ImageAR: {ar:.2f}.")
        elif 'img_width' in meta and 'img_height' in meta:
            w = meta['img_width']
            h = meta['img_height']
            ar = meta.get('img_aspect_ratio', w / h if h > 0 else 1.0)
            meta_parts.append(f"ImageWidth: {int(w)}. ImageHeight: {int(h)}.")
            meta_parts.append(f"ImageAR: {float(ar):.2f}.")

        meta_str = " ".join(meta_parts)
        router_text = f"{meta_str} Question: {sample.text}"
        return router_text

    def _process_image(self, image: Image.Image) -> torch.Tensor:
        """
        Process PIL Image to tensor.

        Args:
            image: PIL Image

        Returns:
            Pixel values tensor [C, H, W]
        """
        inputs = self.image_processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)
        return pixel_values

    def _get_blank_image(self) -> torch.Tensor:
        """
        Create a blank/zero image tensor for text-only samples.

        Returns:
            Zero tensor [C, H, W]
        """
        return torch.zeros(3, self.cfg.image_size, self.cfg.image_size)

    def encode_single(self, sample: Sample) -> Dict[str, torch.Tensor]:
        """
        Encode a single sample into router input tensors.

        Args:
            sample: Input sample

        Returns:
            Dictionary with keys:
            - pixel_values: [1, C, H, W]
            - input_ids: [1, T]
            - attention_mask: [1, T]
        """
        # Process image
        if sample.image is not None and self.cfg.allow_image_only:
            pixel_values = self._process_image(sample.image)
        elif self.cfg.allow_text_only:
            pixel_values = self._get_blank_image()
        else:
            raise ValueError(f"Sample {sample.sample_id} has no image and text-only is not allowed")

        # Process text
        if sample.text:
            router_text = self._build_router_text(sample)
        elif self.cfg.allow_image_only:
            router_text = ""
        else:
            raise ValueError(f"Sample {sample.sample_id} has no text and image-only is not allowed")

        # Tokenize
        encoding = self.tokenizer(
            router_text,
            padding="max_length",
            truncation=True,
            max_length=self.cfg.max_text_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values.unsqueeze(0).to(self.device),
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device),
        }

    def encode_batch(self, samples: List[Sample]) -> Dict[str, torch.Tensor]:
        """
        Encode multiple samples into batched tensors.

        Args:
            samples: List of samples

        Returns:
            Dictionary with keys:
            - pixel_values: [B, C, H, W]
            - input_ids: [B, T]
            - attention_mask: [B, T]
        """
        pixel_values_list = []
        texts = []

        for sample in samples:
            # Process image
            if sample.image is not None:
                pixel_values = self._process_image(sample.image)
            else:
                pixel_values = self._get_blank_image()
            pixel_values_list.append(pixel_values)

            # Build text
            if sample.text:
                router_text = self._build_router_text(sample)
            else:
                router_text = ""
            texts.append(router_text)

        # Stack images
        pixel_values_batch = torch.stack(pixel_values_list, dim=0)

        # Batch tokenize
        encodings = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.cfg.max_text_length,
            return_tensors="pt",
        )

        return {
            "pixel_values": pixel_values_batch.to(self.device),
            "input_ids": encodings["input_ids"].to(self.device),
            "attention_mask": encodings["attention_mask"].to(self.device),
        }
