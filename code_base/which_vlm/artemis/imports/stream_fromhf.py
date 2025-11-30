# Stream Cauldron + PixMo from Hugging Face

# Fetch a single sample directly from Hugging Face using `streaming=True`, without downloading the full dataset. Configure repos/configs with environment variables if needed.
import os
import io
import random
import itertools
import requests
from datasets import load_dataset
from PIL import Image
from IPython.display import display

CAULDRON_REPO = os.environ.get("CAULDRON_REPO", "HuggingFaceM4/the_cauldron")
CAULDRON_CONFIG = os.environ.get("CAULDRON_CONFIG", "ai2d")

PIXMO_REPO = os.environ.get("PIXMO_REPO", "allenai/pixmo-cap")  # override if your PixMo repo differs
PIXMO_CONFIG = os.environ.get("PIXMO_CONFIG", None)  # e.g., "default" or None


def stream_sample(repo: str, config: str | None, split: str = "train", index: int = 0):
    """Return one streaming sample from HF without full download."""
    args = [repo] if config is None else [repo, config]
    ds = load_dataset(*args, streaming=True)
    print(f"Dataset splits: {ds.keys()}")
    iterator = iter(ds[split])
    sample = next(itertools.islice(iterator, index, None))
    return sample


def sample_to_image(sample):
    if isinstance(sample, dict):
        img = None
        if "image" in sample:
            img = sample["image"]
        elif "images" in sample and sample["images"]:
            imgs = sample["images"]
            img = imgs[0] if isinstance(imgs, (list, tuple)) else None
        elif "image_url" in sample:
            url = sample["image_url"]
            try:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception as e:
                print(f"Failed to fetch image_url: {e}")
                return None
        if hasattr(img, "convert"):
            return img.convert("RGB")
        if isinstance(img, (bytes, bytearray)):
            return Image.open(io.BytesIO(img)).convert("RGB")
    return None


def show_sample(label: str, sample):
    print(f"\n--- {label} ---")
    if isinstance(sample, dict):
        keys = [k for k in ["router_task", "task", "question", "prompt", "answer", "ground_truth"] if k in sample]
        print({k: sample[k] for k in keys})
    img = sample_to_image(sample)
    if img is not None:
        display(img)

