# from artemis_final.ares.configs.config import CONFIG_TO_TASK, TASK_GT_TYPE
from ares.configs.config import CONFIG_TO_TASK, TASK_GT_TYPE
import re
from datasets import load_dataset, get_dataset_config_names
from typing import List, Dict, Any, Optional




CAULDRON_REPO = "HuggingFaceM4/the_cauldron"


def return_dataset_configs(repo: str) -> List[str]:
    """
    Return available dataset config names for a given dataset repo.
    """
    return get_dataset_config_names(repo)

def load_cauldron_samples(config_name: str, n_samples: int = 32, random_sample: bool = False) -> List[Dict]:
    """
    Load samples from a Cauldron subset using streaming mode.
    
    Parameters
    ----------
    config_name : str
        The Cauldron subset name (e.g., 'ai2d', 'chartqa').
    n_samples : int
        Number of samples to load.
    random_sample : bool
        If True, load more samples and randomly select n_samples from them.
        If False (default), take the first n_samples in order.
    
    Returns a list of dicts with keys: 'images', 'texts'
    """
    import random
    
    ds = load_dataset(
        CAULDRON_REPO,
        config_name,
        streaming=True,
        # trust_remote_code=True
    )
    
    if random_sample:
        # Load more samples than needed for random selection
        # Use a buffer multiplier to ensure we have enough variety
        buffer_size = min(n_samples * 5, 1000)
        all_samples = list(ds['train'].take(buffer_size))
        
        if len(all_samples) <= n_samples:
            samples = all_samples
        else:
            samples = random.sample(all_samples, n_samples)
        print(f"Randomly sampled {len(samples)} from {len(all_samples)} samples from '{config_name}'")
    else:
        samples = list(ds['train'].take(n_samples))
        print(f"Loaded {len(samples)} samples from '{config_name}' (sequential)")
    
    return samples


def extract_qa_from_sample(sample: Dict) -> Dict[str, Any]:
    """
    Extract image, user prompt, and ground truth from a Cauldron sample.
    
    Cauldron format:
        sample['images'] = [PIL.Image, ...]
        sample['texts'] = [{'user': '...', 'assistant': '...', 'source': '...'}, ...]
    
    Returns:
        {
            'image': PIL.Image,
            'prompt': str,
            'ground_truth': str,
            'source': str
        }
    """
    if not sample.get('images') or not sample.get('texts'):
        return None
    
    image = sample['images'][0]
    text_turn = sample['texts'][0]
    
    return {
        'image': image,
        'prompt': text_turn['user'],
        'ground_truth': text_turn['assistant'],
        'source': text_turn.get('source', 'unknown')
    }


def extract_answer_letter(text: str) -> Optional[str]:
    """
    Extract answer letter (A, B, C, D) from multiple choice responses.
    Handles formats like 'Answer: A', '(A)', 'A.', 'A)', etc.
    """
    patterns = [
        r"Answer:\s*([A-D])",
        r"\(([A-D])\)",
        r"^([A-D])[\.\)]",
        r"^([A-D])$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.strip(), re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None



# =============================================================================
# DATASET LOADER
# =============================================================================

class CauldronLoader:
    """Load and preprocess Cauldron dataset samples."""
    
    REPO = "HuggingFaceM4/the_cauldron"
    
    @classmethod
    def get_available_configs(cls) -> List[str]:
        """Get all available Cauldron configs."""
        return get_dataset_config_names(cls.REPO)
    
    @classmethod
    def load_samples(cls, config_name: str, n_samples: int = 100, random_sample: bool = False) -> List[Dict]:
        """
        Load samples from a Cauldron config using streaming.
        
        Parameters
        ----------
        config_name : str
            The Cauldron config name.
        n_samples : int
            Number of samples to load.
        random_sample : bool
            If True, randomly sample from a larger buffer.
        """
        import random as rnd
        
        ds = load_dataset(
            cls.REPO,
            config_name,
            streaming=True,
        )
        
        if random_sample:
            buffer_size = max(n_samples * 3, 1000)
            all_samples = list(ds['train'].take(buffer_size))
            if len(all_samples) <= n_samples:
                samples = all_samples
            else:
                samples = rnd.sample(all_samples, n_samples)
        else:
            samples = list(ds['train'].take(n_samples))
        
        return samples
    
    @classmethod
    def extract_qa(cls, sample: Dict, config_name: str) -> Optional[Dict[str, Any]]:
        """Extract image, prompt, and ground truth from a sample."""
        if not sample.get('images') or not sample.get('texts'):
            return None
            
        image = sample['images'][0]
        text_turn = sample['texts'][0]
        
        # Get task mapping
        router_task = CONFIG_TO_TASK.get(config_name, 'general_vqa')
        gt_type = TASK_GT_TYPE.get(router_task, 'exact')
        
        # Extract MC options if present
        mc_options = None
        prompt = text_turn['user']
        mc_match = re.findall(r'\(([A-D])\)\s*([^(]+?)(?=\([A-D]\)|$)', prompt)
        if mc_match:
            mc_options = [f"({letter}) {text.strip()}" for letter, text in mc_match]
        
        return {
            'image': image,
            'prompt': prompt,
            'ground_truth': text_turn['assistant'],
            'source': text_turn.get('source', config_name),
            'router_task': router_task,
            'ground_truth_type': gt_type,
            'mc_options': mc_options,
        }
