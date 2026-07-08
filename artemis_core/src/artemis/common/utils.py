import random
import os
import numpy as np
import torch
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def set_seed(seed: int = 42) -> None:
    """
    Set the random seed for reproducibility across all libraries.
    
    Args:
        seed: The integer seed to use.
    """
    logger.info(f"Setting global random seed to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior for CuDNN
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """
    Configure logging with a standard format.
    
    Args:
        level: transform logging level (default INFO)
        log_file: Optional path to write logs to a file.
    """
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logger.info(f"Logging initialized. Writing to {log_file}")
