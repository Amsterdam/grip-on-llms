"""Common LLM utils used accross the different types of models"""
import gc
import logging

import torch


def aggressive_gpu_cleanup():
    # Aggressive GPU memory cleanup
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logging.info(
            f"Pre-cleanup GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        torch.cuda.empty_cache()

        # Log memory status
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logging.info(
            f"Post-cleanup GPU memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
