import gc
import torch

def cleanup_resources():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
