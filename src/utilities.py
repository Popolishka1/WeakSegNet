import torch

def load_device():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[Using device: {DEVICE}]")
    return DEVICE


def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()