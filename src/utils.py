import torch


import torch

def load_device():
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cpu")
    print(f"\n[Using device: {DEVICE}]")
    return DEVICE

def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # MPS currently does not require explicit cache clearing
