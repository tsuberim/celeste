import torch

def get_device(device: str = None) -> torch.device:
    """Pick the best available torch device"""
    if device is not None:
        selected_device = torch.device(device)
    elif torch.cuda.is_available():
        selected_device = torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        selected_device = torch.device("mps")
    else:
        selected_device = torch.device("cpu")
    
    print(f"Using device: {selected_device}")
    return selected_device
