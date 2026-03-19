"""
utils/device.py — Hardware device detection (CUDA / MPS / CPU)
"""
import sys


def get_device() -> str:
    """
    Returns the best available compute device string:
      'cuda'  — NVIDIA GPU via PyTorch
      'mps'   — Apple Silicon via PyTorch
      'cpu'   — fallback
    """
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def get_device_label() -> str:
    device = get_device()
    labels = {
        "cuda": "🟢 CUDA GPU",
        "mps":  "🔵 Apple MPS",
        "cpu":  "⚪ CPU",
    }
    return labels.get(device, device)


if __name__ == "__main__":
    print(get_device_label())
