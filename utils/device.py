"""utils/device.py — Hardware device detection (CUDA / MPS / CPU)"""
import sys

def get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        # Check MPS properly for Apple Silicon
        if hasattr(torch, "mps") and hasattr(torch.mps, "is_available"):
            if torch.mps.is_available():
                return "mps"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    # Platform-based fallback: detect Apple Silicon without torch
    import platform, subprocess
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(["sysctl", "-n", "hw.optional.arm64"],
                                    capture_output=True, text=True)
            if result.stdout.strip() == "1":
                return "mps"
        except Exception:
            pass
    return "cpu"

def get_device_label() -> str:
    device = get_device()
    return {"cuda": "🟢 CUDA GPU", "mps": "🔵 Apple MPS", "cpu": "⚪ CPU"}.get(device, device)
