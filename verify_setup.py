"""
Run this first to confirm that ROCm + PyTorch sees my RX 6800 XT.
    python verify_setup.py
"""
import torch


def main():
    print("=" * 60)
    print("PyTorch / ROCm setup-control")
    print("=" * 60)
    print(f"PyTorch-version:      {torch.__version__}")

    hip = getattr(torch.version, "hip", None)
    print(f"ROCm/HIP-version: {hip if hip else 'missing (NOT a ROCm-build!)'}")

    available = torch.cuda.is_available()
    print(f"GPU available: {available}")

    if not available:
        print("\n[WRONG] NO GPU AVAILABLE.")
        print("  - Confirm that you installed ROCm-wheels (torch.version.hip above).")
        print("  - Run 'rocminfo' and look after gfx1030 / Radeon RX 6800.")
        print("  - Control that your user is in the groups 'render' och 'video'.")
        return

    print(f"Number of GPU:s: {torch.cuda.device_count()}")
    print(f"GPU-name: {torch.cuda.get_device_name(0)}")

    # A little test calc on the GPU
    x = torch.randn(2048, 2048, device="cuda")
    y = torch.mm(x, x)
    torch.cuda.synchronize()
    print(f"Test matrix multiplication on GPU: OK (sum={y.sum().item():.1f})")

    mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Total VRAM: {mem:.1f} GB")
    print("\nEverything looks good - you are ready to train!")


if __name__ == "__main__":
    main()
