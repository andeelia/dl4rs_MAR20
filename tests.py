"""
Quick sanity checks for the dataset and model.

Run: python tests.py
"""

import torch
from provider.dataset_provider import MAR20Dataset, DATASET_ROOT
from model.unet_model import UNetModel


def test_dataset():
    """Verify dataset loads correctly and returns the right shapes."""
    print("Testing dataset loading...")
    ds = MAR20Dataset(dataset_root=DATASET_ROOT, split="val")
    print(f"  Samples loaded: {len(ds)}")

    image, mask = ds[0]
    print(f"  Image shape: {image.shape}  (expected [3, 512, 512])")
    print(f"  Mask shape:  {mask.shape}   (expected [1, 512, 512])")
    assert image.shape == (3, 512, 512), f"Wrong image shape: {image.shape}"
    assert mask.shape == (1, 512, 512), f"Wrong mask shape: {mask.shape}"
    print("  Dataset OK")


def test_model():
    """Verify UNet forward pass produces correct output shape."""
    print("Testing model forward pass...")
    model = UNetModel(in_channels=3, out_channels=1)
    dummy = torch.randn(2, 3, 512, 512)
    output = model(dummy)
    print(f"  Input shape:  {dummy.shape}")
    print(f"  Output shape: {output.shape}  (expected [2, 1, 512, 512])")
    assert output.shape == (2, 1, 512, 512), f"Wrong output shape: {output.shape}"
    print("  Model OK")


def test_forward_pass():
    """Verify dataset + model work together end-to-end."""
    print("Testing end-to-end forward pass...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = MAR20Dataset(dataset_root=DATASET_ROOT, split="val")
    model = UNetModel(in_channels=3, out_channels=1).to(device)

    image, mask = ds[0]
    image_batch = image.unsqueeze(0).to(device)  # add batch dimension

    with torch.no_grad():
        prediction = model(image_batch)

    print(f"  Input:  {image_batch.shape}")
    print(f"  Target: {mask.shape}")
    print(f"  Output: {prediction.shape}")
    print("  End-to-end OK")


if __name__ == "__main__":
    test_dataset()
    test_model()
    test_forward_pass()
    print("\nAll tests passed!")
