"""
Visualize model predictions on the test set.

Usage:
    python result_script.py                          # uses checkpoints/best_model.pt
    python result_script.py --checkpoint my_model.pt # uses a specific checkpoint
"""

import argparse
import os

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

from model.unet_model import UNetModel
from provider.dataset_provider import get_loader


def main():
    parser = argparse.ArgumentParser(description="Visualize model predictions on test set")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt). Defaults to checkpoints/best_model.pt",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to evaluate on (default: test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to save result images (default: results/)",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(__file__)

    # Resolve checkpoint path
    if args.checkpoint is None:
        args.checkpoint = os.path.join(project_root, "checkpoints", "best_model.pt")

    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found at {args.checkpoint}")
        print("Run training.py first to generate a checkpoint.")
        return

    # Create output directory
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = UNetModel(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["net_state_dict"])
    model.eval()

    print(f"Loaded checkpoint: {args.checkpoint} (epoch {checkpoint.get('epoch', '?')})")

    # Load test data
    test_ds = get_loader(base_path=project_root, dataset_type=args.split, batch_size=1)

    # Run inference and save side-by-side comparison images
    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(test_ds, desc="Inference")):
            data = data.to(device)
            prediction = model(data)

            pred_np = prediction.squeeze().numpy()
            target_np = target.squeeze().numpy()

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            im1 = axes[0].imshow(pred_np, cmap="viridis")
            axes[0].set_title("Prediction")
            fig.colorbar(im1, ax=axes[0])

            im2 = axes[1].imshow(target_np, cmap="viridis")
            axes[1].set_title("Ground Truth")
            fig.colorbar(im2, ax=axes[1])

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"result_{i:04d}.png"), dpi=100)
            plt.close(fig)

    print(f"Saved {i + 1} result images to {output_dir}/")


if __name__ == "__main__":
    main()
