import torch
from tqdm import tqdm

from provider.dataset_provider import get_loader
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    test_ds = get_loader(
        base_path="/home/caipi/PycharmProjects/PythonProject1/",
        dataset_type="test",
        batch_size=1
    )

    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                in_channels=3, out_channels=1, init_features=32, pretrained=False, force_reload=True)

    state_dict = torch.load(
        "/home/caipi/PycharmProjects/PythonProject1/BCEWithLogitsLoss_Adam_0.0001/model_epoch28.pt",
        weights_only=False
    )

    model.load_state_dict(state_dict["net_state_dict"])

    loop = tqdm(test_ds)

    for (data, target) in loop:
        prediction = model(data)

        prediction = prediction.detach().squeeze().squeeze().numpy()
        target = target.detach().squeeze().squeeze().numpy()

        # Create a figure with 1 row and 2 columns
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot the Prediction
        im1 = axes[0].imshow(prediction, cmap="viridis")
        axes[0].set_title("Prediction")
        fig.colorbar(im1, ax=axes[0])

        # Plot the Target
        im2 = axes[1].imshow(target, cmap="viridis")
        axes[1].set_title("Target")
        fig.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        plt.show()
