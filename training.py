import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves plots to file instead of showing
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm

from model.unet_model import UNetModel
from provider.dataset_provider import get_loader


def train(model, loss_fn, optimizer, epoch, train_ds, device):
    """Run one training epoch. Returns the average loss over all batches."""
    model.train()

    running_loss = []
    loop = tqdm(train_ds, desc=f"Train epoch {epoch}")

    for (data, target) in loop:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)

        prediction = model(data)
        loss = loss_fn(prediction, target)
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return np.mean(running_loss)


def validation(model, loss_fn, epoch, valid_ds, device):
    """Run one validation epoch. Returns the average loss over all batches."""
    model.eval()

    running_loss = []
    loop = tqdm(valid_ds, desc=f"Val   epoch {epoch}")

    for (data, target) in loop:
        data, target = data.to(device), target.to(device)
        with torch.no_grad():
            prediction = model(data)
            loss = loss_fn(prediction, target)

        running_loss.append(loss.item())
        loop.set_postfix(loss=f"{loss.item():.4f}")

    return np.mean(running_loss)


if __name__ == '__main__':
    # Paths
    PROJECT_ROOT = os.path.dirname(__file__)
    CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Hyperparameters
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    TRAIN_BATCH_SIZE = 4
    VAL_BATCH_SIZE = 4

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model
    model = UNetModel(in_channels=3, out_channels=1).to(device)

    # Loss and optimizer
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = opt.Adam(model.parameters(), lr=LEARNING_RATE)

    # Data
    train_ds = get_loader(base_path=PROJECT_ROOT, dataset_type="train", batch_size=TRAIN_BATCH_SIZE)
    val_ds = get_loader(base_path=PROJECT_ROOT, dataset_type="val", batch_size=VAL_BATCH_SIZE)

    # Training loop
    all_tr_losses = []
    all_val_losses = []
    best_val_loss = float("inf")

    for epoch in range(NUM_EPOCHS):
        tr_loss = train(model, loss_fn, optimizer, epoch, train_ds, device)
        val_loss = validation(model, loss_fn, epoch, val_ds, device)

        all_tr_losses.append(tr_loss)
        all_val_losses.append(val_loss)

        print(f"Epoch {epoch}: train_loss={tr_loss:.4f}, val_loss={val_loss:.4f}")

        # Save checkpoint after every epoch
        checkpoint = {
            "epoch": epoch,
            "net_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": tr_loss,
            "val_loss": val_loss,
        }
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"model_epoch{epoch}.pt"))

        # Keep track of the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, "best_model.pt"))

    # Save loss plot to file
    plt.figure()
    plt.plot(all_tr_losses, color="blue", label="Train")
    plt.plot(all_val_losses, color="red", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and Validation Loss")
    plt.savefig(os.path.join(CHECKPOINT_DIR, "loss_curve.png"), dpi=150)
    print(f"Loss curve saved to {CHECKPOINT_DIR}/loss_curve.png")
    print(f"Best model saved to {CHECKPOINT_DIR}/best_model.pt (val_loss={best_val_loss:.4f})")
