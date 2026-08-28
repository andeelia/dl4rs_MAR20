import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from tqdm import tqdm

from model.unet_model import UNetModel
from provider.dataset_provider import get_loader


def train(model, loss_fn, optimizer, epoch, train_ds):
    model.train()

    running_loss = []

    loop = tqdm(train_ds)

    for (data, target) in loop:
        optimizer.zero_grad(set_to_none=True)

        prediction = model(data.float())

        loss = loss_fn(prediction, target)
        loss.backward()

        optimizer.step()

        running_loss.append(loss.item())
        loop.set_postfix_str(f"Epoch {epoch}: loss is " + str(loss.item()))

    return np.mean(running_loss)



def validation(model, loss_fn, epoch, valid_ds):
    model.eval()

    running_loss = []

    loop = tqdm(valid_ds)

    for (data, target) in loop:
        prediction = model(data.float())

        loss = loss_fn(prediction, target)

        running_loss.append(loss.item())
        loop.set_postfix_str(f"Epoch {epoch}: loss is " + str(loss.item()))

    return np.mean(running_loss)

if __name__ == '__main__':
    BASE_PATH = "/home/caipi/PycharmProjects/PythonProject1/"

    # model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    #     in_channels=3, out_channels=1, init_features=8, pretrained=False)

    model = UNetModel(in_channels=3, out_channels=1)

    loss_fn = nn.BCEWithLogitsLoss()
    optim = opt.Adam(model.parameters(), lr=0.001)
    batch_size = 1

    train_ds = get_loader(base_path=BASE_PATH, dataset_type="train", batch_size=4)
    val_ds = get_loader(base_path=BASE_PATH, dataset_type="validation", batch_size=4)

    all_tr_losses = []
    all_val_losses = []

    for epoch in range(20):
        tr_loss = train(model=model, loss_fn=loss_fn, optimizer=optim, epoch=epoch, train_ds=train_ds)
        val_loss = validation(model=model, loss_fn=loss_fn, epoch=epoch, valid_ds=val_ds)

        all_tr_losses.append(tr_loss)
        all_val_losses.append(val_loss)

    plt.plot(all_tr_losses, color="blue")
    plt.plot(all_val_losses, color="red")
    plt.show()


