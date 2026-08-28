import glob
import os

import torch
import torch.utils.data as td
import skimage.io as io
import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms import CenterCrop, ToTensor, Resize

from tqdm import tqdm

import torchvision.transforms as transforms

custom_transform = transforms.Compose([
    Resize((512, 512))
])

class DeadTreeDataset(td.Dataset):
    def __init__(self, base_path, dataset_type):
        self.dataset = []

        rgb_files = glob.glob(os.path.join(base_path, "data/true_color_images/" + dataset_type + "/RGB_*.png"))

        for rgb_file in tqdm(rgb_files):
            id_ = os.path.basename(rgb_file).replace("RGB_", "").replace(".png", "")

            mask_file = os.path.join(base_path, "data/true_color_images/" + dataset_type + "/mask_" + id_ + ".png")

            rgb_read = io.imread(rgb_file)
            mask_read = io.imread(mask_file)

            rgb_read = np.transpose(rgb_read, (2, 0, 1))
            mask_read = np.expand_dims(mask_read, axis=0)

            rgb_read = torch.IntTensor(rgb_read) / 255.0
            mask_read = torch.Tensor(mask_read) / 255.0

            rgb_read = custom_transform(rgb_read)
            mask_read = custom_transform(mask_read)

            self.dataset.append((rgb_read, mask_read))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


def get_loader(base_path, dataset_type, batch_size):
    dataset = DeadTreeDataset(base_path, dataset_type)

    return td.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )
