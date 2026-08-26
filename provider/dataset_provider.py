"""
Dataset provider for the MAR20 aerial ship detection dataset.

Dataset layout (Pascal VOC-style):
    data/MAR20/
    ├── JPEGImages/              # RGB images as .jpg
    ├── Annotations/
    │   └── Horizontal Bounding Boxes/  # .xml files with <bndbox> annotations
    └── ImageSets/
        └── Main/
            ├── train.txt        # 1132 image IDs
            ├── val.txt          # 199 image IDs
            └── test.txt         # 2510 image IDs

The XML annotations contain multiple <object> entries, each with a class name
(e.g. A2, A10) and a <bndbox> (xmin, ymin, xmax, ymax). We rasterize all
bounding boxes into a single binary segmentation mask per image.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.utils.data as td
from PIL import Image
from tqdm import tqdm
from torchvision.transforms import Resize


# Resize target for both images and masks
TARGET_SIZE = (512, 512)

# Directory containing the full dataset
DATASET_ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "MAR20")


class MAR20Dataset(td.Dataset):
    """
    PyTorch Dataset for the MAR20 aircraft detection dataset.

    Each sample is a (image, mask) pair:
      - image: Float tensor of shape (3, 512, 512), values in [0, 1]
      - mask:  Float tensor of shape (1, 512, 512), binary (0.0 or 1.0)

    Images are loaded from JPEGImages/ and masks are created on-the-fly
    by rasterizing all bounding boxes from the XML annotations.
    """

    def __init__(self, dataset_root, split="train", transform=None):
        """
        Args:
            dataset_root: Path to the MAR20 folder (containing JPEGImages/, etc.)
            split: "train", "val", or "test" — reads the corresponding file from ImageSets/Main/
            transform: Optional torchvision transform applied to the image tensor
        """
        self.dataset_root = dataset_root
        self.transform = transform

        # Read the split file to get the list of image IDs
        split_file = os.path.join(dataset_root, "ImageSets", "Main", f"{split}.txt")
        with open(split_file, "r") as f:
            # Each line is a numeric ID (e.g. "1680"), strip whitespace
            self.image_ids = [line.strip() for line in f if line.strip()]

        # Preload all images and their corresponding masks into memory.
        # This is fast for ~3800 images and avoids repeated disk I/O during training.
        self.samples = []
        for img_id in tqdm(self.image_ids, desc=f"Loading {split} set"):
            image = self._load_image(img_id)
            mask = self._load_mask(img_id)
            self.samples.append((image, mask))

    def _load_image(self, img_id):
        """
        Load a JPEG image and return it as a (3, H, W) float tensor normalized to [0, 1].
        """
        img_path = os.path.join(self.dataset_root, "JPEGImages", f"{img_id}.jpg")
        # Open as RGB to ensure 3 channels even for grayscale images
        pil_image = Image.open(img_path).convert("RGB")
        # Convert to numpy array: shape (H, W, 3), dtype uint8
        np_image = np.array(pil_image)
        # Transpose to (C, H, W) and normalize from [0, 255] to [0.0, 1.0]
        tensor = torch.from_numpy(np_image).permute(2, 0, 1).float() / 255.0
        # Resize to target dimensions
        tensor = Resize(TARGET_SIZE)(tensor)
        return tensor

    def _load_mask(self, img_id):
        """
        Parse the Pascal VOC XML annotation and rasterize all bounding boxes
        into a single binary mask of shape (1, H, W).

        Each XML contains one or more <object> entries like:
            <object>
                <name>A2</name>
                <bndbox>
                    <xmin>485</xmin>
                    <ymin>427</ymin>
                    <xmax>554</xmax>
                    <ymax>500</ymax>
                </bndbox>
            </object>
        """
        xml_path = os.path.join(
            self.dataset_root, "Annotations", "Horizontal Bounding Boxes", f"{img_id}.xml"
        )
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Read image dimensions from the <size> tag
        width = int(root.find("size/width").text)
        height = int(root.find("size/height").text)

        # Create an empty mask (H, W) filled with zeros
        mask = np.zeros((height, width), dtype=np.uint8)

        # Iterate over all <object> entries and draw filled rectangles
        for obj in root.findall("object"):
            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
            # Set the bounding box region to 1 (foreground)
            mask[ymin:ymax, xmin:xmax] = 1

        # Convert to tensor: shape (1, H, W)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()

        # Resize mask to match the target size (nearest-neighbor preserves binary values)
        mask_tensor = Resize(TARGET_SIZE, interpolation=Image.NEAREST)(mask_tensor)

        return mask_tensor

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image, mask = self.samples[index]

        # Apply optional transform to the image only (e.g. augmentations)
        if self.transform is not None:
            image = self.transform(image)

        return image, mask


def get_loader(base_path, dataset_type, batch_size):
    """
    Create a DataLoader for the MAR20 dataset.

    Args:
        base_path: Path to the project root (not the MAR20 folder itself)
        dataset_type: "train", "val", or "test"
        batch_size: Number of samples per batch

    Returns:
        A torch DataLoader that yields (image_batch, mask_batch) tuples.
    """
    dataset = MAR20Dataset(
        dataset_root=DATASET_ROOT,
        split=dataset_type,
    )

    return td.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
    )
