import random
from pathlib import Path
from PIL import Image
from typing import Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import lightning.pytorch as pl


class CleanImageDataset(Dataset):
    def __init__(self, clean_root: str, transform: Optional[Callable] = None):
        self.clean_root = Path(clean_root)
        self.transform = transform
        self.image_paths = []
        self._prepare_paths()

    def _prepare_paths(self):
        if not self.clean_root.exists():
            print(f"Error: Path not found {self.clean_root}")
            return

        extensions = ['*.png', '*.jpg', '*.jpeg']
        for ext in extensions:
            self.image_paths.extend(list(self.clean_root.rglob(ext)))

        print(f"Found {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.image_paths[idx]).convert("RGB")
        except Exception as e:
            print(f"Error loading image index {idx}: {e}")
            return torch.zeros(3, 256, 256)

        if self.transform:
            img = self.transform(img)
        return img


class CropTransform:
    def __init__(self, size=256, is_train=True):
        self.size = size
        self.is_train = is_train

    def __call__(self, img):
        if self.is_train:
            i, j, h, w = T.RandomCrop.get_params(img, output_size=(self.size, self.size))
            img = TF.crop(img, i, j, h, w)
            if random.random() > 0.5:
                img = TF.hflip(img)
        else:
            img = TF.center_crop(img, output_size=(self.size, self.size))
        return TF.to_tensor(img)


class SubsetWrapper(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        return self.transform(self.subset[idx])

    def __len__(self):
        return len(self.subset)


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, clean_root: str, batch_size: int = 16, patch_size: int = 256,
                 num_workers: int = 4, val_split: float = 0.2):
        super().__init__()
        self.clean_root = clean_root
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage: Optional[str] = None):
        full_dataset = CleanImageDataset(self.clean_root, transform=None)
        val_len = int(len(full_dataset) * self.val_split)
        train_len = len(full_dataset) - val_len

        train_ds, val_ds = random_split(
            full_dataset, [train_len, val_len],
            generator=torch.Generator().manual_seed(42)
        )
        self.train_ds = SubsetWrapper(train_ds, CropTransform(self.patch_size, is_train=True))
        self.val_ds   = SubsetWrapper(val_ds,   CropTransform(self.patch_size, is_train=False))

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)


if __name__ == "__main__":
    CLEAN_PATH = "/home/nisp/Jie/MarkNull/MarkNull-A/Dataset/Watermark"

    dm = ImageDataModule(clean_root=CLEAN_PATH, batch_size=8, patch_size=256)
    dm.setup()

    batch = next(iter(dm.train_dataloader()))
    print(f"Batch shape: {batch.shape}")  # [8, 3, 256, 256]