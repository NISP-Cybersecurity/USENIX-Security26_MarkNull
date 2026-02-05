import os
import glob
import random
from pathlib import Path
from PIL import Image
from typing import Optional, Callable, Tuple

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import pytorch_lightning as pl

# -----------------------------------------------------------------------------
# 1. Custom Dataset: Responsible for matching file paths
# -----------------------------------------------------------------------------
class WatermarkDataset(Dataset):
    def __init__(
        self, 
        clean_root: str, 
        watermarked_root: str, 
        transform: Optional[Callable] = None
    ):
        self.clean_root = Path(clean_root)
        self.watermarked_root = Path(watermarked_root)
        self.transform = transform
        self.image_pairs = []

        self._prepare_pairs()

    def _prepare_pairs(self):
        
        if not self.clean_root.exists():
            print(f"Error: Path not found {self.clean_root}")
            return

        method_dirs = [d for d in self.clean_root.iterdir() if d.is_dir()]

        for method_dir in method_dirs:
            method_name = method_dir.name
            
            # Get all images under this method directory
            extensions = ['*.png', '*.jpg', '*.jpeg', '*.ong']
            images = []
            for ext in extensions:
                images.extend(list(method_dir.glob(ext)))
            
            for img_path_clean in images:
                file_name = img_path_clean.name
                
                # Construct the corresponding watermarked image path
                img_path_watermarked = self.watermarked_root / method_name / file_name
                
                # Fix filename typo: if clean is .ong, assume corresponding watermarked might be .png
                # Or based on your actual file structure, assume filenames are identical
                if file_name.endswith('.ong'):
                     # If your file is actually named .ong, keep it; if it's a typo, logic below handles .png
                     pass 

                # Only add to list if files exist on both sides
                if img_path_watermarked.exists():
                    self.image_pairs.append((str(img_path_clean), str(img_path_watermarked)))
                else:
                    # If clean is .ong but watermarked is .png, try replacing extension to find it
                    if file_name.endswith('.ong'):
                        fixed_name = file_name.replace('.ong', '.png')
                        img_path_watermarked_fix = self.watermarked_root / method_name / fixed_name
                        if img_path_watermarked_fix.exists():
                             self.image_pairs.append((str(img_path_clean), str(img_path_watermarked_fix)))

        print(f"Found {len(self.image_pairs)} valid image pairs.")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        clean_path, watermarked_path = self.image_pairs[idx]

        try:
            clean_img = Image.open(clean_path).convert("RGB")
            watermarked_img = Image.open(watermarked_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image pair index {idx}: {e}")
            return torch.zeros(3, 256, 256), torch.zeros(3, 256, 256)

        if self.transform:
            clean_img, watermarked_img = self.transform(clean_img, watermarked_img)

        return clean_img, watermarked_img

# -----------------------------------------------------------------------------
# 2. Synchronized Transform Class: Ensures consistent cropping for positive/negative samples
# -----------------------------------------------------------------------------
class PairedCropTransform:
    def __init__(self, size=256, is_train=True):
        self.size = size
        self.is_train = is_train

    def __call__(self, img1, img2):
        # 1. Random Crop (during training) / Center Crop (during validation/testing)
        if self.is_train:
            # Use torchvision to get random crop parameters
            i, j, h, w = T.RandomCrop.get_params(
                img1, output_size=(self.size, self.size)
            )
            img1 = TF.crop(img1, i, j, h, w)
            img2 = TF.crop(img2, i, j, h, w)
            
            # 2. Random horizontal flip
            if random.random() > 0.5:
                img1 = TF.hflip(img1)
                img2 = TF.hflip(img2)
        else:
            # Validation set uses center crop
            img1 = TF.center_crop(img1, output_size=(self.size, self.size))
            img2 = TF.center_crop(img2, output_size=(self.size, self.size))

        # 3. Convert to Tensor
        img1 = TF.to_tensor(img1)
        img2 = TF.to_tensor(img2)

        return img1, img2

# -----------------------------------------------------------------------------
# 3. Lightning DataModule
# -----------------------------------------------------------------------------
class SubsetWrapper(Dataset):
    """
    Helper class: used to apply different Transforms to split Subsets
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, idx):
        clean_img, watermarked_img = self.subset[idx]
        return self.transform(clean_img, watermarked_img)
        
    def __len__(self):
        return len(self.subset)

class WatermarkDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        clean_root: str, 
        watermarked_root: str, 
        batch_size: int = 16, 
        patch_size: int = 256,
        num_workers: int = 4,
        val_split: float = 0.2
    ):
        super().__init__()
        self.clean_root = clean_root
        self.watermarked_root = watermarked_root
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.val_split = val_split

    def setup(self, stage: Optional[str] = None):
        train_transform = PairedCropTransform(size=self.patch_size, is_train=True)
        val_transform = PairedCropTransform(size=self.patch_size, is_train=False)

        full_dataset = WatermarkDataset(
            self.clean_root, 
            self.watermarked_root, 
            transform=None
        )

        total_len = len(full_dataset)
        val_len = int(total_len * self.val_split)
        train_len = total_len - val_len

        # Random split
        self.train_ds, self.val_ds = random_split(
            full_dataset, [train_len, val_len], 
            generator=torch.Generator().manual_seed(42)
        )

        # Inject transform
        self.train_ds = SubsetWrapper(self.train_ds, train_transform)
        self.val_ds = SubsetWrapper(self.val_ds, val_transform)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True
        )

if __name__ == "__main__":
    # Configure your paths
    ATTACKED_PATH = "" # 
    WATERMARKED_PATH = ""

    dm = WatermarkDataModule(
        clean_root=ATTACKED_PATH,
        watermarked_root=WATERMARKED_PATH,
        batch_size=8,
        patch_size=256
    )

    dm.setup()
    
    train_loader = dm.train_dataloader()
    try:
        clean_batch, watermarked_batch = next(iter(train_loader))
        print(f"Clean Batch Shape: {clean_batch.shape}")
        print(f"Watermarked Batch Shape: {watermarked_batch.shape}")
    except StopIteration:
        print("Dataset is empty. Check paths.")