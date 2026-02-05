import argparse
import os
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from brisque import BRISQUE
import numpy as np

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def load_image_as_tensor(path: Path) -> torch.Tensor:
    """
    Load an image from path and convert to a float tensor in [0, 1], shape (3, H, W).
    """
    img = Image.open(path).convert("RGB")
    to_tensor = transforms.ToTensor()  # -> [0,1], (C,H,W)
    tensor = to_tensor(img)
    return tensor


def collect_pairs(dir1: Path, dir2: Path):
    """
    Collect image pairs with the same filename in dir1 and dir2.
    Returns a list of (path1, path2).
    """
    files1 = {p.name: p for p in dir1.iterdir() if p.is_file() and is_image_file(p)}
    files2 = {p.name: p for p in dir2.iterdir() if p.is_file() and is_image_file(p)}

    common_names = sorted(set(files1.keys()) & set(files2.keys()))

    pairs = [(files1[name], files2[name]) for name in common_names]
    return pairs

def brisque_eval(img_paths):
    brique_sum = 0
    for i in os.listdir(img_paths):
        img = Image.open(img_paths + "/" +i)
        ndarray = np.asarray(img)
        obj = BRISQUE(url=False)
        score = obj.score(img=ndarray)
        brique_sum += score
    return brique_sum / len(os.listdir(img_paths))


def img_quality_eval(folder1: str, folder2: str, device: str = "cuda"):
    dir1 = Path(folder1)
    dir2 = Path(folder2)

    assert dir1.is_dir(), f"{dir1} is not a directory"
    assert dir2.is_dir(), f"{dir2} is not a directory"

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pairs = collect_pairs(dir1, dir2)
    if not pairs:
        print("No matching image filenames found between the two folders.")
        return

    print(f"Found {len(pairs)} matching image pairs.")

    # Initialize metrics
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type="vgg").to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    fid_metric = FrechetInceptionDistance().to(device)

    lpips_sum = 0.0
    psnr_sum = 0.0
    ssim_sum = 0.0
    n = 0

    for path1, path2 in pairs:
        img1 = load_image_as_tensor(path1)
        img2 = load_image_as_tensor(path2)

        # Ensure same spatial size
        if img1.shape != img2.shape:
            print(f"[Warning] Shape mismatch for {path1.name}: {img1.shape} vs {img2.shape}. "
                  f"Resizing second to first.")
            _, H, W = img1.shape
            img2_pil = transforms.ToPILImage()(img2)
            img2 = transforms.ToTensor()(transforms.Resize((H, W))(img2_pil))

        img1_batch = img1.unsqueeze(0).to(device)  # (1,3,H,W)
        img2_batch = img2.unsqueeze(0).to(device)

        # --- LPIPS ---
        img1_lpips = img1_batch * 2.0 - 1.0
        img2_lpips = img2_batch * 2.0 - 1.0
        lpips_val = lpips_metric(img1_lpips, img2_lpips).item()
        lpips_sum += lpips_val

        # --- PSNR ---
        psnr_val = psnr_metric(img1_batch, img2_batch).item()
        psnr_sum += psnr_val

        # --- SSIM ---
        ssim_val = ssim_metric(img1_batch, img2_batch).item()
        ssim_sum += ssim_val

        # --- FID ---
        # Treat folder1 as real, folder2 as fake
        fid_img1 = (img1_batch * 255.0).clamp(0, 255).to(torch.uint8)
        fid_img2 = (img2_batch * 255.0).clamp(0, 255).to(torch.uint8)

        fid_metric.update(fid_img1, real=True)
        fid_metric.update(fid_img2, real=False)

        n += 1

    # Compute averages
    lpips_avg = lpips_sum / n
    psnr_avg = psnr_sum / n
    ssim_avg = ssim_sum / n
    fid_val = fid_metric.compute().item()
    brisque_avg = brisque_eval(folder2)
    
    print("======================================")
    print(f"Folder 1 (real): {dir1}")
    print(f"Folder 2 (fake): {dir2}")
    print(f"#pairs         : {n}")
    print("======================================")
    print(f"{'Metric':<15} | {'Value':>12} | Note")
    print("-" * 55)
    print(f"{'LPIPS (avg)':<15} | {lpips_avg:>12.6f} | lower = better")
    print(f"{'FID':<15} | {fid_val:>12.6f} | lower = better (dataset-level)")
    print(f"{'PSNR (avg)':<15} | {psnr_avg:>12.4f} | higher = better (dB)")
    print(f"{'SSIM (avg)':<15} | {ssim_avg:>12.6f} | higher = better (max 1.0)")
    print("======================================")
    # One-line, comma-separated summary for easy copy into Excel/Sheets
    print(f"{lpips_avg:.6f}\t{fid_val:.6f}\t{psnr_avg:.4f}\t{ssim_avg:.6f}")

    return {
        "LPIPS": lpips_avg,
        "FID": fid_val,
        "PSNR": psnr_avg,
        "SSIM": ssim_avg,
        "BRISQUE": brisque_avg,
    }
