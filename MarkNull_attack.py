import torch
import torch.nn.functional as F

import torch
from diffusers import StableDiffusionPipeline
import sys
from utils.inverse_initial_noise import load_image  
from torchvision import transforms
from torchvision.utils import save_image
import os
import argparse
from tqdm import tqdm
from utils.img_quality import img_quality_eval
from utils.marknull_nlas import WMRemover

if __name__ == "__main__":
   
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Proxy Model 
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to(device)
            
    attacker = WMRemover(pipe=pipe,
                        device=device,
                        dtype=torch.float16,
                        epsilon=1,
                        lr=0.05,
                        num_steps=50)
    dataset_path = "./Watermarked"
    tasks = os.listdir(dataset_path)
    
    f = open("img_quality_metric.txt", "a")
    for task in tasks:
        wm_img_path = os.path.join(dataset_path, task)
        img_files = os.listdir(wm_img_path)

        target_path = ""
        save_path = target_path + wm_img_path.split("/")[-1] 
        
        os.makedirs(save_path, exist_ok=True)
        for img_name in tqdm(img_files[:100]):
            img_path = os.path.join(wm_img_path, img_name)
            image = load_image(img_path)
            reconstructed = attacker.attack_removal(image, prompt="", negative_prompt="")
            print(os.path.join(save_path, img_name))
        
        metrics = img_quality_eval(wm_img_path,save_path, device)
        f.write(f"Task: {task}, LPIPS: {metrics['LPIPS']:.6f}, FID: {metrics['FID']:.6f}, PSNR: {metrics['PSNR']:.4f}, SSIM: {metrics['SSIM']:.6f}, BRISQUE: {metrics['BRISQUE']:.6f}\n")
