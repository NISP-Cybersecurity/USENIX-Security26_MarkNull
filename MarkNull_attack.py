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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default=None)
args = parser.parse_args()


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
    
    # Image Attack Removal
    if "SVD" not in args.input_dir:
        img_files = os.listdir(args.input_dir)

        target_path = "Attacked/MarkNull/"
        save_path = target_path + args.input_dir.split("/")[-1] 
        os.makedirs(save_path, exist_ok=True)
        
        for img_name in tqdm(img_files):
            img_path = os.path.join(args.input_dir, img_name)
            image = load_image(img_path)
            attacked_img = attacker.attack_removal(image, prompt="", negative_prompt="")
            save_image(attacked_img, os.path.join(save_path, img_name))    
            
        # metrics = img_quality_eval(args.input_dir, save_path, device)
        # print(f"LPIPS: {metrics['LPIPS']:.6f}, FID: {metrics['FID']:.6f}, PSNR: {metrics['PSNR']:.4f}, SSIM: {metrics['SSIM']:.6f}, BRISQUE: {metrics['BRISQUE']:.6f}\n")

    # Video Attack Removal
    else:
        video_files = os.listdir(args.input_dir)

       
        for video in video_files:
            frame_path = os.path.join(args.input_dir, video)
            frame_file = os.listdir(frame_path) 
        
            save_path =  "Attacked/MarkNull/" + args.input_dir.split("/")[-1] + "/" + video
            os.makedirs(save_path, exist_ok=True)
        
            for frame in tqdm(frame_file):
                img_path = os.path.join(args.input_dir, video, frame)
                image = load_image(img_path)
                attacked_frame = attacker.attack_removal(image, prompt="", negative_prompt="")
                save_image(attacked_frame, os.path.join(save_path, frame))    
                
                