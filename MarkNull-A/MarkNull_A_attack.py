
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

from WRN import Restormer
import cv2
from torch.nn import functional as F
import re
from diffusers import StableDiffusionPipeline
from utils.inverse_initial_noise import load_image, decode_vae_with_grad, encode_vae_with_grad, ddim_inversion_to_noise, encode_vae, decode_vae

  
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = Restormer(None, None).to(device)
model_path = "WRN.ckpt"
if ".pt" in model_path:
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    model.load_state_dict(torch.load(model_path, map_location=device)["state_dict"])

model.eval()


pipe = StableDiffusionPipeline.from_pretrained(
                "sd-legacy/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
            ).to(device)

def regenerate_img(img, pipe):
    
    z0 = encode_vae(pipe.vae, img, pipe.device, torch.float16)
    img = decode_vae(pipe.vae, z0)
    return img
    
def remove_watermark(model, img_path, save_path):
   
    toPIL = transforms.ToPILImage()
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    
    trans = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Lambda(lambda x: x / 255.0)
    ])
    
    img_tensor = trans(image).unsqueeze(0).to(device)
    #img_tensor  = regenerate_img(img_tensor, pipe)

    with torch.no_grad():
        output = model(img_tensor.to(device).to(torch.float32))
    
    output = torch.clamp(output, 0, 1)
    output = regenerate_img(output, pipe)
    
    output_img = toPIL(output.squeeze())
    output_img.save(save_path)
    return output_img


from tqdm import tqdm



def batch_process_images(input_dir, output_dir, length = 100):
    
    all_files = os.listdir(input_dir)[:length]
    
    for img_file in tqdm(all_files):
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, img_file)
        remove_watermark(model, input_path, output_path)

if __name__ == "__main__":
    
    input_dir = "" 
    output_dir = "" 
    os.makedirs(output_dir, exist_ok=True)
    batch_process_images(input_dir, output_dir, 100)
        
       
