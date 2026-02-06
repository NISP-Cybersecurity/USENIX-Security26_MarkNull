import torch
import torch.nn.functional as F
import seaborn as sns
import torch
from diffusers import StableDiffusionPipeline
import sys
from utils.inverse_initial_noise import load_image, encode_vae, ddim_inversion_to_noise, ddim_zT_to_z0, decode_vae
from torchvision import transforms
from torchvision.utils import save_image
import os
import argparse
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


device = "cuda:2" if torch.cuda.is_available() else "cpu"
pipe = StableDiffusionPipeline.from_pretrained(
    #"sd-legacy/stable-diffusion-v1-5",
    "stabilityai/stable-diffusion-2-1-base",
    torch_dtype=torch.float16,
).to(device)
dtype = torch.float16


lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device).eval()
lpips_fn.requires_grad_(False)
totensor = transforms.ToTensor()


def detect_attack(image_path, pipe, threshold=0.05):
    image = load_image(image_path)
    img_tensor = totensor(image).unsqueeze(0).to(device=device, dtype=dtype)
    z0 = encode_vae(pipe.vae, img_tensor, device, dtype)
    zT, _, _ = ddim_inversion_to_noise(
            pipe,
            image=None,
            prompt="",
            negative_prompt="",
            num_inference_steps=50,
            guidance_scale=7.5,
            x0=z0,
        )
    
    z0_rec = ddim_zT_to_z0(pipe, zT)
    img_rec = decode_vae(pipe.vae, z0_rec)
    img_rec = torch.clamp(img_rec, 0.0, 1.0)
    diff_lpips = lpips_fn(img_tensor, img_rec).item()
    diff_mse =  F.mse_loss(img_tensor, img_rec, reduction='none').view(img_tensor.shape[0], -1).mean().item()
    save_image(img_rec, "1.png")
    #print(diff_lpips, diff_mse)
    return diff_lpips, diff_mse 



def calculate_optimal_threshold(scores_clean, scores_attacked, method='percentile', alpha=0.05):
    """
    scores_clean: list/array of LPIPS errors for normal watermarked images
    scores_attacked: list/array of LPIPS errors for attacked images
    """
    scores_clean = np.array(scores_clean)
    scores_attacked = np.array(scores_attacked)

    print(f"Mean Clean Error: {np.mean(scores_clean):.4f} ± {np.std(scores_clean):.4f}")
    print(f"Mean Attack Error: {np.mean(scores_attacked):.4f} ± {np.std(scores_attacked):.4f}")

    threshold = 0.0

    if method == 'percentile':
        
        threshold = np.percentile(scores_clean, 100 * (1 - alpha))
        print(f"\n[Method: Fixed FPR at {alpha}]")
        print(f"Threshold determined at {100*(1-alpha)}th percentile: {threshold:.5f}")
        
    elif method == 'youden':
        y_true = [0] * len(scores_clean) + [1] * len(scores_attacked)
        y_scores = np.concatenate([scores_clean, scores_attacked])
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        J = tpr - fpr
        ix = np.argmax(J)
        threshold = thresholds[ix]
        
        print(f"\n[Method: Youden Index (Best Balance)]")
        print(f"Best Threshold: {threshold:.5f}, TPR={tpr[ix]:.3f}, FPR={fpr[ix]:.3f}")
   
    
    return threshold

def draw_img(att_dataset,save_path):
    images = os.listdir(att_dataset)

    clean_lpips_ls = []
    clean_mse_ls = []
    att_lpips_ls = []
    att_mse_ls = []

    for fname in images:
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
            continue

        att_image_path = os.path.join(att_dataset, fname)
        clean_image_path = os.path.join(f"./Watermarked/{att_dataset.split('/')[-1]}/", fname)

        # attacked
        att_lpips, att_mse = detect_attack(att_image_path, pipe)
        att_lpips_ls.append(float(att_lpips))
        att_mse_ls.append(float(att_mse))

        # clean
        clean_lpips, clean_mse = detect_attack(clean_image_path, pipe)
        clean_lpips_ls.append(float(clean_lpips))
        clean_mse_ls.append(float(clean_mse))

    att_lpips_arr = np.asarray(att_lpips_ls, dtype=np.float32)
    #att_mse_arr   = np.asarray(att_mse_ls,   dtype=np.float32)
    clean_lpips_arr = np.asarray(clean_lpips_ls, dtype=np.float32)
    #clean_mse_arr   = np.asarray(clean_mse_ls,   dtype=np.float32)

    T_best = calculate_optimal_threshold(clean_lpips_arr, att_lpips_arr, method='youden')
    plot_distribution(clean_lpips_arr , att_lpips_arr , T_best, save_path, metric_name="LPIPS")
    
if __name__ == "__main__":
    datasets = "./Attacked/MarkNull/"
    for i in os.listdir(datasets):
        att_dataset = datasets + "/" + i
        save_path = f"./Defense_res/MarkNull/{i}_LPIPS.pdf"
        draw_img(att_dataset, save_path)
