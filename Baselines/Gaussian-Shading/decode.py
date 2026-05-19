import argparse
import copy
from tqdm import tqdm
import torch
from transformers import CLIPModel, CLIPTokenizer
from inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
import open_clip
from optim_utils import *
from io_utils import *
from image_utils import *
from watermark import *
from datasets import load_dataset
from PIL import Image, ImageOps
import torchvision.transforms as T

########
import torch
from diffusers import StableDiffusionPipeline
import sys

random.seed(123)
all_prompts = load_dataset('Gustavosta/Stable-Diffusion-Prompts')['train']
prompts = all_prompts.shuffle(seed=42)

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_path, subfolder='scheduler')
    pipe = InversableStableDiffusionPipeline.from_pretrained(
            args.model_path,
            scheduler=scheduler,
            torch_dtype=torch.float16,
    )
    pipe.safety_checker = None
    pipe = pipe.to(device)

    tester_prompt = ''
    text_embeddings = pipe.get_text_embedding(tester_prompt)
    
    #acc
    acc = []
    #CLIP Scores
    clip_scores = []
    save_path = args.path
    img_files = os.listdir(save_path)
    
    tpr_detection_sum = []
    tpr_traceability_sum = []
    args.num = len(img_files)
    
    for i in tqdm(range(args.num)):
        
        set_random_seed(i)
        watermark = Gaussian_Shading_chacha(args.channel_copy, args.hw_copy, args.fpr, args.user_number)
        img_path = os.path.join(save_path, f'{i}.png')
        if os.path.exists(img_path):
                
            image_w = Image.open(img_path).convert("RGB")
            image_w = transform_img(image_w).unsqueeze(0).to(text_embeddings.dtype).to(device)
            image_latents_w = pipe.get_image_latents(image_w, sample=False)
            reversed_latents_w = pipe.forward_diffusion(
                latents=image_latents_w,
                text_embeddings=text_embeddings,
                guidance_scale=1,
                num_inference_steps=args.num_inversion_steps,
            )

            #acc metric
            acc_metric = watermark.eval_watermark(reversed_latents_w)
            acc.append(acc_metric)
        #tpr metric
            tpr_detection, tpr_traceability = watermark.get_tpr()
            tpr_detection_sum.append(tpr_detection)
            tpr_traceability_sum.append(tpr_traceability)
    print(f'Acc: {sum(acc)/len(acc)}')
    print(f'TPR_detection: {sum(tpr_detection_sum)/len(tpr_detection_sum)}')
    print(f'TPR_traceability: {sum(tpr_traceability_sum)/len(tpr_traceability_sum)}')
    #save metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gaussian Shading')
    parser.add_argument('--num', default=20, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--num_inversion_steps', default=None, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)
    parser.add_argument('--channel_copy', default=1, type=int)
    parser.add_argument('--hw_copy', default=8, type=int)
    parser.add_argument('--user_number', default=1000000, type=int)
    parser.add_argument('--fpr', default=0.000001, type=float)
    parser.add_argument('--output_path', default='./output/')
    parser.add_argument('--chacha', action='store_true', help='chacha20 for cipher')
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--dataset_path', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--model_path', default='sd2-community/stable-diffusion-2-1-base')

    # for image distortion
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--random_crop_ratio', default=None, type=float)
    parser.add_argument('--random_drop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--median_blur_k', default=None, type=int)
    parser.add_argument('--resize_ratio', default=None, type=float)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--sp_prob', default=None, type=float) 
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--path', type=str, default='attacked_images')



    args = parser.parse_args()

    if args.num_inversion_steps is None:
        args.num_inversion_steps = args.num_inference_steps

    main(args)
