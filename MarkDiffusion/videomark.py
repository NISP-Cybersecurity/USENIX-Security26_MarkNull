import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch
from watermark.auto_watermark import AutoWatermark
from utils.diffusion_config import DiffusionConfig
from diffusers import TextToVideoSDPipeline, StableVideoDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler, DDIMInverseScheduler, DDIMScheduler
from dotenv import load_dotenv
load_dotenv()
from tqdm import tqdm
import os
import numpy as np
import cv2
from PIL import Image
import random
import argparse



#model_path = os.getenv('T2V_MODEL_PATH')
model_path = "damo-vilab/text-to-video-ms-1.7b"
scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

pipe = TextToVideoSDPipeline.from_pretrained(
    model_path, 
    scheduler=scheduler,
    torch_dtype=torch.float16 if device == 'cuda' else torch.float32
).to(device)
diffusion_config = DiffusionConfig(
    pipe = pipe,
    scheduler = scheduler,
    device = device,
    image_size = (256, 256),        
    num_inference_steps = 25,       
    guidance_scale = 7.5,           
    gen_seed = 42,                  
    init_latents_seed= 42,         
    num_frames = 16,                
    inversion_type = "ddim"         
)


def save_video_frames(watermarked_video, save_dir, prefix="frame"):

    # 创建文件夹（如果不存在）
    os.makedirs(save_dir, exist_ok=True)

    # 保存每一帧
    for idx, frame in enumerate(watermarked_video):
        if not isinstance(frame, Image.Image):
            frame = Image.fromarray(frame)  # 如果是 numpy array，转换为 PIL Image
        filename = f"{prefix}_{idx:02d}.png"
        frame.save(os.path.join(save_dir, filename))

    print(f"Saved {len(watermarked_video)} frames to {save_dir}")

def read_frames(frame_path):
    frames = sorted([
        f for f in os.listdir(frame_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
    ])
    frame_list = [
        (im.resize((256, 256), resample=getattr(Image, "Resampling", Image).LANCZOS) if im.size != (256, 256) else im) for im in (Image.open(os.path.join(frame_path, f)).convert("RGB") for f in frames)]
    return frame_list



with open("./dataset/vbench/prompts_per_dimension/overall_consistency.txt", 'r', encoding='utf-8') as f:
    prompts = [line.strip() for line in f]
    

mywatermark = AutoWatermark.load('VideoMark', algorithm_config=f'config/VideoMark.json', diffusion_config=diffusion_config)

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ver', type=int, choices=[0, 1], required=True,
                    help='0: generate watermarked videos, 1: verify attacked videos')
parser.add_argument('--input_dir', type=str, default=None,
                    help='(ver=1) directory of attacked videos')
args = parser.parse_args()

test_num = 10
if args.ver == 0:
    # Generate watermarked videos
    all_bit_acc = 0
    for i in tqdm(range(test_num)):
        watermarked_video = mywatermark.generate_watermarked_media(input_data=prompts[i], num_frames=16)
        save_video_frames(watermarked_video, f"../Watermarked/SVD_VideoMark/video_{i}")

        detection_result = mywatermark.detect_watermark_in_media(watermarked_video, detector_type='bit_acc')
        all_bit_acc += detection_result['bit_acc']

    print("avg bit acc:", all_bit_acc / test_num)

elif args.ver == 1:
    # Verify attacked videos
    all_bit_acc = 0
    assert args.input_dir is not None, "Please specify --input_dir for verification."
    video_list = os.listdir(args.input_dir)

    for i in video_list:
        frame_path = os.path.join(args.input_dir, i)
        video = read_frames(frame_path)
        detection_result = mywatermark.detect_watermark_in_media(video, detector_type='bit_acc')
        all_bit_acc += detection_result['bit_acc']

    print("avg bit acc:", all_bit_acc / len(video_list))