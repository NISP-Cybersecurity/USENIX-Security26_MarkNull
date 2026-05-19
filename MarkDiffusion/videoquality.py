from evaluation.pipelines.video_quality_analysis import DirectVideoQualityAnalysisPipeline
from evaluation.dataset import VBenchDataset
from watermark.auto_watermark import AutoWatermark
from evaluation.tools.video_quality_analyzer import SubjectConsistencyAnalyzer,ImagingQualityAnalyzer
import torch
import os
import numpy as np
import cv2
from PIL import Image

def read_frames(frame_path):
    frames = sorted([
        f for f in os.listdir(frame_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
    ])
    
    frame_list = [Image.open(os.path.join(frame_path, f)).convert("RGB") for f in frames]
    
    return frame_list


imganalyzer = ImagingQualityAnalyzer()
path = "/home/nisp/Jie/security26/Attacked/UnMarker/SVD_VideoShield/"
score = 0
for i in os.listdir(path):
    frame_path = path + i
    video = read_frames(frame_path)
    score += imganalyzer.analyze(video)

print("avg score:", score/len(os.listdir(path)))
