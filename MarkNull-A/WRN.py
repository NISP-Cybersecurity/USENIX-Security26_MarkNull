import sys
sys.path.insert(0, '/home/nvidia/Jie/Security26')
import torch
import torch.nn as nn
import numpy as np
from torchvision import models
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity 
import math
import clip
import os, torchvision
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import kornia
from CosRemoval.inverse_initial_noise import load_image, decode_vae_with_grad, encode_vae_with_grad, ddim_inversion_to_noise, encode_vae
from diffusers import StableDiffusionPipeline
import logging
logging.getLogger("diffusers").setLevel(logging.ERROR)



def NLAS_loss(z_adv, z_T, epsilon=1e-5):
    z_adv_norm = F.normalize(z_adv.view(z_adv.shape[0], -1), p=2, dim=1)
    z_T_norm = F.normalize(z_T.view(z_T.shape[0], -1), p=2, dim=1)
    cos_sim = (z_adv_norm * z_T_norm).sum(dim=1)
    cos_sim = torch.clamp(cos_sim, -1 + epsilon, 1 - epsilon)
    theta = torch.acos(cos_sim)
    return ((theta - torch.pi/2) ** 2).mean()




def zt_loss(pipe, pred_img, target_img, device='cuda:1'):

    
    def get_imgs_zT(batch_images):
        with torch.no_grad():
            z0 = encode_vae(pipe.vae, batch_images, device=device, dtype=torch.float16,)
            
            zT, _, _ = ddim_inversion_to_noise(
                pipe,
                image=None,
                prompt=[""]*batch_images.shape[0],
                negative_prompt=[""]*batch_images.shape[0],
                num_inference_steps=50,
                guidance_scale=7.5,
                x0=z0,
            )
            return zT

    def get_imgs_z0(batch_images):
        z0 = encode_vae_with_grad(pipe.vae, batch_images, device=device, dtype=torch.float16)
        return z0
    
    z0_pred = get_imgs_z0(pred_img.to(pipe.device))
    zT_target = get_imgs_zT(target_img.to(pipe.device)) 
    #loss = contrastive_removal_loss(zT_pred, zT_target)
    #loss = F.cosine_similarity(zT_pred.view(zT_pred.shape[0], -1), zT_target.view(zT_target.shape[0], -1), dim=1).mean()
    #loss = - normalized_euclidean_distance(z0_pred.view(z0_pred.shape[0], -1), zT_target.view(zT_target.shape[0], -1)) + 2
    loss = geodesic_loss(z0_pred, zT_target)
    return loss.to(pred_img.device)
    
    
def to_vis_gray(x):
    x = x.detach().cpu().mean(dim=1, keepdim=True)  # [B,1,H,W]
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x


class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs 
                    for base_lr in self.base_lrs]
        else:
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            return [self.min_lr + 0.5 * (base_lr - self.min_lr) * 
                   (1 + math.cos(math.pi * progress))
                   for base_lr in self.base_lrs]






class MDTA(nn.Module):
    def __init__(self, channels, num_heads):
        super(MDTA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.qkv_conv = nn.Conv2d(channels * 3, channels * 3, kernel_size=3, padding=1, groups=channels * 3, bias=False)
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        b, c, h, w = x.shape
        #x = x.contiguous()
        q, k, v = self.qkv_conv(self.qkv(x)).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1).contiguous()) * self.temperature, dim=-1)
        out = self.project_out(torch.matmul(attn, v).reshape(b, -1, h, w))
        return out


class GDFN(nn.Module):
    def __init__(self, channels, expansion_factor):
        super(GDFN, self).__init__()

        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(hidden_channels * 2, hidden_channels * 2, kernel_size=3, padding=1,
                              groups=hidden_channels * 2, bias=False)
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(x)).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads, expansion_factor):
        super(TransformerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(channels)
        self.attn = MDTA(channels, num_heads)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = GDFN(channels, expansion_factor)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x + self.attn(self.norm1(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                          .contiguous().reshape(b, c, h, w))
        x = x + self.ffn(self.norm2(x.reshape(b, c, -1).transpose(-2, -1).contiguous()).transpose(-2, -1)
                         .contiguous().reshape(b, c, h, w))
        return x


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class Restormer(pl.LightningModule):
    def __init__(self, lr, epochs, num_blocks=[4, 6, 6, 8], num_heads=[1, 2, 4, 8], channels=[48, 96, 192, 384], num_refinement=4, expansion_factor=2.66):
        super(Restormer, self).__init__()
        
        self.lr = lr
        self.n_epoch = epochs
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg',normalize=True)


           
        self.ssim_loss = kornia.losses.SSIMLoss(window_size=11, reduction='mean')
        self.pipe = StableDiffusionPipeline.from_pretrained(
                "sd-legacy/stable-diffusion-v1-5",
                torch_dtype=torch.float16,
            ).to('cuda:1')
        
        self.embed_conv = nn.Conv2d(3, channels[0], kernel_size=3, padding=1, bias=False)

        self.encoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(
            num_ch, num_ah, expansion_factor) for _ in range(num_tb)]) for num_tb, num_ah, num_ch in
                                       zip(num_blocks, num_heads, channels)])
        # the number of down sample or up sample == the number of encoder - 1
        self.downs = nn.ModuleList([DownSample(num_ch) for num_ch in channels[:-1]])
        self.ups = nn.ModuleList([UpSample(num_ch) for num_ch in list(reversed(channels))[:-1]])
        # the number of reduce block == the number of decoder - 1
        self.reduces = nn.ModuleList([nn.Conv2d(channels[i], channels[i - 1], kernel_size=1, bias=False)
                                      for i in reversed(range(2, len(channels)))])
        # the number of decoder == the number of encoder - 1
        self.decoders = nn.ModuleList([nn.Sequential(*[TransformerBlock(channels[2], num_heads[2], expansion_factor)
                                                       for _ in range(num_blocks[2])])])
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[1], expansion_factor)
                                             for _ in range(num_blocks[1])]))
        # the channel of last one is not change
        self.decoders.append(nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                             for _ in range(num_blocks[0])]))

        self.refinement = nn.Sequential(*[TransformerBlock(channels[1], num_heads[0], expansion_factor)
                                          for _ in range(num_refinement)])
        self.output = nn.Conv2d(channels[1], 3, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        fo = self.embed_conv(x)

        out_enc1 = self.encoders[0](fo)
   
        out_enc2 = self.encoders[1](self.downs[0](out_enc1))
        out_enc3 = self.encoders[2](self.downs[1](out_enc2))
        out_enc4 = self.encoders[3](self.downs[2](out_enc3))

        #low_freq_tensor, mid_freq_tensor, high_freq_tensor = self.fft(out_enc4)
        
        # out_enc4 = self.msfm(low_freq_tensor, mid_freq_tensor, high_freq_tensor)
        
        
        out_dec3 = self.decoders[0](self.reduces[0](torch.cat([self.ups[0](out_enc4), out_enc3], dim=1)))
        out_dec2 = self.decoders[1](self.reduces[1](torch.cat([self.ups[1](out_dec3), out_enc2], dim=1)))
        fd = self.decoders[2](torch.cat([self.ups[2](out_dec2), out_enc1], dim=1))
        fr = self.refinement(fd)
        out = self.output(fr) + x
        return out
    

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=int(0.1 * self.n_epoch),  # 10%的epoch用于warmup
            total_epochs=self.n_epoch,
            min_lr=1e-6
        )
        return { 'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch', 'frequency': 1} }

    def _tensor_size(self, t):
        return t.size()[1]*t.size()[2]*t.size()[3]
    
    
    def _sanitize_01(self, x):
        x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0)
        if torch.isnan(x).any():
            print("Warning: NaN values found in output tensor.")
        return x.clamp(0.0, 1.0)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop
        _ , wm_img = batch
        output = self(wm_img)
        output = output.clamp(0.0, 1.0)

        lpips_loss = self.lpips(output.float(), wm_img.float())
        ssim_loss = self.ssim_loss(output.float(), wm_img.float())
        mse_loss = F.mse_loss(wm_img, output)
        zt_loss_value = zt_loss(self.pipe, output, wm_img)

        loss =  20 * mse_loss + lpips_loss + 5 * zt_loss_value + 0.5 * ssim_loss
                
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train_lpips', lpips_loss,prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train_mse', mse_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        #self.log('train_noise_loss', noise_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log('train_zt_loss', zt_loss_value, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train_ssim', ssim_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        #self.log('train_fft_loss', fft_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)

        return {'loss': loss}


    def validation_step(self, batch, batch_idx):
         # training_step defines the train loop
        _ , wm_img = batch
        output = self(wm_img)
        output = output.clamp(0.0, 1.0)
        lpips_loss = self.lpips(output.float(), wm_img.float())
        ssim_loss = self.ssim_loss(output.float(), wm_img.float())
     
        mse_loss = F.mse_loss(wm_img, output)
        zt_loss_value = zt_loss(self.pipe, output, wm_img)

        loss =  20 * mse_loss + lpips_loss + 4 * zt_loss_value + ssim_loss
        
        
        self.log('val_loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('val_lpips', lpips_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log('val_mse', mse_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        #self.log('val_noise_loss', noise_loss, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        #self.log('val_zt_loss', zt_loss_value, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    
  
        return loss
    
if __name__ == "__main__":
    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    model = Restormer(lr=1e-4, epochs=100).to(device)
    x = torch.randn(1, 3, 256, 256).to(device)
    y = model(x)
    print(y.shape)  
