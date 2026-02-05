import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import numpy as np
import sys
import kornia.losses
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from diffusers import StableDiffusionLatentUpscalePipeline
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image



def NLAS(z_adv, z_T, epsilon=1e-5):
    z_adv_norm = F.normalize(z_adv.view(z_adv.shape[0], -1), p=2, dim=1)
    z_T_norm = F.normalize(z_T.view(z_T.shape[0], -1), p=2, dim=1)
    cos_sim = (z_adv_norm * z_T_norm).sum(dim=1)
    cos_sim = torch.clamp(cos_sim, -1 + epsilon, 1 - epsilon)
    theta = torch.acos(cos_sim)
    return ((theta - torch.pi/2) ** 2).mean()

class OptAttackRemover:
    def __init__(
        self,
        pipe,
        device,
        dtype=torch.float16,
        epsilon=0.5,
        lr=0.01,
        num_steps=50,
        lambda_sim=3,
        lambda_lpips=1.0,
        lambda_ssim=1.5,
        lambda_mse=20,
        num_inference_steps=50,
        guidance_scale=7.5,
        verbose=False,
        SR=False,
    ):
        self.pipe = pipe
        self.device = device
        self.dtype = dtype

        self.epsilon = epsilon
        self.lr = lr
        self.num_steps = num_steps
        self.lambda_sim = lambda_sim
        self.lambda_lpips = lambda_lpips
        self.lambda_ssim = lambda_ssim
        self.lambda_mse = lambda_mse
        
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale

        self.verbose = verbose
        self.SR = SR

        print("Initializing Loss Metrics...")
        self.lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(self.device).eval()
        self.lpips_fn.requires_grad_(False)
        
        self.ssim_loss_fn = kornia.losses.SSIMLoss(window_size=11, reduction='mean')

        if self.SR:
            print("Loading Upscaler...")
            self.pipe_x2 = StableDiffusionLatentUpscalePipeline.from_pretrained(
                "stabilityai/sd-x2-latent-upscaler",
                torch_dtype=self.dtype
            ).to(self.device)
            try:
                self.pipe_x2.enable_xformers_memory_efficient_attention()
            except:
                pass
        else:
            self.pipe_x2 = None

    def _optimizer_based_attack(
        self,
        x_orig: torch.Tensor,
        z0_orig: torch.Tensor, 
        zT_orig: torch.Tensor
    ) -> torch.Tensor:
        
        zT_orig = zT_orig.detach()
        z0_orig = z0_orig.detach()
        x_orig = x_orig.detach()

        batch_size = z0_orig.shape[0]
        zT_orig_flat = zT_orig.view(batch_size, -1)

        z0_adv = z0_orig.clone().detach().to(dtype=torch.float32)
        z0_adv.requires_grad = True
        
        optimizer = torch.optim.Adam([z0_adv], lr=self.lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 80], gamma=0.3) 

        if self.verbose:
            print(f"Starting Optimizer Attack: steps={self.num_steps}, lr={self.lr}, eps={self.epsilon}")

        scaler = torch.cuda.amp.GradScaler()

        iterator = range(self.num_steps)
        if self.verbose:
            iterator = tqdm(iterator, desc="Attack Optimization")

        for i in iterator:
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=True):
                
                loss_contrast = NLAS(z0_adv, zT_orig)
                from .inverse_initial_noise import decode_vae_with_grad
                x_adv = decode_vae_with_grad(self.pipe.vae, z0_adv)
                
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
                
                loss_lpips = self.lpips_fn(x_adv, x_orig).mean()
                loss_ssim = self.ssim_loss_fn(x_adv, x_orig).mean() 
                mse_dist = F.mse_loss(x_adv, x_orig, reduction='none').view(batch_size, -1).mean()

                loss = (
                    self.lambda_sim * loss_contrast
                    + self.lambda_lpips * loss_lpips
                    + self.lambda_mse * mse_dist
                    + self.lambda_ssim * loss_ssim
                )

            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_([z0_adv], max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            with torch.no_grad():
                diff = z0_adv - z0_orig
                diff = torch.clamp(diff, -self.epsilon, self.epsilon)
                z0_adv.copy_(z0_orig + diff)

        return z0_adv.detach().to(dtype=self.dtype)

    def reconstruct_image(self, z0_adv):
        from .inverse_initial_noise import decode_vae
        
        with torch.no_grad():
            image_recon = decode_vae(self.pipe.vae, z0_adv)

        if self.SR and self.pipe_x2 is not None:
            upscaled_image = self.pipe_x2(
                prompt="",
                image=image_recon,
                num_inference_steps=20,
                guidance_scale=0.0,
                output_type="pt"
            ).images

            image_recon = F.interpolate(
                upscaled_image, 
                size=(512, 512), 
                mode='bicubic', 
                align_corners=False, 
                antialias=True
            )

        return image_recon

    def attack_removal(self, image, prompt="", negative_prompt=""):
        from .inverse_initial_noise import encode_vae, ddim_inversion_to_noise

        totensor = transforms.ToTensor()
        if isinstance(image, Image.Image):
            x_orig_tensor = totensor(image).unsqueeze(0).to(self.device, self.dtype)
        else:
            x_orig_tensor = image.to(self.device, self.dtype)
            if len(x_orig_tensor.shape) == 3:
                x_orig_tensor = x_orig_tensor.unsqueeze(0)

        z0_orig = encode_vae(self.pipe.vae, image, self.device, self.dtype)
        
        if self.verbose:
            print("--- Computing z_T via DDIM inversion ---")

        zT_orig, _, _ = ddim_inversion_to_noise(
            self.pipe,
            image=None,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            x0=z0_orig,
        )

        z0_adv = self._optimizer_based_attack(
            x_orig=x_orig_tensor,
            z0_orig=z0_orig,
            zT_orig=zT_orig
        )

        image_recon = self.reconstruct_image(z0_adv)
        
        return image_recon