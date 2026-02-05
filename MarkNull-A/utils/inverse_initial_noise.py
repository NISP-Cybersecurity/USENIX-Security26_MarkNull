import argparse, os, numpy as np, torch
from PIL import Image
from diffusers import StableDiffusionPipeline, DDIMScheduler, DDIMInverseScheduler
from argparse import Namespace
from tqdm import tqdm

# ---- utils ----
def load_image(path, size=512):
    img = Image.open(path).convert("RGB")
    #if size is not None:
    #img = img.resize((size, size), Image.BICUBIC)
    return img

@torch.no_grad()
def encode_vae(vae, image, device, dtype, assume_01=True):
    """
    image can be:
      - PIL / numpy: shape (H, W, 3), 0-255
      - torch.Tensor: shape (3, H, W) or (B, 3, H, W),
        values assumed to be in [0,1] if assume_01=True
    """
    if isinstance(image, torch.Tensor):
        # Ensure float tensor and move to target device
        x = image.to(device=device, dtype=dtype)

        # Normalize shape to [B, 3, H, W]
        if x.dim() == 3:
            # [3,H,W] -> [1,3,H,W]
            x = x.unsqueeze(0)
        elif x.dim() == 4:
            pass  # [B,3,H,W] use directly
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")

        # If currently [0,1], map to [-1,1]
        if assume_01:
            x = x.clamp(0, 1) * 2.0 - 1.0

    else:
        # Compatible with PIL.Image or numpy array: (H,W,3), 0-255
        arr = np.array(image).astype(np.float32) / 255.0  # [0,1]
        arr = (arr * 2.0 - 1.0).transpose(2, 0, 1)[None]  # -> [1,3,H,W]
        x = torch.from_numpy(arr).to(device=device, dtype=dtype)

    posterior = vae.encode(x).latent_dist
    latents = posterior.sample() * 0.18215  # SD 2.x scaling
    return latents


def encode_vae_with_grad(vae, image, device, dtype, assume_01=True):
    """
    image can be:
      - PIL / numpy: shape (H, W, 3), 0-255
      - torch.Tensor: shape (3, H, W) or (B, 3, H, W),
        values assumed to be in [0,1] if assume_01=True
    """
    if isinstance(image, torch.Tensor):
        # Ensure float tensor and move to target device
        x = image.to(device=device, dtype=dtype)

        # Normalize shape to [B, 3, H, W]
        if x.dim() == 3:
            # [3,H,W] -> [1,3,H,W]
            x = x.unsqueeze(0)
        elif x.dim() == 4:
            pass  # [B,3,H,W] use directly
        else:
            raise ValueError(f"Unsupported tensor shape: {x.shape}")

        # If currently [0,1], map to [-1,1]
        if assume_01:
            x = x.clamp(0, 1) * 2.0 - 1.0

    else:
        # Compatible with PIL.Image or numpy array: (H,W,3), 0-255
        arr = np.array(image).astype(np.float32) / 255.0  # [0,1]
        arr = (arr * 2.0 - 1.0).transpose(2, 0, 1)[None]  # -> [1,3,H,W]
        x = torch.from_numpy(arr).to(device=device, dtype=dtype)

    posterior = vae.encode(x).latent_dist
    latents = posterior.sample() * 0.18215  # SD 2.x scaling
    return latents

def combine_cfg(eps_uncond, eps_text, guidance_scale):
    return eps_uncond + guidance_scale * (eps_text - eps_uncond)

@torch.no_grad()
def decode_vae(vae, latents):
    latents = 1 / 0.18215 * latents
    recon = vae.decode(latents).sample
    recon = (recon / 2 + 0.5).clamp(0, 1)
    return recon    


def decode_vae_with_grad(vae, latents):
    latents = 1 / 0.18215 * latents
    recon = vae.decode(latents).sample
    recon = (recon / 2 + 0.5).clamp(0, 1)
    return recon    

# ---- inversion core ----
@torch.no_grad()
def ddim_inversion_to_noise(
    pipe: StableDiffusionPipeline,
    image: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    use_base: bool = False,
    x0 = None,
    ):
    """
    Returns:
        x_T (torch.Tensor): recovered initial noise, shape (1,4,H/8,W/8)
        latents_traj (list[torch.Tensor]): trajectory from x0 -> ... -> x_T
    """
    device = pipe.device
    dtype = pipe.unet.dtype

    # 1) Encode prompt embeddings (match CFG settings)
    # Internal diffusers method; returns concatenated (unconditional + conditional) text embeddings
    text_embeds = pipe._encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=(guidance_scale is not None and guidance_scale > 1.0),
        negative_prompt=negative_prompt if (guidance_scale and guidance_scale > 1.0) else None,
    )

    # 2) Encode image -> latents x0
    if x0 is None:
        x0 = encode_vae(pipe.vae, image, device, dtype)
    else:
        x0 = x0.to(device)

    # 3) Use DDIM inverse scheduler to push x0 -> x_T
    #    Make sure generation-time scheduler config is DDIM
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    inv_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    inv_scheduler.set_timesteps(num_inference_steps, device=device)

    xt = x0
    latents_traj = [xt.clone()]

    for i, t in enumerate(inv_scheduler.timesteps):
        # prepare model input (CFG: duplicate batch)
        if guidance_scale and guidance_scale > 1.0:
            model_in = torch.cat([xt, xt], dim=0)
        else:
            model_in = xt

        # predict epsilon
        noise_pred = pipe.unet(
            model_in, t, encoder_hidden_states=text_embeds, 
        ).sample

        if guidance_scale and guidance_scale > 1.0:
            eps_uncond, eps_text = noise_pred.chunk(2, dim=0)
            noise_pred = combine_cfg(eps_uncond, eps_text, guidance_scale)

        # inverse step: x_t -> x_{t+1} (going towards larger t)
        inv_out = inv_scheduler.step(noise_pred, t, xt)
        xt = inv_out.prev_sample  # here 'prev' means towards higher noise
        latents_traj.append(xt.clone())

    x_T = xt
    return x_T, latents_traj, x0


@torch.no_grad()
def unet_noise(
    pipe: StableDiffusionPipeline,
    image: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    use_base: bool = False,
    x0 = None,
    ):
    """
    Returns:
        x_T (torch.Tensor): recovered initial noise, shape (1,4,H/8,W/8)
        latents_traj (list[torch.Tensor]): trajectory from x0 -> ... -> x_T
    """
    device = pipe.device
    dtype = pipe.unet.dtype

    # 1) Encode prompt embeddings (match CFG settings)
    # Internal diffusers method; returns concatenated (unconditional + conditional) text embeddings
    text_embeds = pipe._encode_prompt(
        prompt=prompt,
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=(guidance_scale is not None and guidance_scale > 1.0),
        negative_prompt=negative_prompt if (guidance_scale and guidance_scale > 1.0) else None,
    )

    noise = 0
    # 2) Encode image -> latents x0
    if x0 is None:
        x0 = encode_vae(pipe.vae, image, device, dtype)
    else:
        x0 = x0.to(device)

    # 3) Use DDIM inverse scheduler to push x0 -> x_T
    #    Make sure generation-time scheduler config is DDIM
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    inv_scheduler = DDIMInverseScheduler.from_config(pipe.scheduler.config)
    inv_scheduler.set_timesteps(num_inference_steps, device=device)

    xt = x0
    latents_traj = [xt.clone()]

    for i, t in enumerate(inv_scheduler.timesteps):
        # prepare model input (CFG: duplicate batch)
        if guidance_scale and guidance_scale > 1.0:
            model_in = torch.cat([xt, xt], dim=0)
        else:
            model_in = xt

        # predict epsilon
        noise_pred = pipe.unet(
            model_in, t, encoder_hidden_states=text_embeds, 
        ).sample

        if guidance_scale and guidance_scale > 1.0:
            eps_uncond, eps_text = noise_pred.chunk(2, dim=0)
            noise_pred = combine_cfg(eps_uncond, eps_text, guidance_scale)

        noise +=  noise_pred 
    return noise

def save_noise(noise, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(noise, out_path + ".pt")
    np.save(out_path + ".npy", noise.detach().cpu().numpy())
    print(f"[+] saved noise to: {out_path}.npy")
    print(f"    stats: mean={noise.mean().item():.4f}, std={noise.std().item():.4f}, shape={tuple(noise.shape)}")

@torch.no_grad()
def ddim_zT_to_z0(pipe, zT, prompt= "",
                  num_inference_steps=50, guidance_scale=7.5, eta=0.0):
    """
    pipe: diffusers.StableDiffusionPipeline (or compatible)
    zT:   [B, 4, 64, 64] initial latent (noise or latent at time t)
    prompt_embeds: [B, L, D]
    neg_prompt_embeds: [B, L, D] (optional, for CFG)
    """
    prompt_embeds, neg_prompt_embeds = pipe.encode_prompt(
            prompt=prompt,
            device=pipe.device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,   # negative required only for CFG
            negative_prompt=""                  # pass empty string if no negative prompt
            )   
    device = zT.device
    scheduler = pipe.scheduler  # Ensure it is DDIMScheduler or compatible
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Set eta (some schedulers set in set_timesteps, others in step)
    if hasattr(scheduler, "eta"):
        scheduler.eta = eta

    latents = zT

    use_cfg = (neg_prompt_embeds is not None) and (guidance_scale is not None) and (guidance_scale > 1.0)
    if use_cfg:
        # Concatenate to 2B for classifier-free guidance
        embeds = torch.cat([neg_prompt_embeds, prompt_embeds], dim=0)
    else:
        embeds = prompt_embeds

    for t in scheduler.timesteps:
        if use_cfg:
            latent_in = torch.cat([latents, latents], dim=0)  # [2B,4,64,64]
        else:
            latent_in = latents

        # UNet predicts noise epsilon
        noise_pred = pipe.unet(latent_in, t, encoder_hidden_states=embeds).sample

        # CFG: epsilon = eps_uncond + s*(eps_cond - eps_uncond)
        if use_cfg:
            noise_uncond, noise_text = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_text - noise_uncond)

        # DDIM one-step update: get z_{t-1}
        # Note: step parameter names differ slightly across diffusers versions
        out = scheduler.step(noise_pred, t, latents, eta=eta) if "eta" in scheduler.step.__code__.co_varnames \
              else scheduler.step(noise_pred, t, latents)
        latents = out.prev_sample

    z0 = latents
    return z0



# if __name__ == "__main__":
    
#      wm_method = "StableSignature"
#      image_path = f"/home/nisp/Jie/Sherlock/SDP/{wm_method}/train/watermarked"
#      #image_path = "/home/nisp/Jie/Sherlock/SDP/SDP_SD2.1_clean/train/no_watermark"
#      image_list = os.listdir(image_path)
    
#      Z_T_list = []
#      Z_target_list = []
#      Z_0_list = []

#      Z_T_save_path = 
#      Z_target_save_path = 
#      Z_0_save_path = 
    
#      os.makedirs(os.path.dirname(Z_T_save_path), exist_ok=True)
    
    

    # for image_name in tqdm(image_list, desc="Processing images"):
    #      args.image = os.path.join(image_path, image_name)
    #      Z_T, Z_target, Z_0 = main(args)
    #      Z_T_list.append(Z_T)
    #      Z_target_list.append(Z_target)
    #      Z_0_list.append(Z_0)
        
    #      Z_T_array = torch.cat(Z_T_list, dim=0).cpu().numpy()
    #      Z_target_array = torch.cat(Z_target_list, dim=0).cpu().numpy()
    #      Z_0_array = torch.cat(Z_0_list, dim=0).cpu().numpy()
    #      #print(f"[i] current {image_name}: Z_T shape: {Z_T_array.shape}, Z_target shape: {Z_target_array.shape}, Z_0 shape: {Z_0_array.shape}")
        
    # np.save(Z_T_save_path, Z_T_array)
    # np.save(Z_target_save_path, Z_target_array)
    # np.save(Z_0_save_path, Z_0_array)