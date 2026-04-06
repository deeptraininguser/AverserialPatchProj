"""
vae_utils.py
============
Stable Diffusion VAE helpers – encode / decode latents for the
adversarial patch optimisation.
"""

import torch
from diffusers import StableDiffusionPipeline


_vae = None  # module-level cache


def load_vae(device: str = "cuda"):
    """Load the SD-v1.4 VAE and cache it."""
    global _vae
    if _vae is None:
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4"
        )
        _vae = pipe.vae.to(device).eval()
    return _vae


def get_vae():
    """Return the cached VAE (must call ``load_vae`` first)."""
    if _vae is None:
        raise RuntimeError("VAE not loaded. Call load_vae(device) first.")
    return _vae


def decode_latents_grad(latents):
    """Decode latents **with** gradient support (for training)."""
    vae = get_vae()
    latents = 1 / 0.18215 * latents
    imgs = vae.decode(latents).sample
    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    return imgs


def decode_latents(latents, device: str = "cuda"):
    """Decode latents **without** gradient (for evaluation)."""
    vae = get_vae()
    with torch.no_grad():
        with torch.amp.autocast(device):
            latents = 1 / 0.18215 * latents
            with torch.no_grad():
                imgs = vae.decode(latents).sample
            imgs = (imgs / 2 + 0.5).clamp(0, 1)
    return imgs


def encode_imgs(imgs, device: str = "cuda"):
    """Encode images into the latent space (no grad)."""
    vae = get_vae()
    with torch.no_grad():
        with torch.amp.autocast(device):
            imgs = 2 * imgs - 1
            posterior = vae.encode(imgs).latent_dist
            latents = posterior.sample() * 0.18215
    return latents
