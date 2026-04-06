"""
augmentation.py
===============
Augmentation pipelines and the perspective-warp helper used during
adversarial patch training.
"""

from __future__ import annotations
import numpy as np
import torch
import torchvision.transforms as T
import kornia.geometry.transform


# ---- Augmentation transforms ------------------------------------------------

def random_blur(img: torch.Tensor) -> torch.Tensor:
    k_size = int(np.random.choice([3, 5, 7]))
    sigma = np.random.uniform(0.1, 1.0)
    return T.GaussianBlur(kernel_size=k_size, sigma=(sigma, sigma))(img)


jitter = T.Compose([
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    random_blur,
])

jitter_total_photo = T.Compose([
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
])

jitter_with_hue = T.Compose([
    T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
    T.GaussianBlur(kernel_size=5, sigma=(0.1, 1.0)),
])


# ---- Warp helper -------------------------------------------------------------

def warp(decoded_latents: torch.Tensor, H_t: torch.Tensor, dst_shape: tuple):
    """Warp a batch of patches onto frames using perspective homography.

    Parameters
    ----------
    decoded_latents : (N, C, H, W)
    H_t : (B, 3, 3)
    dst_shape : (height, width) of the destination frame

    Returns
    -------
    Tensor of shape (N, B, C, Hd, Wd)
    """
    warped_imgs = []
    for decoded_latent in decoded_latents:
        img = decoded_latent.unsqueeze(0).float().repeat(H_t.shape[0], 1, 1, 1)
        w = kornia.geometry.transform.warp_perspective(img, H_t, dst_shape)
        warped_imgs.append(w)
    return torch.stack(warped_imgs, dim=0)


# ---- Augmentor factory -------------------------------------------------------

def make_augmentor(augmentor_model, aug_weight: float, device: str = "cuda"):
    """Return a callable ``augmentor(x)`` that blends the photometric
    augmentor model output with the original patch.
    """
    def _augmentor(x):
        return augmentor_model(x).to(device) * aug_weight + x * (1 - aug_weight)
    return _augmentor
