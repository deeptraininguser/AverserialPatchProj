"""
visualization.py
================
Plotting helpers for best-patch presentation and training diagnostics.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms

from vae_utils import decode_latents


def plot_best_patch_summary(
    all_patches: torch.Tensor,
    performance: list[tuple[int, float]],
    patch_success_history: dict | None = None,
    patch_augmented_success_history: dict | None = None,
):
    """Create a 2×3 summary figure (best/2nd/3rd patch, bar chart,
    evolution, worst patch).
    """
    bp_idx, bp_ar = performance[0]
    bp = all_patches[bp_idx]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Patch image
    axes[0, 0].imshow(bp.permute(1, 2, 0).cpu().numpy())
    axes[0, 0].set_title(f"Best Patch #{bp_idx + 1}\nAug: {bp_ar:.1%}")
    axes[0, 0].axis("off")

    # Top-5 bar chart
    top5 = performance[:5]
    x = range(len(top5))
    aug_rates = [r for _, r in top5]
    axes[0, 1].bar(x, aug_rates, alpha=0.8)
    axes[0, 1].set_xticks(list(x))
    axes[0, 1].set_xticklabels([f"#{i + 1}" for i, _ in top5])
    axes[0, 1].set_ylabel("Aug success rate")
    axes[0, 1].set_title("Top 5 patches")

    # Evolution (if data available)
    if patch_augmented_success_history and bp_idx in patch_augmented_success_history:
        ep = list(range(len(patch_augmented_success_history[bp_idx])))
        axes[0, 2].plot(ep, patch_augmented_success_history[bp_idx], "s-", label="Aug")
        if patch_success_history and bp_idx in patch_success_history:
            axes[0, 2].plot(ep, patch_success_history[bp_idx], "o-", label="Clean")
        axes[0, 2].axhline(0.5, color="red", linestyle="--", alpha=0.7)
        axes[0, 2].set_title(f"Patch #{bp_idx + 1} evolution")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    else:
        axes[0, 2].axis("off")

    # 2nd / 3rd best
    for sub_idx, ax in zip([1, 2], [axes[1, 0], axes[1, 1]]):
        if sub_idx < len(performance):
            pi, pr = performance[sub_idx]
            axes[1, sub_idx - 1].imshow(all_patches[pi].permute(1, 2, 0).cpu().numpy())
            axes[1, sub_idx - 1].set_title(f"#{sub_idx + 1}: Patch #{pi + 1}\nAug: {pr:.1%}")
            axes[1, sub_idx - 1].axis("off")

    # Worst
    wi, wr = performance[-1]
    axes[1, 2].imshow(all_patches[wi].permute(1, 2, 0).cpu().numpy())
    axes[1, 2].set_title(f"Worst: Patch #{wi + 1}\nAug: {wr:.1%}")
    axes[1, 2].axis("off")

    plt.tight_layout()
    plt.show()
    return fig
