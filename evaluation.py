"""
evaluation.py
=============
Post-training evaluation, ablation CSV export, and best-patch selection.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from vae_utils import decode_latents


def save_ablation_csvs(
    train_results: dict,
    cfg: dict,
    tracker=None,
):
    """Save epoch-level and per-patch metrics to CSV for ablation analysis."""
    results_dir = Path(cfg.get("results_dir", "./results")) / "ablation_tracking"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_name = train_results["model_name"]
    ts = train_results["timestamp"]
    to_rejuvenate = cfg.get("training", {}).get("rejuvenation", {}).get("enabled", False)
    fname_base = f"{model_name}_{ts}"

    # 1. Epoch metrics
    epoch_df = pd.DataFrame(train_results["epoch_metrics_history"])
    ep_path = results_dir / f"rejuv_{to_rejuvenate}_{fname_base}_epoch_metrics.csv"
    epoch_df.to_csv(ep_path, index=False)
    print(f"Saved epoch metrics → {ep_path}")

    # 2. Per-patch metrics
    num_patches = train_results["num_patches"]
    records = []
    for pidx in range(num_patches):
        for eidx, (c, a, r) in enumerate(zip(
            train_results["all_patch_clean_rates"][pidx],
            train_results["all_patch_aug_rates"][pidx],
            train_results["all_patch_robustness"][pidx],
        )):
            records.append({
                "patch_idx": pidx,
                "eval_epoch": eidx * 5,
                "clean_rate": c,
                "aug_rate": a,
                "robustness": r,
            })
    if records:
        patch_df = pd.DataFrame(records)
        pp_path = results_dir / f"rejuv_{to_rejuvenate}_{fname_base}_patch_metrics.csv"
        patch_df.to_csv(pp_path, index=False)
        print(f"Saved patch metrics → {pp_path}")
        if tracker:
            tracker.log_asset(str(pp_path))

    # 3. Summary
    summary = {
        "experiment_name": train_results.get("experiment_name", fname_base),
        "end_time": datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S"),
        "best_success_rate": train_results["best_success_rate"],
        "best_loss": train_results["best_loss"],
        "training_stopped_early": train_results["training_stopped"],
        "final_aug_weight": train_results["final_aug_weight"],
    }
    summary_df = pd.DataFrame([summary])
    sp_path = results_dir / f"rejuv_{to_rejuvenate}_{fname_base}_summary.csv"
    summary_df.to_csv(sp_path, index=False)
    print(f"Saved summary → {sp_path}")

    if tracker:
        tracker.log_asset(str(ep_path))
        tracker.log_asset(str(sp_path))


def save_top_patches(
    train_results: dict,
    height: int,
    width: int,
    device: str = "cuda",
):
    """Decode the latent batch and save the top-performing patches as PNG.

    Returns the directory path and the ``all_patches`` tensor.
    """
    import torchvision
    resizer = torchvision.transforms.Resize((height, width))
    latent_batch = train_results["latent_batch"].to(device)
    all_patches = resizer(decode_latents(latent_batch).float())
    num_patches = all_patches.shape[0]

    model_name = train_results["model_name"]
    ts = train_results["timestamp"]
    lsz = latent_batch.shape[-1]

    results_dir = train_results.get("results_dir", "./results")
    save_dir = f"{results_dir}/{model_name}_{lsz}x{lsz}_{ts}_top_patches/"
    os.makedirs(save_dir, exist_ok=True)

    to_pil = transforms.ToPILImage()
    aug_rates = train_results["all_patch_aug_rates"]
    performance = []
    for pidx in range(num_patches):
        last_ar = aug_rates[pidx][-1] if aug_rates[pidx] else 0
        performance.append((pidx, last_ar))
    performance.sort(key=lambda x: x[1], reverse=True)

    saved = []
    for pidx, rate in performance:
        if rate > 0.2:
            pil = to_pil(all_patches[pidx].cpu())
            path = os.path.join(save_dir, f"{pidx}_{str(rate)[:4]}.png")
            pil.save(path)
            saved.append(path)

    print(f"Saved {len(saved)} patches → {save_dir}")
    return save_dir, all_patches, performance
