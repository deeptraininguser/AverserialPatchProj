"""
training.py
===========
The main adversarial-patch training loop, extracted from
single_debug_v2_untargeted.ipynb.
"""

from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchvision
import tqdm as tqdm_mod

from vae_utils import decode_latents_grad, decode_latents
from augmentation import jitter, jitter_total_photo, warp, make_augmentor


# ---- Scheduler helper --------------------------------------------------------

def create_scheduler(optimizer, config: dict, total_steps: int):
    """Create a learning-rate scheduler from a config dict (or return None)."""
    if not config or not config.get("type"):
        return None
    stype = config["type"]
    if stype == "OneCycleLR":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.get("max_lr", 0.1),
            total_steps=total_steps,
            pct_start=config.get("pct_start", 0.0),
        )
    if stype == "StepLR":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.get("step_size", 50),
            gamma=config.get("gamma", 0.9),
        )
    if stype == "CosineAnnealingLR":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get("T_max", total_steps),
            eta_min=config.get("eta_min", 0.0001),
        )
    print(f"Unknown scheduler type '{stype}', using constant LR.")
    return None


def get_current_lr(optimizer, scheduler):
    if scheduler is not None:
        return scheduler.get_last_lr()[0]
    return optimizer.param_groups[0]["lr"]


# ---- Main training function --------------------------------------------------

def train_adversarial_patches(
    cfg: dict,
    train_loader,
    val_loader,
    valid_frames,
    height: int,
    width: int,
    orig_clases: torch.Tensor,
    predict_raw,
    predict_raw_dev,
    weights,
    augmentor_model,
    tracker,  # ExperimentTracker
    model_name: str,
    device: str = "cuda",
):
    """Run the full adversarial-patch optimisation loop.

    Returns
    -------
    dict with keys: best_latent, latent_batch, all_patches, metrics, …
    """
    from consts import latent_size, latent_batch_size

    tcfg = cfg.get("training", {})
    num_epochs = tcfg.get("num_epochs", 100)
    blend_ratio = tcfg.get("blend_ratio", 1.0)
    initial_lr = tcfg.get("learning_rate", 0.1)
    latent_var = tcfg.get("latent_variance", 0.8)
    grad_clip = tcfg.get("gradient_clip_norm", 3.0)
    aug_weight = tcfg.get("aug_weight_initial", 0.5)
    to_rejuvenate = tcfg.get("rejuvenation", {}).get("enabled", False)

    aug_prob = tcfg.get("augmentation", {})
    patch_jitter_p = aug_prob.get("patch_jitter_prob", 0.7)
    augmentor_p = aug_prob.get("augmentor_prob", 0.7)
    total_photo_p = aug_prob.get("total_photo_jitter_prob", 0.7)

    orig_loss_w = tcfg.get("orig_class_loss_weight", 5.0)
    target_loss_w = tcfg.get("target_class_loss_weight", 3.0)
    orig_vs_target = tcfg.get("orig_vs_target_ratio", 0.01)

    sched_cfg = tcfg.get("scheduler", {})
    use_scheduler = sched_cfg.get("enabled", False)

    es_cfg = tcfg.get("early_stopping", {})
    es_aug_thresh = es_cfg.get("aug_weight_threshold", 1.0)
    es_sr_thresh = es_cfg.get("success_rate_threshold", 0.9)

    resizer = torchvision.transforms.Resize((height, width))
    dst_shape = valid_frames[0].shape[:2]

    curr_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")

    # Target classes
    tc_list = tcfg.get("target_classes", [])
    target_classes = torch.tensor(tc_list, device=device) if tc_list else torch.tensor([], device=device)
    if target_classes.numel():
        print(f"Target classes: {[weights.meta['categories'][int(i)] for i in target_classes.cpu().numpy()]}")

    # ---- Latent init ---------------------------------------------------------
    latent_batch = torch.randn(
        (latent_batch_size, 4, latent_size, latent_size), device=device
    ) * latent_var
    latent_batch.requires_grad = True

    latent_opt = torch.optim.Adam([latent_batch], lr=initial_lr)
    scheduler = (
        create_scheduler(latent_opt, sched_cfg, num_epochs) if use_scheduler else None
    )

    orig_clases_np = orig_clases.cpu().numpy()
    num_patches = latent_batch.shape[0]

    # Augmentor
    augmentor = make_augmentor(augmentor_model, aug_weight, device)

    # Log experiment parameters
    tracker.log_parameters({
        "aug_weight": aug_weight,
        "num_epochs": num_epochs,
        "blend_ratio": blend_ratio,
        "initial_lr": initial_lr,
        "latent_batch_size": latent_batch_size,
        "latent_size": latent_size,
        "model_name": model_name,
        "scheduler": sched_cfg.get("type") if use_scheduler else "None",
        "to_rejuvenate": to_rejuvenate,
        "use_scheduler": use_scheduler,
    })

    # Set experiment name
    try:
        first_class_name = weights.meta["categories"][orig_clases[0].item()]
        experiment_name = f"{model_name}_{curr_ts}_{first_class_name}"
        tracker.set_name(experiment_name)
    except Exception:
        experiment_name = f"{model_name}_{curr_ts}"

    # ---- Tracking state ------------------------------------------------------
    losses = []
    success_rates = []
    best_loss = float("inf")
    best_latent = None
    best_success_rate = 0.0
    best_patch_idx: Optional[int] = None
    training_stopped = False
    patch_history: list = []
    epoch_metrics_history: list = []

    all_patch_clean_rates = {i: [] for i in range(num_patches)}
    all_patch_aug_rates = {i: [] for i in range(num_patches)}
    all_patch_robustness = {i: [] for i in range(num_patches)}

    results_dir = cfg.get("results_dir", "./results")
    os.makedirs(results_dir, exist_ok=True)

    # ==== TRAINING LOOP ======================================================
    epoch = 0
    while epoch < num_epochs:
        epoch_losses = []
        epoch_success_rates = []
        patch_success_history = {i: [] for i in range(num_patches)}
        patch_augmented_success_history = {i: [] for i in range(num_patches)}

        for batch_idx, (frames_batch, H_t_batch) in tqdm_mod.tqdm(
            enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}"
        ):
            latent_opt.zero_grad()
            frames_batch = frames_batch.to(device)
            H_t_batch = H_t_batch.to(device)

            adv_patch = resizer(decode_latents_grad(latent_batch).float())
            if len(patch_history) < 10000:
                patch_history.append(adv_patch.detach().cpu())

            # Patch augmentation
            adv_patch_aug = adv_patch
            if torch.rand(1).item() > (1 - patch_jitter_p):
                adv_patch_aug = jitter(adv_patch_aug)
            if torch.rand(1).item() > (1 - augmentor_p):
                adv_patch_aug = torch.stack(
                    [augmentor(x).to(device) for x in adv_patch_aug]
                )

            # Warp & blend
            w_mask = warp(adv_patch_aug * 0 + 1, H_t_batch, dst_shape)
            w_patch = warp(adv_patch_aug, H_t_batch, dst_shape)
            blended = (
                ((w_mask != 0) * -blend_ratio + 1) * frames_batch
                + w_patch * blend_ratio
            )
            blended = blended.squeeze(1)
            if torch.rand(1).item() > (1 - total_photo_p):
                blended = jitter_total_photo(blended)

            batch_frames = blended.view(-1, *blended.shape[1:])

            with torch.autocast(device_type=device):
                logits = predict_raw(batch_frames)
                if (logits != logits).any():
                    raise RuntimeError("NaN in logits")
                probs = torch.softmax(logits, dim=1)

            # ---- Loss --------------------------------------------------------
            orig_class_probs = probs[:, orig_clases]
            orig_loss = orig_loss_w * torch.log(
                orig_class_probs.sum(dim=1) + 1e-10
            ).mean()

            if target_classes.numel() > 0:
                target_probs = probs[:, target_classes]
                target_loss = -target_loss_w * torch.log(
                    target_probs.max(dim=1)[0] + 1e-10
                ).mean()
            else:
                target_loss = torch.tensor(0.0, device=device)

            if target_classes.numel() > 0:
                total_loss = orig_loss * orig_vs_target + target_loss * (1 - orig_vs_target)
            else:
                total_loss = orig_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_([latent_batch], max_norm=grad_clip)
            latent_opt.step()

            epoch_losses.append(total_loss.item())

            with torch.no_grad():
                preds = logits.argmax(dim=1)
                succ = sum(p.item() not in orig_clases_np for p in preds)
                sr = succ / len(preds)
                epoch_success_rates.append(sr)

                step = epoch * len(train_loader) + batch_idx
                tracker.log_metric("total_loss", total_loss.item(), step=step)
                tracker.log_metric("orig_loss", orig_loss.item(), step=step)
                tl_val = target_loss.item() if isinstance(target_loss, torch.Tensor) else target_loss
                tracker.log_metric("target_loss", tl_val, step=step)
                tracker.log_metric("batch_success_rate", sr, step=step)

        # ---- End of epoch bookkeeping ----------------------------------------
        if scheduler is not None:
            scheduler.step()

        avg_loss = np.mean(epoch_losses)
        avg_sr = np.percentile(epoch_success_rates, 90)
        losses.append(avg_loss)
        success_rates.append(avg_sr)

        cur_lr = get_current_lr(latent_opt, scheduler)
        epoch_metrics_history.append({
            "epoch": int(epoch),
            "avg_loss": float(avg_loss),
            "avg_success_rate": float(avg_sr),
            "learning_rate": float(cur_lr),
            "aug_weight": float(aug_weight),
        })
        tracker.log_metric("epoch_avg_loss", avg_loss, step=epoch)
        tracker.log_metric("epoch_avg_success", avg_sr, step=epoch)
        tracker.log_metric("learning_rate", cur_lr, step=epoch)
        tracker.log_metric("aug_weight", aug_weight, step=epoch)

        # ---- Per-patch evaluation (every 5 epochs or success > 0.3) ----------
        if epoch % 5 == 0 or avg_sr > 0.3:
            with torch.no_grad():
                all_patches = resizer(decode_latents(latent_batch).float())
                val_batch_limit = min(3, len(val_loader))

                for pidx in range(num_patches):
                    clean_ok = aug_ok = 0
                    total_clean = total_aug = 0
                    sp = all_patches[pidx: pidx + 1]

                    for vbi, (vf, vh) in enumerate(val_loader):
                        if vbi >= val_batch_limit:
                            break
                        vf, vh = vf.to(device), vh.to(device)

                        wm = warp(sp * 0 + 1, vh, dst_shape)
                        wp = warp(sp, vh, dst_shape)
                        cb = ((wm != 0) * -blend_ratio + 1) * vf + wp * blend_ratio
                        cb = cb.view(-1, *cb.shape[2:])
                        cl = predict_raw_dev(cb)
                        cp = cl.argmax(dim=1)

                        ap = jitter(sp)
                        ap = torch.stack([augmentor(x).to(device) for x in ap])
                        wma = warp(ap * 0 + 1, vh, dst_shape)
                        wpa = warp(ap, vh, dst_shape)
                        ab = ((wma != 0) * -blend_ratio + 1) * vf + wpa * blend_ratio
                        ab = ab.squeeze(0)
                        ab = jitter_total_photo(ab)
                        ab = ab.view(-1, *ab.shape[1:])
                        al = predict_raw_dev(ab)
                        apred = al.argmax(dim=1)

                        for p in cp:
                            total_clean += 1
                            if p.item() not in orig_clases_np:
                                clean_ok += 1
                        for p in apred:
                            total_aug += 1
                            if p.item() not in orig_clases_np:
                                aug_ok += 1

                    cr = clean_ok / total_clean if total_clean else 0
                    ar = aug_ok / total_aug if total_aug else 0
                    rr = (ar / cr) if cr > 0 else 0
                    patch_success_history[pidx].append(cr)
                    patch_augmented_success_history[pidx].append(ar)
                    all_patch_clean_rates[pidx].append(float(cr))
                    all_patch_aug_rates[pidx].append(float(ar))
                    all_patch_robustness[pidx].append(float(rr))

                    # Log patch images: clean patch + augmented patch
                    pimg = (sp[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    tracker.log_image(np.ascontiguousarray(pimg), name=f"Patch_{pidx + 1}", step=epoch)

                    aug_pimg = (ap[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                    tracker.log_image(np.ascontiguousarray(aug_pimg), name=f"Patch_{pidx + 1}_Augmented", step=epoch)

                    # Log augmented blended frames for the best patch (up to 3 samples)
                    if best_patch_idx is not None and pidx == best_patch_idx:
                        import cv2 as _cv2
                        categories = weights.meta["categories"]
                        for si in range(min(3, len(ab))):
                            bimg = (ab[si].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                            bimg = np.ascontiguousarray(bimg)
                            prob = 100 * torch.softmax(al[si], dim=0)[apred[si]]
                            res = categories[apred[si]]
                            _cv2.putText(bimg, f'Pred: {res}: {prob:.2f}%', (10, 30),
                                         _cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                            tracker.log_image(bimg, name=f"Augmented Patch idx {pidx + 1}", step=epoch)

            # -- Rank patches
            if patch_success_history[0]:
                lap = sorted(
                    [(i, patch_augmented_success_history[i][-1]) for i in range(num_patches)],
                    key=lambda x: x[1],
                    reverse=True,
                )
                best_patch_idx, best_patch_rate = lap[0]

                # Increase aug_weight adaptively
                top_aug_rate = torch.mean(torch.tensor([x[1] for x in lap[:1]])).item()
                if top_aug_rate > 0.7:
                    old_aw = aug_weight
                    if aug_weight < 0.7:
                        aug_weight += 0.1
                    elif aug_weight < 0.9:
                        aug_weight += 0.05
                    else:
                        aug_weight += 0.01
                    aug_weight = min(aug_weight, 1.0)
                    augmentor = make_augmentor(augmentor_model, aug_weight, device)
                    print(f"  [aug_weight] Top-patch aug SR = {top_aug_rate:.2%} > 0.70 → aug_weight {old_aw:.3f} → {aug_weight:.3f}")
                else:
                    print(f"  [aug_weight] Top-patch aug SR = {top_aug_rate:.2%} <= 0.70 → aug_weight unchanged ({aug_weight:.3f})")

                for rank, (pi, ar) in enumerate(lap[:5], 1):
                    cr = patch_success_history[pi][-1]
                    rr = (ar / cr) if cr > 0 else 0
                    tracker.log_metric(f"patch_{pi}_clean_rate", cr, step=epoch)
                    tracker.log_metric(f"patch_{pi}_aug_rate", ar, step=epoch)
                    tracker.log_metric(f"patch_{pi}_robustness", rr, step=epoch)

                # Log best / worst patch eval summary
                bp_idx, bp_ar = lap[0]
                wp_idx, wp_ar = lap[-1]
                tracker.log_metric("best_patch_clean_rate_eval", patch_success_history[bp_idx][-1], step=epoch)
                tracker.log_metric("best_patch_aug_rate_eval", bp_ar, step=epoch)
                tracker.log_metric("worst_patch_clean_rate_eval", patch_success_history[wp_idx][-1], step=epoch)
                tracker.log_metric("worst_patch_aug_rate_eval", wp_ar, step=epoch)

                # Save checkpoints
                if aug_weight >= 0.5 and best_patch_rate > 0.6:
                    bpl = latent_batch[best_patch_idx: best_patch_idx + 1].clone().detach()
                    sp_path = f"{results_dir}/successful_patch_{best_patch_idx + 1}_epoch_{epoch}_{model_name}_{curr_ts}.pt"
                    torch.save(bpl, sp_path)
                    tracker.log_asset(sp_path)
                    fb_path = f"{results_dir}/full_latent_batch_epoch_{epoch}_{model_name}_{curr_ts}.pt"
                    torch.save(latent_batch.clone().detach(), fb_path)
                    tracker.log_asset(fb_path)

                    if aug_weight >= es_aug_thresh and best_patch_rate >= es_sr_thresh:
                        print(f"Patch #{best_patch_idx + 1} hit {best_patch_rate:.1%} – stopping.")
                        training_stopped = True
                        break

                # Rejuvenation
                if to_rejuvenate and lap[0][1] > 0.2:
                    sstate = scheduler.state_dict() if scheduler else None
                    lb = latent_batch.clone().detach()
                    lb_best = lb[[x[0] for x in lap[:5]]]
                    lb[[x[0] for x in lap[-5:]]] = lb_best + torch.randn_like(lb_best) * 0.1
                    latent_batch = lb
                    latent_batch.requires_grad = True
                    latent_opt = torch.optim.Adam([latent_batch], lr=initial_lr)
                    if use_scheduler:
                        scheduler = create_scheduler(latent_opt, sched_cfg, num_epochs)
                        if sstate:
                            scheduler.load_state_dict(sstate)

        if training_stopped:
            break

        if avg_sr > best_success_rate or (avg_sr == best_success_rate and avg_loss < best_loss):
            best_success_rate = avg_sr
            best_loss = avg_loss
            best_latent = latent_batch.clone().detach()

        print(
            f"Epoch {epoch:3d}/{num_epochs} | Loss: {avg_loss:.3f} | "
            f"Success: {avg_sr:.1%} | LR: {cur_lr:.5f} | Aug: {aug_weight:.3f}"
        )
        tracker.log_metric("best_success_rate", best_success_rate)
        tracker.log_metric("best_loss", best_loss)
        epoch += 1

    # ---- Save final results --------------------------------------------------
    if best_latent is not None:
        final_path = f"{results_dir}/best_latent_final_{model_name}.pt"
        torch.save(best_latent, final_path)
        tracker.log_asset(final_path)

    return {
        "best_latent": best_latent,
        "latent_batch": latent_batch.detach(),
        "best_success_rate": best_success_rate,
        "best_loss": best_loss,
        "best_patch_idx": best_patch_idx,
        "losses": losses,
        "success_rates": success_rates,
        "epoch_metrics_history": epoch_metrics_history,
        "all_patch_clean_rates": all_patch_clean_rates,
        "all_patch_aug_rates": all_patch_aug_rates,
        "all_patch_robustness": all_patch_robustness,
        "model_name": model_name,
        "timestamp": curr_ts,
        "patch_history": patch_history,
        "training_stopped": training_stopped,
        "final_aug_weight": aug_weight,
        "num_patches": num_patches,
        "experiment_name": experiment_name,
    }
