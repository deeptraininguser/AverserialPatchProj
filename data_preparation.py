"""
data_preparation.py
===================
Load captured multi-view frames, detect ArUco markers, compute
homographies, and build PyTorch DataLoaders for the adversarial training.
"""

from __future__ import annotations

import os
import glob
import pickle as pkl
from typing import Tuple, List

import cv2
import cv2.aruco as aruco
import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import tqdm as tqdm_mod

from consts import border_size, displayed_aruco_code


# ---- helpers ----------------------------------------------------------------

tt = torchvision.transforms.ToTensor()


def load_photometric_calibration(calibration_dir: str = "./photometric_calibrations"):
    """Load the most recent photometric calibration pickle.

    Returns
    -------
    data : dict   – full calibration dict
    height : int
    width : int
    """
    files = glob.glob(os.path.join(calibration_dir, "photometric_calibration_*.pkl"))
    files.sort(key=os.path.getmtime, reverse=True)
    path = files[0]
    print(f"Loading photometric calibration from {path}")
    with open(path, "rb") as f:
        data = pkl.load(f)
    return data, data["height"], data["width"]


# ---- ArUco border refinement ------------------------------------------------

def find_border_drop_point(gray: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Refine ArUco corner positions to the actual printed border edge."""
    sub = np.subtract
    add = np.add
    borders_drop_points = []
    for idx, operators in enumerate(
        ([sub, sub], [add, sub], [add, add], [sub, add])
    ):
        margin = 1
        a, b = int(c[idx][0]), int(c[idx][1])
        diag_idxs = np.arange(5)
        nca = operators[0](a, diag_idxs)
        ncb = operators[1](b, diag_idxs)
        nc = np.stack([nca, ncb], axis=1)
        diag_line_vals = gray[nc[:, 1], nc[:, 0]].astype(np.float32)
        diag_line_vals_diff = np.diff(diag_line_vals)
        if np.all(diag_line_vals_diff >= 0):
            borders_drop_points.append((nca[0], ncb[0]))
            continue
        first_neg = min(
            np.where(diag_line_vals_diff < 0)[0][0] + margin,
            len(diag_line_vals_diff) - 1,
        )
        borders_drop_points.append((nca[first_neg], ncb[first_neg]))
    return np.array(borders_drop_points)


# ---- Frame loading & homography computation ---------------------------------

def load_frames_and_homographies(
    cfg: dict,
    predict_raw,
    weights,
    orig_clases: torch.Tensor,
    height: int,
    width: int,
    on_remote: bool = False,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Scan the latest capture directory, detect ArUco markers, filter by
    class, and compute per-frame homographies.

    Returns (valid_frames, Hs).
    """
    aruco_dict_type = cv2.aruco.DICT_4X4_50
    aruco_dict = aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)

    caps_dir = cfg.get("capture", {}).get("captures_dir", "captures_frames_multiview")
    ls = os.listdir(f"./{caps_dir}")
    captures = sorted(
        [f for f in ls if f.startswith("captures_frames_multiview_")],
        key=lambda x: int(x.split("_")[-1]),
    )
    cap_dir = f"./{caps_dir}/{captures[-1]}"
    print(f"Using capture dir: {cap_dir}")

    valid_frame_paths = glob.glob(f"{cap_dir}/*.png")

    orig_img_corners = np.array(
        [
            [border_size, border_size],
            [width - border_size, border_size],
            [width - border_size, height - border_size],
            [border_size, height - border_size],
        ],
        dtype=np.float32,
    )

    valid_frames: list[np.ndarray] = []
    Hs: list[np.ndarray] = []
    found_aruco_count = 0
    outlier_classes: list = []

    for path in tqdm_mod.tqdm(valid_frame_paths, desc="Loading frames"):
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            continue
        found_aruco_count += 1

        with torch.no_grad():
            pr = predict_raw(tt(img).cuda().unsqueeze(0))
            cat = weights.meta["categories"][pr.argmax(1)]
            if not on_remote:
                pres_img = img.copy()
                aruco.drawDetectedMarkers(pres_img, corners, ids)
                cv2.putText(
                    pres_img,
                    f"pred: {cat} {pr.max().item():.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("pres", cv2.cvtColor(pres_img, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

        if (
            ids is not None
            and displayed_aruco_code in ids
            and pr.argmax(1) in orig_clases
        ):
            idx = np.where(ids.flatten() == displayed_aruco_code)[0][0]
            c = corners[idx][0]
            try:
                unbordred = find_border_drop_point(gray, c)
            except Exception:
                print("unbordering failed")
                continue
            H, _ = cv2.findHomography(orig_img_corners, unbordred, cv2.RANSAC)
            Hs.append(H)
            valid_frames.append(img)
        else:
            outlier_classes.append(pr.argmax(1))

    cv2.destroyAllWindows()
    print(
        f"Detected ArUco in {found_aruco_count}/{len(valid_frame_paths)} frames. "
        f"Valid: {len(valid_frames)}"
    )

    # Sub-sample
    max_frames = cfg.get("training", {}).get("max_frames", 50)
    random_idx = np.random.permutation(min(len(valid_frames), max_frames))
    valid_frames = [valid_frames[i] for i in random_idx]
    Hs = [Hs[i] for i in random_idx]
    print(f"Using {len(valid_frames)} frames for training.")
    return valid_frames, Hs


# ---- PyTorch Dataset ---------------------------------------------------------

class FramesDataset(Dataset):
    """Pairs of (frame_tensor, homography_matrix)."""

    def __init__(self, frames: list[np.ndarray], Hs: list[np.ndarray]):
        self.frames = frames
        self.Hs = Hs

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_tensor = tt(self.frames[idx])
        H = self.Hs[idx].astype(np.float32)
        return frame_tensor, H


def build_dataloaders(
    valid_frames: list[np.ndarray],
    Hs: list[np.ndarray],
    train_split: float = 0.8,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train / val / test DataLoaders."""
    ds = FramesDataset(valid_frames, Hs)
    total = len(ds)
    n_train = int(total * train_split)
    n_val = int(total * val_split)
    n_test = total - n_train - n_val
    print(f"Dataset split: train={n_train}, val={n_val}, test={n_test}")

    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        ds, [n_train, n_val, n_test]
    )
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=len(val_ds) or 1, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds) or 1, shuffle=False)
    return train_loader, val_loader, test_loader
