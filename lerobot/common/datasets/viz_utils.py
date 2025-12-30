import h5py
import torch
import os
import numpy as np
import random
import re
import math
import glob
import json
from typing import List

import einops
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

def draw_clipped_arrow_fixed_head(
    img_bgr: np.ndarray,
    p0: tuple[int, int],
    p1: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int = 2,
    head_len_px: int = 10,   # ✅ 화살표 머리 길이(픽셀)
    head_w_px: int = 7,      # ✅ 화살표 머리 폭(픽셀)
):
    """
    p0->p1 화살표를
    - 이미지 경계로 clip해서 가능한 부분만 그림
    - 화살표 머리(head)는 픽셀 길이로 고정 (cv2.arrowedLine tipLength 미사용)
    """
    H, W = img_bgr.shape[:2]
    rect = (0, 0, W, H)

    ok, c0, c1 = cv2.clipLine(rect, p0, p1)
    if not ok:
        return False, None, None

    x0, y0 = c0
    x1, y1 = c1

    # shaft
    cv2.line(img_bgr, (x0, y0), (x1, y1), color, thickness, lineType=cv2.LINE_AA)

    # direction
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    L = (dx*dx + dy*dy) ** 0.5
    if L < 1e-6:
        cv2.circle(img_bgr, (x0, y0), max(1, thickness), color, -1)
        return True, (x0, y0), (x1, y1)

    ux, uy = dx / L, dy / L  # unit dir
    # head base center (move back from tip)
    hb_x = x1 - head_len_px * ux
    hb_y = y1 - head_len_px * uy
    # perpendicular unit
    px, py = -uy, ux

    left  = (int(round(hb_x + (head_w_px/2) * px)), int(round(hb_y + (head_w_px/2) * py)))
    right = (int(round(hb_x - (head_w_px/2) * px)), int(round(hb_y - (head_w_px/2) * py)))
    tip   = (int(round(x1)), int(round(y1)))

    tri = np.array([tip, left, right], dtype=np.int32)
    cv2.fillConvexPoly(img_bgr, tri, color, lineType=cv2.LINE_AA)

    return True, (x0, y0), (x1, y1)

def project_world_point_to_pixel_cam_to_world(
    K: np.ndarray,
    cam_to_world: np.ndarray,
    p_world: np.ndarray,
    eps: float = 1e-6,
):
    """
    MuJoCo convention:
      - camera looks along -Z
      - camera +Y is up
    Returns (u,v) in pixel coords (origin: top-left), or None if not projectable.
    """
    cam_T_world = np.linalg.inv(cam_to_world)

    pw = np.array([p_world[0], p_world[1], p_world[2], 1.0], dtype=np.float64)
    pc = cam_T_world @ pw
    X, Y, Z = float(pc[0]), float(pc[1]), float(pc[2])

    depth = -Z  # ✅ MuJoCo: in-front -> Z is negative
    if depth <= eps:
        return None

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    u = fx * (X / depth) + cx
    v = cy - fy * (Y / depth)   # ✅ Y up -> v down
    return (u, v)

def _extract_eef_world_pos(robot_eef_abs_poses):
    """
    robot_eef_abs_poses -> (3,) world position
    지원:
      - (4,4)
      - (>=3,) : 앞 3개를 xyz로 사용
    """
    if isinstance(robot_eef_abs_poses, torch.Tensor):
        arr = robot_eef_abs_poses.detach().cpu().numpy()
    else:
        arr = np.asarray(robot_eef_abs_poses)

    arr = arr.squeeze()

    if arr.shape == (4, 4):
        return arr[:3, 3].copy()

    if arr.shape[-1] >= 3:
        return arr[..., :3].copy()

    raise ValueError(f"Unsupported robot_eef_abs_poses shape: {arr.shape}")


def save_rgb_image(rgb_b3hw: torch.Tensor, path: str):
    """
    Save a normalized RGB tensor as an 8-bit PNG/JPG.

    Args:
        rgb_b3hw: torch.Tensor (B,3,H,W) or (3,H,W), normalized in [0,1]
        path: output file path (e.g., "/tmp/img.png")
    """
    x = rgb_b3hw.detach().float().cpu()

    # allow (3,H,W) or (B,3,H,W)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"Expected (B,3,H,W) or (3,H,W), got {tuple(x.shape)}")

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    x0 = x[0]  # (3,H,W)
    x0 = x0.clamp(0.0, 1.0)

    # (3,H,W) -> (H,W,3) uint8
    img_hwc = (x0.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)

    # save (cv2 expects BGR)
    img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)


def save_rgb_image(rgb_b3hw: torch.Tensor, path: str):
    """
    Save a normalized RGB tensor as an 8-bit PNG/JPG.

    Args:
        rgb_b3hw: torch.Tensor (B,3,H,W) or (3,H,W), normalized in [0,1]
        path: output file path (e.g., "/tmp/img.png")
    """
    x = rgb_b3hw.detach().float().cpu()

    # allow (3,H,W) or (B,3,H,W)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4 or x.shape[1] != 3:
        raise ValueError(f"Expected (B,3,H,W) or (3,H,W), got {tuple(x.shape)}")

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    x0 = x[0]  # (3,H,W)
    x0 = x0.clamp(0.0, 1.0)

    # (3,H,W) -> (H,W,3) uint8
    img_hwc = (x0.permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)

    # save (cv2 expects BGR)
    img_bgr = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_bgr)