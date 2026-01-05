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
    realworld: bool = False,
):
    """
    [MuJoCo convention]
      - camera looks along -Z
      - camera +Y is up
    Returns (u,v) in pixel coords (origin: top-left), or None if not projectable.

    [Realworld - Standard OpenCV Convention]
    - Camera looks along +Z
    - Camera +Y is down (Image coordinates)
    - Camera +X is right
    """
    cam_T_world = np.linalg.inv(cam_to_world)

    pw = np.array([p_world[0], p_world[1], p_world[2], 1.0], dtype=np.float64)
    pc = cam_T_world @ pw
    X, Y, Z = float(pc[0]), float(pc[1]), float(pc[2])

    if realworld:
        depth = Z
        if depth <= eps:
            return None
    else:
        depth = -Z
        if depth <= eps:
            return None

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])


    if realworld:
        u = fx * (X / Z) + cx
        v = fy * (Y / Z) + cy 
    else:
        u = fx * (X / Z) + cx
        v = cy - fy * (Y / Z)
    
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


def _get_motion_dynamics_basis(intrinsic_matrix, cam_to_world: np.ndarray):
        """
        cam_to_world: (4,4) camera pose matrix from pose_set (agentview).
                    This is T_{w<-c} (world_from_camera).

        Returns:
            torch.Tensor (3,2) on CUDA:
                [ [ux, vx],
                [uy, vy],
                [uz, vz] ]
            each row is a unit 2D direction vector in image (u,v) space corresponding to
            +X, +Y, +Z axes of the world/robot frame.
        """

        K = intrinsic_matrix
        cx, cy = float(K[0, 2]), float(K[1, 2])

        cam_to_world = np.asarray(cam_to_world, dtype=np.float32)
        assert cam_to_world.shape == (4, 4)

        # R_wc: world_from_cam rotation
        R_wc = cam_to_world[:3, :3]          # (3,3)
        # R_cw: cam_from_world rotation
        R_cw = R_wc.T

        # world/robot axes unit directions
        dirs_w = np.stack([
            np.array([1.0, 0.0, 0.0], dtype=np.float32),  # +X
            np.array([0.0, 1.0, 0.0], dtype=np.float32),  # +Y
            np.array([0.0, 0.0, 1.0], dtype=np.float32),  # +Z
        ], axis=0)  # (3,3)

        eps = 1e-8
        basis_uv = np.zeros((3, 2), dtype=np.float32)

        for i in range(3):
            d_w = dirs_w[i]                       # (3,)
            d_c = (R_cw @ d_w.reshape(3, 1)).reshape(3)  # (3,)

            # homogeneous image direction (vanishing point)
            p = (K @ d_c.reshape(3, 1)).reshape(3)  # (3,)
            denom = float(p[2])
            if abs(denom) < eps:
                denom = eps if denom >= 0 else -eps

            u = float(p[0] / denom)
            v = float(p[1] / denom)

            vec = np.array([u - cx, v - cy], dtype=np.float32)  # direction from principal point
            n = float(np.linalg.norm(vec))
            if n < eps:
                # degenerate fallback
                vec = p[:2].astype(np.float32)
                n = float(np.linalg.norm(vec)) + eps

            basis_uv[i] = vec / (n + eps)

        return torch.from_numpy(basis_uv).float()

def _make_motion_basis_axis_rgb_tensor_cam_to_world(
        rgb_tensor: torch.Tensor,                  # (3,H,W) or (B,3,H,W) in [0,1]
        motion_dynamics_basis: torch.Tensor,        # (6,) or (3,2)
        intrinsic_matrix: np.ndarray | torch.Tensor | None = None,  # (3,3) shared
        cam_to_world: np.ndarray | torch.Tensor | None = None,      # (4,4) shared
        robot_eef_abs_poses: np.ndarray | torch.Tensor | None = None,  # (7,) or (B,7)
        origin_robot: bool = False,
        origin_fallback: str = "pp",               # "pp" or "center"
        arrow_len: int = 60,
        line_thickness: int = 2,
        return_overlay: bool = False,
        overlay_alpha: float = 0.85,
        realworld: bool = False,
    ):
        """
        Returns:
        - unbatched input (3,H,W): (axis_tensor: (3,H,W), origin_xy: (ox,oy))
        - batched input (B,3,H,W): (axis_tensor: (B,3,H,W), origins: List[(ox,oy)])
        """

        def to_numpy(x):
            if x is None:
                return None
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)

        # --- normalize rgb to batched ---
        input_batched = (rgb_tensor.ndim == 4)
        if rgb_tensor.ndim == 3:
            rgb_b = rgb_tensor.unsqueeze(0)  # (1,3,H,W)
        elif rgb_tensor.ndim == 4:
            rgb_b = rgb_tensor              # (B,3,H,W)
        else:
            raise ValueError(f"rgb_tensor must be (3,H,W) or (B,3,H,W), got {tuple(rgb_tensor.shape)}")

        B, C, H, W = rgb_b.shape
        if C < 3:
            raise ValueError(f"rgb_tensor must have at least 3 channels, got C={C}")

        # --- basis (shared) -> (3,2) numpy ---
        if motion_dynamics_basis.ndim == 1:
            basis = motion_dynamics_basis.view(3, 2)
        else:
            basis = motion_dynamics_basis
        basis_np = basis.detach().float().cpu().numpy()  # (3,2)

        # --- shared K, c2w as numpy (for projection) ---
        K_np = to_numpy(intrinsic_matrix) if intrinsic_matrix is not None else None
        c2w_np = to_numpy(cam_to_world) if cam_to_world is not None else None

        if origin_robot:
            if K_np is None or c2w_np is None:
                # origin_robot=True인데 K/c2w가 없으면 투영 불가 -> fallback으로 처리
                pass
            else:
                if K_np.shape != (3, 3):
                    raise ValueError(f"intrinsic_matrix must be (3,3), got {K_np.shape}")
                if c2w_np.shape != (4, 4):
                    raise ValueError(f"cam_to_world must be (4,4), got {c2w_np.shape}")

        # --- eef poses normalize ---
        eef_np = None
        if robot_eef_abs_poses is not None:
            eef_np = to_numpy(robot_eef_abs_poses)
            # allow (7,) or (B,7)
            if eef_np.ndim == 1:
                if eef_np.shape[0] != 7:
                    raise ValueError(f"robot_eef_abs_poses must be (7,) got {eef_np.shape}")
                eef_np = np.broadcast_to(eef_np[None, :], (B, 7))
            elif eef_np.ndim == 2:
                if eef_np.shape != (B, 7):
                    raise ValueError(f"robot_eef_abs_poses must be (B,7) got {eef_np.shape}, expected {(B,7)}")
            else:
                raise ValueError(f"robot_eef_abs_poses must be (7,) or (B,7), got ndim={eef_np.ndim}")

        axis_list = []
        origins = []

        for b in range(B):
            rgb_i = rgb_b[b]
            H_i, W_i = int(rgb_i.shape[1]), int(rgb_i.shape[2])

            # 1) origin from EEF projection if available
            ox = oy = None
            if origin_robot and (c2w_np is not None) and (eef_np is not None) and (K_np is not None):
                p_world = eef_np[b, :3]  # (3,)
                uv = project_world_point_to_pixel_cam_to_world(K_np, c2w_np, p_worl, realworld=realworld)
                if uv is not None:
                    u, v = uv
                    ox = int(round(float(u))); oy = int(round(float(v)))
                    ox = max(0, min(W_i - 1, ox))
                    oy = max(0, min(H_i - 1, oy))

            # 2) fallback origin
            if ox is None or oy is None:
                if origin_fallback == "pp":
                    if K_np is None:
                        ox, oy = W_i // 2, H_i // 2
                    else:
                        ox, oy = int(round(float(K_np[0, 2]))), int(round(float(K_np[1, 2])))
                        ox = max(0, min(W_i - 1, ox))
                        oy = max(0, min(H_i - 1, oy))
                elif origin_fallback == "center":
                    ox, oy = W_i // 2, H_i // 2
                else:
                    raise ValueError("origin_fallback must be 'pp' or 'center'")

            # draw base
            if return_overlay:
                base_rgb = (rgb_i[:3].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
            else:
                base_rgb = np.zeros((H_i, W_i, 3), dtype=np.uint8)

            img_bgr = cv2.cvtColor(base_rgb, cv2.COLOR_RGB2BGR)

            colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR: x=red y=green z=blue
            origin_xy = (ox, oy)
            cv2.circle(img_bgr, origin_xy, 3, (255, 255, 255), -1)

            for i in range(3):
                du = float(basis_np[i, 0])
                if realworld:
                    dv = float(basis_np[i, 1]) 
                else:
                    dv = -float(basis_np[i, 1])  # image v-axis flip
                end_xy = (int(round(ox + arrow_len * du)),
                        int(round(oy + arrow_len * dv)))

                draw_clipped_arrow_fixed_head(
                    img_bgr,
                    origin_xy,
                    end_xy,
                    colors[i],
                    thickness=line_thickness,
                    head_len_px=8,
                    head_w_px=6,
                )

            out_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # uint8

            if return_overlay:
                rgb_u8 = (rgb_i[:3].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
                out_rgb = (overlay_alpha * out_rgb + (1.0 - overlay_alpha) * rgb_u8).astype(np.uint8)

            axis_tensor_i = torch.from_numpy(out_rgb).float().permute(2, 0, 1) / 255.0
            axis_tensor_i = axis_tensor_i.to(device=rgb_tensor.device)

            axis_list.append(axis_tensor_i)
            origins.append((ox, oy))

        axis_tensor = torch.stack(axis_list, dim=0)  # (B,3,H,W)

        if input_batched:
            return axis_tensor, origins
        else:
            return axis_tensor[0], origins[0]