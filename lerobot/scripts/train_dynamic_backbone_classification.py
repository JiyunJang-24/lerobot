#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from einops import rearrange
import copy
import logging
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any
import json
import torch
import os
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer
from PIL import Image
import torch.nn.functional as F

import numpy as np
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
    save_stats,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy
from lerobot.common.policies.diffusion.modeling_diffusion import load_unimatch_backbone
from lerobot.common.datasets.lerobot_dataset import MultiLeRobotDataset, LeRobotDataset
from Depth_Anything_V2.depth_anything_v2.dpt import DepthAnythingV2
import matplotlib.pyplot as plt
import cv2
def save_and_show_depth_batch(depth_batch, prefix="img0_depth"):
    """
    depth_batch: (B, H, W) float tensor
    저장: {prefix}_{i:04d}.png  (TURBO 컬러맵, BGR)
    """
    depth_batch = depth_batch.detach().cpu()
    B, H, W = depth_batch.shape

    for i in range(B):
        d = depth_batch[i]

        # NaN/Inf 처리
        d = torch.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        # 샘플별 min-max 정규화 -> [0,1]
        d_min = float(d.min())
        d_max = float(d.max())
        d_norm = (d - d_min) / (max(d_max - d_min, 1e-6))

        # 8-bit로 변환
        d_u8 = (d_norm.numpy() * 255.0).astype(np.uint8)

        # 컬러맵 (BGR)
        d_color = cv2.applyColorMap(d_u8, cv2.COLORMAP_TURBO)

        # 파일 저장
        cv2.imwrite(f"{prefix}_{i:04d}.png", d_color)

        # 화면 표시 (matplotlib은 RGB)
        plt.figure(figsize=(4,4))
        plt.imshow(cv2.cvtColor(d_color, cv2.COLOR_BGR2RGB))
        plt.title(f"{prefix}[{i}]  min={d_min:.3f}, max={d_max:.3f}")
        plt.axis("off")
        plt.show()

@parser.wrap()
def train(cfg: TrainPipelineConfig):    
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)
    
    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)
    
    logging.info("Creating policy")
    if isinstance(dataset, MultiLeRobotDataset):
        ds_meta = dataset._datasets[0].meta
        ds_meta.stats = dataset.stats
        ds_meta.aug_stats = dataset.aug_stats
    else:
        ds_meta = dataset.meta
    model = load_unimatch_backbone(device, evaluation=False, num_dynamic_feature=cfg.policy.num_dynamic_feature, use_linear_prob = cfg.use_linear_prob)
    model.to(device)
    learnable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        learnable_params,
        # lr=getattr(cfg.optimizer, "lr", 3e-3),
        lr = 3e-3,
        weight_decay=getattr(cfg.optimizer, "weight_decay", 1e-2),
    )
    # 스케줄러가 필요하면 적절히 생성하거나 None 유지
    lr_scheduler = None
    grad_scaler = GradScaler(device.type, enabled=getattr(cfg.policy, "use_amp", False))
    criterion = torch.nn.CrossEntropyLoss()
    step = 0  # number of policy updates (forward + backward + optim)

    if cfg.resume: 
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    num_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in model.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        if isinstance(dataset, MultiLeRobotDataset):
            episode_data_index_list = []
            for ds in dataset._datasets:
                episode_data_index_list.append(copy.deepcopy(ds.episode_data_index))    # 一定要 deepcopy, .copy()都不行
            increase_num = 0
            for i in range(0, len(episode_data_index_list)):
                episode_data_index_list[i]["from"] += increase_num
                episode_data_index_list[i]["to"] += increase_num
                increase_num = episode_data_index_list[i]["to"][-1]
            episode_data_index = {
                "from": torch.cat([ds["from"] for ds in episode_data_index_list]),
                "to": torch.cat([ds["to"] for ds in episode_data_index_list]),
            }
        else:
            episode_data_index = dataset.episode_data_index
        sampler = EpisodeAwareSampler(
            episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)
    model.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
        "acc": AverageMeter("acc", ":.3f"),
        "conf": AverageMeter("conf", ":.3f")
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    use_amp = getattr(cfg.policy, "use_amp", False)
    if cfg.use_depth:
        model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
        encoder = 'vitb'
        depth_backbone = DepthAnythingV2(**model_configs[encoder])
        depth_backbone.load_state_dict(torch.load(f'/root/Desktop/workspace/yujin/shortcut-learning-in-grps/lerobot/Depth_Anything_V2/checkpoints/depth_anything_v2_{encoder}.pth'))
        depth_backbone = depth_backbone.to(device).eval()

    for _ in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)       # batch['observation.image'] 是 bz x history x 3 x w x h
        train_tracker.dataloading_s = time.perf_counter() - start_time
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)
        # ====== 입력/타깃 구성 ======
        if "dynamic.image" in batch:
            dyimg = batch["dynamic.image"]  # (B,S,N,C,H,W) or (1,B,S,N,C,H,W)
            dyact = batch["dynamic.action"] # (B,S,D)       or (1,B,S,D)

            # (카메라 축) 길이 1이면 제거. 멀티카메라면 명시 인덱스 선택 필요.
            if dyimg.ndim == 7:
                assert dyimg.size(0) == 1 and dyact.size(0) == 1, \
                    f"multi-camera 입력이면 cam idx를 선택하세요. got {tuple(dyimg.shape)}, {tuple(dyact.shape)}"
                dyimg = dyimg.squeeze(0)
                dyact = dyact.squeeze(0)

            # 이제 (B,S,N,C,H,W) / (B,S,D) 가정
            # 카메라(=시퀀스) 축을 앞으로: (S,B,...) 로 바꾸고, S와 B를 합쳐 한 번에 처리
            dyn_images  = rearrange(dyimg, "b s n c h w -> s b n c h w")  # (S,B,N,C,H,W)
            dyn_actions = rearrange(dyact, "b s d     -> s b d"    )      # (S,B,D)

            S, B, N, C, H, W = dyn_images.shape
            assert N == 2, f"dynamic.image의 프레임 수 N은 2여야 합니다. got N={N}"

            flat_imgs    = rearrange(dyn_images,  "s b n c h w -> (s b) n c h w")  # (S*B, 2, 3, H, W)
            flat_actions = rearrange(dyn_actions, "s b d       -> (s b) d")        # (S*B, 7)

            img0 = flat_imgs[:, 0]  # (S*B, 3, H, W)
            img1 = flat_imgs[:, 1]  # (S*B, 3, H, W)
            action = flat_actions   # (S*B, 7)

            img0_224 = F.interpolate(img0, size=(224,224), mode='bilinear', align_corners=False, antialias=True).clamp_(0, 1).mul_(255.0)
            img1_224 = F.interpolate(img1, size=(224,224), mode='bilinear', align_corners=False, antialias=True).clamp_(0, 1).mul_(255.0)
    

            if cfg.use_depth:
                img0_224_depth = F.interpolate(img0, size=(224,224), mode='bilinear', align_corners=False, antialias=True)
                img1_224_depth = F.interpolate(img1, size=(224,224), mode='bilinear', align_corners=False, antialias=True)
                with torch.no_grad():
                    img0_depth = depth_backbone(img0_224_depth)
                    img1_depth = depth_backbone(img1_224_depth)
                if cfg.viz:
                    save_and_show_depth_batch(img0_depth, prefix="img0_depth")
                    save_and_show_depth_batch(img1_depth, prefix="img1_depth")
            # 타깃도 S배 확장
            assert "angle_class" in batch, "Dataset이 item['angle_class'] (Long)을 제공해야 합니다."
            #target = batch["angle_class"].long().repeat_interleave(S)  # (S*B,)
            target = batch["angle_class"].long()

        else:
            assert ValueError("this is not applying")

        # 안전 체크
        assert img0.dim() == 4 and img1.dim() == 4, f"{img0.shape=}, {img1.shape=}"
        assert action.dim() == 2 and action.size(1) == 7, f"{action.shape=}"


        # ====== Forward / Loss ======
        model.train()
        with torch.autocast(device_type=device.type) if use_amp else nullcontext():
            if cfg.use_depth:
                logits = model(img0_224, img1_224, depth_0=img0_depth, depth_1=img1_depth, action=action, angle=target)  # (B, num_classes)
            else:
                logits = model(img0_224, img1_224, action=action)  # (B, num_classes)
            loss = criterion(logits, target)
            pred = logits.argmax(dim=1)
            acc = (pred == target).float().mean().item()
            conf = torch.softmax(logits, dim=1).max(dim=1).values.mean().item()
        train_tracker.acc = acc
        train_tracker.conf = conf    
        # ====== Backward ======
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.optimizer.grad_clip_norm, error_if_nonfinite=False)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step()
        # ====== Metrics ======
        train_tracker.loss = loss.item()
        train_tracker.grad_norm = grad_norm.item()
        train_tracker.lr = optimizer.param_groups[0]["lr"]
        train_tracker.update_s = time.perf_counter() - start_time
    

        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_logger.log_dict(train_tracker.to_dict(), step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            try: 
                ds_meta.stats['aug_stats'] = ds_meta.aug_stats
            except:
                print("ds_meta has not aug_stats")

            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                    "step": step,
                    "stats": ds_meta.stats,
                },
                os.path.join(checkpoint_dir, "checkpoint.pt"),
            )
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

    if eval_env:
        eval_env.close()
    logging.info("End of training")


if __name__ == "__main__":
    init_logging()
    train()
