#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import numpy as np
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    init_logging,
    has_method,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy
from lerobot.common.datasets.lerobot_dataset import MultiLeRobotDataset, LeRobotDataset

# ------------------------
# DDP helpers
# ------------------------
def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()

def get_rank():
    return dist.get_rank() if is_dist_avail_and_initialized() else 0

def is_main_process():
    return get_rank() == 0

def setup_distributed():
    """Initialize ProcessGroup from torchrun env (LOCAL_RANK, RANK, WORLD_SIZE)."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1 and not is_dist_avail_and_initialized():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://",
                                world_size=world_size, rank=rank)
    return local_rank, world_size, rank

def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.barrier()
        dist.destroy_process_group()

# ------------------------
# Train step
# ------------------------
def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    grad_scaler: GradScaler,
    lr_scheduler=None,
    use_amp: bool = False,
    lock=None,
    device: torch.device | None = None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy.train()
    with torch.autocast(device_type=device.type) if use_amp else nullcontext():
        loss, output_dict = policy.forward(batch)

    grad_scaler.scale(loss).backward()
    grad_scaler.unscale_(optimizer)

    grad_norm = torch.nn.utils.clip_grad_norm_(
        policy.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    with lock if lock is not None else nullcontext():
        grad_scaler.step(optimizer)
    grad_scaler.update()
    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(policy, "update"):
        policy.update()

    train_metrics.loss = loss.item()
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict

# ------------------------
# Main
# ------------------------
@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()

    # ===== DDP setup =====
    local_rank, world_size, global_rank = setup_distributed()
    distributed = world_size > 1

    # Logging / W&B: rank0만 자세히
    if is_main_process():
        logging.info(pformat(cfg.to_dict()))
    else:
        logging.getLogger().setLevel(logging.WARNING)

    if cfg.wandb.enable and cfg.wandb.project and is_main_process():
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        if is_main_process():
            logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed + global_rank)  # 각 rank마다 시드 살짝 섞기

    # ===== Device / Backend 설정 =====
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset") if is_main_process() else None
    dataset = make_dataset(cfg)

    # ===== Eval env (rank0 전용 권장) =====
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None and is_main_process():
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size)

    logging.info("Creating policy") if is_main_process() else None
    if isinstance(dataset, MultiLeRobotDataset):
        ds_meta = dataset._datasets[0].meta
        ds_meta.stats = dataset.stats
        ds_meta.aug_stats = dataset.aug_stats
    else:
        ds_meta = dataset.meta

    policy = make_policy(
        cfg=cfg.policy,
        ds_meta=ds_meta,
        libero_dataset=True,
    ).to(device)

    # DDP 래핑
    if distributed:
        policy = torch.nn.parallel.DistributedDataParallel(
            policy,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,  # 필요시 True
        )

    logging.info("Creating optimizer and scheduler") if is_main_process() else None
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)
    grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    step = 0
    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    # 파라미터 통계는 rank0에서만
    if is_main_process():
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # ===== Sampler / Dataloader =====
    if distributed:
        # 분산 기본 샘플러 사용 (에피소드 경계 엄격 보존이 필요하면 rank별 샤딩 전략으로 EpisodeAwareSampler를 감싸세요)
        sampler = DistributedSampler(dataset, shuffle=True, drop_last=False)
        shuffle = False
    else:
        if hasattr(cfg.policy, "drop_n_last_frames"):
            shuffle = False
            if isinstance(dataset, MultiLeRobotDataset):
                episode_data_index_list = []
                for ds in dataset._datasets:
                    episode_data_index_list.append(copy.deepcopy(ds.episode_data_index))
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
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    dl_iter = cycle(dataloader)
    policy.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }
    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    if is_main_process():
        logging.info("Start offline training on a fixed dataset")

    for _ in range(step, cfg.steps):
        # 분산 셔플 시드 주기적 갱신 (원하는 주기로)
        if distributed and isinstance(dataloader.sampler, DistributedSampler) and (step % 1000 == 0):
            dataloader.sampler.set_epoch(step)

        start_time = time.perf_counter()
        batch = next(dl_iter)
        print(batch["action"])
        train_tracker.dataloading_s = time.perf_counter() - start_time

        # 배치 텐서만 로컬 rank 디바이스로
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            grad_scaler=grad_scaler,
            lr_scheduler=lr_scheduler,
            use_amp=cfg.policy.use_amp,
            device=device,
        )

        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        # ----- LOG (rank0) -----
        if is_main_process() and is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        # ----- CKPT (rank0) -----
        if is_main_process() and cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            try:
                ds_meta.stats['aug_stats'] = ds_meta.aug_stats
            except Exception:
                print("ds_meta has not aug_stats")

            # DDP 래핑 해제하여 저장
            to_save = policy.module if isinstance(policy, torch.nn.parallel.DistributedDataParallel) else policy
            save_checkpoint(checkpoint_dir, step, cfg, to_save, optimizer, ds_meta.stats, lr_scheduler)
            update_last_checkpoint(checkpoint_dir)
            if wandb_logger:
                wandb_logger.log_policy(checkpoint_dir)

        # ----- EVAL (rank0 권장) -----
        if is_main_process() and cfg.env and is_eval_step:
            step_id = get_step_identifier(step, cfg.steps)
            logging.info(f"Eval policy at step {step}")
            with (
                torch.no_grad(),
                torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext(),
            ):
                eval_info = eval_policy(
                    eval_env,
                    to_save if 'to_save' in locals() else (policy.module if isinstance(policy, torch.nn.parallel.DistributedDataParallel) else policy),
                    cfg.eval.n_episodes,
                    videos_dir=cfg.output_dir / "eval" / f"videos_step_{step_id}",
                    max_episodes_rendered=4,
                    start_seed=cfg.seed,
                )

            eval_metrics = {
                "avg_sum_reward": AverageMeter("∑rwrd", ":.3f"),
                "pc_success": AverageMeter("success", ":.1f"),
                "eval_s": AverageMeter("eval_s", ":.3f"),
            }
            eval_tracker = MetricsTracker(
                cfg.batch_size, dataset.num_frames, dataset.num_episodes, eval_metrics, initial_step=step
            )
            eval_tracker.eval_s = eval_info["aggregated"].pop("eval_s")
            eval_tracker.avg_sum_reward = eval_info["aggregated"].pop("avg_sum_reward")
            eval_tracker.pc_success = eval_info["aggregated"].pop("pc_success")
            logging.info(eval_tracker)
            if wandb_logger:
                wandb_log_dict = {**eval_tracker.to_dict(), **eval_info}
                wandb_logger.log_dict(wandb_log_dict, step, mode="eval")
                if "video_paths" in eval_info and len(eval_info["video_paths"]) > 0:
                    wandb_logger.log_video(eval_info["video_paths"][0], step, mode="eval")

    if eval_env and is_main_process():
        eval_env.close()

    if is_main_process():
        logging.info("End of training")

    cleanup_distributed()


if __name__ == "__main__":
    init_logging()
    train()
