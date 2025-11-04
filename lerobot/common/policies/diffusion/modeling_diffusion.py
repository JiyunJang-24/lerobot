#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
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
"""Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"

TODO(alexander-soare):
  - Remove reliance on diffusers for DDPMScheduler and LR scheduler.
"""

import math
from collections import deque
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn
from unimatch.unimatch import UniMatch, UniMatchVisionBackbone, Flow2LLaMAAdapter

from lerobot.common.constants import OBS_ENV, OBS_ROBOT
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize, Normalize_With_Aug, Unnormalize_With_Aug
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)

optical_backbone_cfg = {
    "feature_channels":128,
    "num_scales":2,
    "upsample_factor":4,
    "num_head":1,
    "ffn_dim_expansion":4,
    "num_transformer_layers":6,
    "reg_refine":None,
    "task":'flow',
    "resume_eval": "lerobot/unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth",
    "resume_train": "unimatch/pretrained/gmflow-scale2-regrefine6-mixdata-train320x576-4e7b215d.pth",
    "strict_resume": False,
}

def load_optical_backbone(device, evaluation, use_dynamic_common_feature=False, num_dynamic_feature=3):
    #define optical backbone class
    backbone_model = UniMatch(feature_channels=optical_backbone_cfg["feature_channels"],
                    num_scales=optical_backbone_cfg["num_scales"],
                    upsample_factor=optical_backbone_cfg["upsample_factor"],
                    num_head=optical_backbone_cfg["num_head"],
                    ffn_dim_expansion=optical_backbone_cfg["ffn_dim_expansion"],
                    num_transformer_layers=optical_backbone_cfg["num_transformer_layers"],
                    reg_refine=optical_backbone_cfg["reg_refine"],
                    task=optical_backbone_cfg["task"],
                    ).to(device)
    if evaluation==False:
        optical_backbone_cfg["resume"] = optical_backbone_cfg["resume_train"]
    else:
        optical_backbone_cfg["resume"] = optical_backbone_cfg["resume_eval"]

    if optical_backbone_cfg["resume"]:
        print('Load checkpoint: %s' % optical_backbone_cfg["resume"])
        checkpoint = torch.load(optical_backbone_cfg["resume"])
        backbone_model.load_state_dict(checkpoint['model'], strict=optical_backbone_cfg["strict_resume"])
    backbone_projector_model = UniMatchVisionBackbone(base_unimatch=backbone_model, fuse_multiscale=False, use_dynamic_common_feature=use_dynamic_common_feature, num_dynamic_feature=num_dynamic_feature)

    return backbone_projector_model

class DiffusionPolicy(PreTrainedPolicy):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    config_class = DiffusionConfig
    name = "diffusion"

    def __init__(
        self,
        config: DiffusionConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        dataset_aug_stats: dict[str, dict[str, Tensor]] | None = None,
        evaluation = False,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        if self.config.use_dynamic_feature:
            config.output_features['dynamic.action'] = config.output_features['action']
            dataset_stats['dynamic.action'] = dataset_stats['action']
        self.normalize_targets = Normalize_With_Aug(
            config.output_features, config.normalization_mapping, dataset_stats, dataset_aug_stats
        )
        self.unnormalize_outputs = Unnormalize_With_Aug(
            config.output_features, config.normalization_mapping, dataset_stats, dataset_aug_stats
        )
        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None
        self.config.evaluation = evaluation
        self.diffusion = DiffusionModel(config)

        self.reset()

    def get_optim_params(self) -> dict | list[nn.Parameter]:
        """
        Return only trainable params.
        - If train_dynamic_with_frozen_dp: only unet_dynamic and dynamic_encoder projector-like layers.
        - Else: everything with requires_grad=True (freeze 된 것은 자동 제외).
        """
        # 기본: requires_grad=True 모두 (freeze된 모듈은 자동 제외)
        return [p for p in self.diffusion.parameters() if p.requires_grad]

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
            "dynamic.image":  deque(maxlen=1),
            "dynamic.action": deque(maxlen=1)
        }
        if self.config.image_features:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], augmented_info: dict = None) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        # Note: It's important that this happens after stacking the images into a single key.
        
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            # stack n latest observations from the queue
            batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self.diffusion.generate_actions(batch)

            # TODO(rcadene): make above methods return output dictionary?
            if self.config.use_normalize_for_action:
                if augmented_info is not None:
                    actions = self.unnormalize_outputs({"action": actions, "augmented_info" : augmented_info})["action"]
                else:
                    actions = self.unnormalize_outputs({"action": actions})["action"]
            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        if self.config.use_normalize_for_action:
            batch = self.normalize_targets(batch)

        if self.config.train_dynamic_with_frozen_dp:
            loss = self.diffusion.compute_loss_dynamic(batch)
        else:
            loss = self.diffusion.compute_loss(batch)

        # no output_dict so returning None
        return loss, None


def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    """
    Factory for noise scheduler instances of the requested type. All kwargs are passed
    to the scheduler.
    """
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class DiffusionModel(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        # Build observation encoders (depending on which observations are provided).
        if self.config.use_robot_state:
            global_cond_dim = self.config.robot_state_feature.shape[0]
        else:
            global_cond_dim = 0
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config) for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images

        dynamic_cond_dim = 0

        if self.config.use_dynamic_feature:
            num_images = len(self.config.image_features)
            self.dynamic_encoder = load_optical_backbone(get_device_from_parameters(self), 
                                                        evaluation=self.config.evaluation, 
                                                        use_dynamic_common_feature=self.config.use_dynamic_common_feature,
                                                        num_dynamic_feature=self.config.num_dynamic_feature)
            if self.config.use_dynamic_common_feature:
                dynamic_cond_dim = self.dynamic_encoder.feature_dim * self.config.num_dynamic_feature
            else:
                dynamic_cond_dim = self.dynamic_encoder.feature_dim

        langauge_cond_dim = 0
        if self.config.use_language:
            self.language_encoder = TinyFrozenTextEncoder()
            langauge_cond_dim = self.language_encoder.out_dim

        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        if self.config.train_dynamic_with_frozen_dp:
            self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps + langauge_cond_dim)
            self.unet_dynamic = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps + dynamic_cond_dim + self.config.horizon * 7 + langauge_cond_dim) 
        else:
            self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps + dynamic_cond_dim + langauge_cond_dim)

        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    # ========= inference  ============
    def conditional_sample(
        self, batch_size: int, global_cond: Tensor | None = None, generator: torch.Generator | None = None
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
        sample = torch.randn(
            size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # Predict model output.
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            # Compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample

        return sample

    # ========= inference  ============
    def conditional_sample_dynamic(
        self, batch_size: int, global_cond: Tensor | None = None, global_cond_dynamic: Tensor | None = None, generator: torch.Generator | None = None
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
        sample = torch.randn(
            size=(batch_size, self.config.horizon, self.config.action_feature.shape[0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # Predict model output.
            base_model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            B, H, A = base_model_output.shape
            base_flat = base_model_output.reshape(B, H * A).detach()                     # (B, H*A), stop-grad

            # dtype mismatch 방지
            if base_flat.dtype != global_cond_dynamic.dtype:
                base_flat = base_flat.to(global_cond_dynamic.dtype)

            # concat: (B, D_dyn) + (B, H*A) -> (B, D_dyn + H*A)
            cond_dyn = torch.cat([global_cond_dynamic, base_flat], dim=-1)      # (B, D_dyn + H*A)
            dynamic_output = self.unet_dynamic(
                sample,
                torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device),
                global_cond=cond_dyn,
            )
            output = dynamic_output
            # Compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(output, t, sample, generator=generator).prev_sample

        return sample

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        import pdb; pdb.set_trace()
        batch_size, n_obs_steps = batch[OBS_ROBOT].shape[:2]
        global_cond_feats = []
        if self.config.use_language:
            task = batch['task'] # list, len(batch['task']) = batch_size
            language_feature = self.language_encoder(task)
            global_cond_feats.append(language_feature)

        if self.config.use_robot_state:
            global_cond_feats.append(batch[OBS_ROBOT])
        # Extract image features.
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                # Combine batch and sequence dims while rearranging to make the camera index dimension first.
                images_per_camera = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                # Separate batch and sequence dims back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                # Combine batch, sequence, and "which camera" dims before passing to shared encoder.
                img_features = self.rgb_encoder(
                    einops.rearrange(batch["observation.images"], "b s n ... -> (b s n) ...")
                )
                # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            #img_features는 batch_size * n_obs_steps * feature_dim
            global_cond_feats.append(img_features)
            if self.config.use_dynamic_feature:
                # Combine batch and sequence dims while rearranging to make the camera index dimension first.
                if batch["dynamic.image"].ndim == 7:
                    batch["dynamic.image"] = batch["dynamic.image"].squeeze(0)
                    batch["dynamic.action"] = batch["dynamic.action"].squeeze(0)
                dynamic_images = einops.rearrange(batch["dynamic.image"], "b s n ... -> s b n ...")
                dynamic_actions = einops.rearrange(batch["dynamic.action"], "b s n ... -> s b n ...")

                S, B, N, C, H, W = dynamic_images.shape  # N should be 2
                # 1) (S, B) → (S*B) 로 평탄화하여 한 번에 dynamic_encoder에 넣기
                flat_imgs    = einops.rearrange(dynamic_images, "s b n c h w -> (s b) n c h w")   # [(S*B), 2, 3, H, W]
                flat_actions = einops.rearrange(dynamic_actions, "s b d -> (s b) d")              # [(S*B), 7]

                o_t   = flat_imgs[:, 0]  # [(S*B), 3, H, W]
                o_tkp = flat_imgs[:, 1]  # [(S*B), 3, H, W]
                # 2) 인코더 한 번 호출 (기존 zip+cat 루프 제거)
                #    encoder 출력은 [(S*B), sfeat, f] 라고 가정 (sfeat == self.config.num_dynamic_feature)
                flat_feats = self.dynamic_encoder(o_t, o_tkp, flat_actions)  # [(S*B), f]
                if self.config.use_dynamic_common_feature:
                    dynamic_features = flat_feats #[B, f]
                else:
                    # 3) 원래 순서 보존하며 (S, B, sfeat, f) 로 복원
                    feats_SB = einops.rearrange(flat_feats, "(s b) f -> s b f", s=S, b=B)

                    # 4) 기존 코드에서 zip over S 후 torch.cat(dim=0) 했던 결과와 동일한 축 조합으로 재구성
                    #    즉, [S*B, sfeat, f] 로 다시 펴서 'dynamic_features_list'를 만든다 (순서 동일)
                    dynamic_features = einops.rearrange(feats_SB, "s b f -> b s f", b=B, s=self.config.num_dynamic_feature)

                # 3) 원래 순서 보존하며 (S, B, sfeat, f) 로 복원
                # feats_SB = einops.rearrange(flat_feats, "(s b) f -> s b f", s=S, b=B)

                # 4) 기존 코드에서 zip over S 후 torch.cat(dim=0) 했던 결과와 동일한 축 조합으로 재구성
                #    즉, [S*B, sfeat, f] 로 다시 펴서 'dynamic_features_list'를 만든다 (순서 동일)
                # dynamic_features = einops.rearrange(flat_feats, "s b f -> b s f", b=B, s=self.config.num_dynamic_feature)
                dynamic_features = flat_feats
                #dynamic_features는 batch_size * num_dynamic_feature * feature_dim
                global_cond_feats.append(dynamic_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV])

        feats_flat = [x.contiguous().flatten(start_dim=1) for x in global_cond_feats]  # [B, T_i*D]
        out = torch.cat(feats_flat, dim=-1)  # [B, sum_i(T_i*D)]
        # Concatenate features then flatten to (B, global_cond_dim).
        return out
    
    def _prepare_global_conditioning_frozen_dynamic(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch[OBS_ROBOT].shape[:2]
        global_cond_feats = []
        global_cond_feats_dynamic = []
        
        if self.config.use_language:
            task = batch['task'] # list, len(batch['task']) = batch_size
            language_feature = self.language_encoder(task)
            global_cond_feats.append(language_feature)

        if self.config.use_robot_state:
            global_cond_feats.append(batch[OBS_ROBOT])
        # Extract image features.
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                # Combine batch and sequence dims while rearranging to make the camera index dimension first.
                images_per_camera = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                # Separate batch and sequence dims back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                # Combine batch, sequence, and "which camera" dims before passing to shared encoder.
                img_features = self.rgb_encoder(
                    einops.rearrange(batch["observation.images"], "b s n ... -> (b s n) ...")
                )
                # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            #img_features는 batch_size * n_obs_steps * feature_dim
            global_cond_feats.append(img_features)
            global_cond_feats_dynamic.append(img_features)
            if self.config.use_dynamic_feature:
                # Combine batch and sequence dims while rearranging to make the camera index dimension first.
                # dynamic_images: [S, B, 2, 3, 256, 256]
                # dynamic_actions: [S, B, 7]
                if batch["dynamic.image"].ndim == 7:
                    batch["dynamic.image"] = batch["dynamic.image"].squeeze(0)
                    batch["dynamic.action"] = batch["dynamic.action"].squeeze(0)
                dynamic_images = einops.rearrange(batch["dynamic.image"], "b s n ... -> s b n ...")
                dynamic_actions = einops.rearrange(batch["dynamic.action"], "b s n ... -> s b n ...")

                S, B, N, C, H, W = dynamic_images.shape  # N should be 2
                # 1) (S, B) → (S*B) 로 평탄화하여 한 번에 dynamic_encoder에 넣기
                flat_imgs    = einops.rearrange(dynamic_images, "s b n c h w -> (s b) n c h w")   # [(S*B), 2, 3, H, W]
                flat_actions = einops.rearrange(dynamic_actions, "s b d -> (s b) d")              # [(S*B), 7]

                o_t   = flat_imgs[:, 0]  # [(S*B), 3, H, W]
                o_tkp = flat_imgs[:, 1]  # [(S*B), 3, H, W]

                # 2) 인코더 한 번 호출 (기존 zip+cat 루프 제거)
                #    encoder 출력은 [(S*B), sfeat, f] 라고 가정 (sfeat == self.config.num_dynamic_feature)
                flat_feats = self.dynamic_encoder(o_t, o_tkp, flat_actions)  # [(S*B), f]
                if self.config.use_dynamic_common_feature:
                    dynamic_features = flat_feats
                else:
                    # 3) 원래 순서 보존하며 (S, B, sfeat, f) 로 복원
                    feats_SB = einops.rearrange(flat_feats, "(s b) f -> s b f", s=S, b=B)

                    # 4) 기존 코드에서 zip over S 후 torch.cat(dim=0) 했던 결과와 동일한 축 조합으로 재구성
                    #    즉, [S*B, sfeat, f] 로 다시 펴서 'dynamic_features_list'를 만든다 (순서 동일)
                    dynamic_features = einops.rearrange(feats_SB, "s b f -> b s f", b=B, s=self.config.num_dynamic_feature)

                # dynamic_features: [batch_size, num_dynamic_feature, S * f]  (카메라/특징 s 축은 유지, n=S 는 feature 차원으로 흡수)
                global_cond_feats_dynamic.append(dynamic_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV])
        feats_flat = [x.contiguous().flatten(start_dim=1) for x in global_cond_feats]  # [B, T_i*D]
        out = torch.cat(feats_flat, dim=-1)  # [B, sum_i(T_i*D)]
        feats_flat_dynamic = [x.contiguous().flatten(start_dim=1) for x in global_cond_feats_dynamic]  # [B, T_i*D]
        out_dynamic = torch.cat(feats_flat_dynamic, dim=-1)  # [B, sum_i(T_i*D)]

        # Concatenate features then flatten to (B, global_cond_dim).
        return out, out_dynamic
    
    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        if self.config.train_dynamic_with_frozen_dp == True:
            global_cond, global_cond_dynamic = self._prepare_global_conditioning_frozen_dynamic(batch)
            actions = self.conditional_sample_dynamic(batch_size, global_cond=global_cond, global_cond_dynamic=global_cond_dynamic)
        else:
            global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)
            # run sampling
            actions = self.conditional_sample(batch_size, global_cond=global_cond)
        

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        assert "observation.images" in batch or "observation.environment_state" in batch
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch["action"]
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch["action"]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()
    
    def compute_loss_dynamic_one_more_noise(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # Input validation.
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        assert "observation.images" in batch or "observation.environment_state" in batch
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond, global_cond_dynamic = self._prepare_global_conditioning_frozen_dynamic(batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch["action"]
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

        # Sample noise to add to the trajectory.
        # Sample a random noising timestep for each item in the batch.
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        #TODO JY: eps, timesteps를 위에서 썼던걸 그대로 쓰는게 맞을지 확인
        noisy_trajectory = self.noise_scheduler.add_noise(pred, eps, timesteps)

        # Run the denoising network (that might denoise the trajectory, or attempt to predict the noise).
        pred = self.unet_dynamic(noisy_trajectory, timesteps, global_cond=global_cond_dynamic)
        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch["action"]
        else:
            raise ValueError(f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()

    def compute_loss_dynamic(self, batch: dict[str, Tensor]) -> Tensor:
        assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        assert "observation.images" in batch or "observation.environment_state" in batch
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # 준비: cond (base/dynamic), base는 freeze 전제
        global_cond, global_cond_dynamic = self._prepare_global_conditioning_frozen_dynamic(batch)

        x0 = batch["action"]                                # (B, H, A)
        eps = torch.randn_like(x0)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(x0.shape[0],),
            device=x0.device,
        ).long()

        # 1) 한 번만 forward diffusion: x_t
        x_t = self.noise_scheduler.add_noise(x0, eps, timesteps)

        # 2) base 출력 (gradient 차단)
        with torch.no_grad():
            base_out = self.unet(x_t, timesteps, global_cond=global_cond)
            # base가 epsilon/x0/v 중 뭘 내는지에 따라 분기 필요
        B, H, A = base_out.shape
        base_flat = base_out.reshape(B, H * A).detach()                     # (B, H*A), stop-grad

        # dtype mismatch 방지
        if base_flat.dtype != global_cond_dynamic.dtype:
            base_flat = base_flat.to(global_cond_dynamic.dtype)

        # concat: (B, D_dyn) + (B, H*A) -> (B, D_dyn + H*A)
        cond_dyn = torch.cat([global_cond_dynamic, base_flat], dim=-1)      # (B, D_dyn + H*A)
        # 3) dynamic 보정량 (같은 입력 x_t, 같은 t)
        dyn_out = self.unet_dynamic(x_t, timesteps, global_cond=cond_dyn)

        # 4) 타깃/결합 방식
        pred_type = self.config.prediction_type
        if pred_type == "epsilon":
            # 최종 ε̂ = ε_base + Δε
            pred = dyn_out
            target = eps
        elif pred_type == "sample":  # x0 파라미터화
            pred = dyn_out
            target = x0
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        # (선택) 보정량 크기 regularization: Δ를 불필요하게 크게 만들지 않도록
        if getattr(self.config, "lambda_residual_l2", 0.0) > 0:
            loss = loss + self.config.lambda_residual_l2 * (dyn_out ** 2)

        # 패딩 마스킹
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()

    def freeze_rgb_and_unet(self, freeze_bn_and_dropout: bool = True) -> dict:
        """
        Freeze rgb_encoder and unet so they don't get updated during training.
        Optionally put them in eval() to stop BatchNorm/Dropout updates.

        Args:
            freeze_bn_and_dropout (bool): If True, calls .eval() on modules to stop BN stats
                                        updates and disable Dropout during forward.

        Returns:
            dict: counts of frozen parameters for each module
                e.g., {"rgb_encoder": 12_345_678, "unet": 45_678_901, "total": 58_024_579}
        """
        frozen_counts = {"rgb_encoder": 0, "unet": 0, "total": 0}

        def _freeze_module(m: nn.Module) -> int:
            cnt = 0
            for p in m.parameters():
                if p.requires_grad:
                    p.requires_grad = False
                cnt += p.numel()
            if freeze_bn_and_dropout:
                m.eval()  # stop BN running stats & Dropout
            return cnt

        # 1) rgb_encoder (단일 or ModuleList 모두 지원)
        if hasattr(self, "rgb_encoder") and self.rgb_encoder is not None:
            if isinstance(self.rgb_encoder, nn.ModuleList):
                cnt = 0
                for enc in self.rgb_encoder:
                    cnt += _freeze_module(enc)
                frozen_counts["rgb_encoder"] = cnt
            else:
                frozen_counts["rgb_encoder"] = _freeze_module(self.rgb_encoder)

        # 2) unet (주의: unet_dynamic 은 요청에 없으므로 그대로 둠)
        if hasattr(self, "unet") and self.unet is not None:
            frozen_counts["unet"] = _freeze_module(self.unet)

        frozen_counts["total"] = frozen_counts["rgb_encoder"] + frozen_counts["unet"]
        return frozen_counts

class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://arxiv.org/pdf/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class DiffusionRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        # Set up optional preprocessing.
        
        if config.xyg_resize_shape is not None:
            self.resize = torchvision.transforms.Resize(config.xyg_resize_shape)
        else:
            self.resize = None

        if config.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                # raise ValueError(
                #     "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                # )
                print('WARNING: xyg do not care it just for trying')
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        # The dummy input should take the number of image channels from `config.image_features` and it should
        # use the height and width from `config.crop_shape` if it is provided, otherwise it should use the
        # height and width from `config.image_features`.

        # Note: we have a check in the config class to make sure all images have the same shape.
        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        if self.resize is not None:
            x = self.resize(x)
            
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(predicate(m) for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionConv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalUnet1d(nn.Module):
    """A 1D convolutional UNet with FiLM modulation for conditioning.

    Note: this removes local conditioning as compared to the original diffusion policy code.
    """

    def __init__(self, config: DiffusionConfig, global_cond_dim: int):
        super().__init__()

        self.config = config

        # Encoder for the diffusion timestep.
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # The FiLM conditioning dimension.
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim

        # In channels / out channels for each downsampling block in the Unet's encoder. For the decoder, we
        # just reverse these.
        in_out = [(config.action_feature.shape[0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        # Unet encoder.
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # Processing in the middle of the auto-encoder.
        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # Unet decoder.
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2, because it takes the encoder's skip connection as well
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Upsample as long as it is not the last block.
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.action_feature.shape[0], 1),
        )

    def forward(self, x: Tensor, timestep: Tensor | int, global_cond=None) -> Tensor:
        """
        Args:
            x: (B, T, input_dim) tensor for input to the Unet.
            timestep: (B,) tensor of (timestep_we_are_denoising_from - 1).
            global_cond: (B, global_cond_dim)
            output: (B, T, input_dim)
        Returns:
            (B, T, input_dim) diffusion model prediction.
        """
        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t")

        timesteps_embed = self.diffusion_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b d t -> b t d")
        return x


class DiffusionConditionalResidualBlock1d(nn.Module):
    """ResNet style 1D convolutional block with FiLM modulation for conditioning."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        # Set to True to do scale modulation with FiLM as well as bias modulation (defaults to False meaning
        # FiLM just modulates bias).
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()

        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation (https://arxiv.org/abs/1709.07871) outputs per-channel bias and (maybe) scale.
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # A final convolution for dimension matching the residual (if needed).
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        # Get condition embedding. Unsqueeze for broadcasting to `out`, resulting in (B, out_channels, 1).
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            # Treat the embedding as a list of scales and biases.
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            # Treat the embedding as biases.
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out


from transformers import AutoTokenizer, AutoModel

class TinyFrozenTextEncoder(nn.Module):
    """
    초경량 언어 인코더 (pretrained만 사용, 완전 동결)
    - 기본: prajjwal1/bert-tiny (L=2, H=128 수준)
    - 출력은 mean-pooling 후, 선형 투영으로 out_dim (기본 64)
    - forward는 no_grad + eval로 동작 (GPU/CPU 자동 할당)
    """
    def __init__(
        self,
        model_name: str = "prajjwal1/bert-tiny",
        tokenizer_name: str | None = None,  # None이면 model_name과 동일한 토크나이저
        out_dim: int = 128,
        max_length: int = 64,  # 짧은 프롬프트 가정: 속도/메모리 최적화
        normalize: bool = False,  # 필요하면 L2 정규화
    ):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name, use_fast=True)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()  # 평가 모드 고정
        for p in self.model.parameters():
            p.requires_grad = False  # 완전 동결
        self.out_dim = out_dim
        hidden = self.model.config.hidden_size
        self.max_length = max_length
        # self.proj = nn.Linear(hidden, out_dim) if out_dim != hidden else nn.Identity()
        self.normalize = normalize

    @staticmethod
    def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # (B, L, 1)
        summed = (last_hidden_state * mask).sum(dim=1)                  # (B, H)
        denom = mask.sum(dim=1).clamp(min=1e-9)                         # (B, 1)
        return summed / denom

    @torch.no_grad()
    def forward(self, texts: list[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        tok = {k: v.to(device) for k, v in tok.items()}
        out = self.model(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
        feat = self._mean_pool(out.last_hidden_state, tok["attention_mask"])  # (B, H)
        # feat = self.proj(pooled)  # (B, out_dim)
        if self.normalize:
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat