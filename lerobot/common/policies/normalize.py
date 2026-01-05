#!/usr/bin/env python

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
import numpy as np
import torch
from torch import Tensor, nn

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature


def create_stats_buffers(
    features: dict[str, PolicyFeature],
    norm_map: dict[str, NormalizationMode],
    stats: dict[str, dict[str, Tensor]] | None = None,
) -> dict[str, dict[str, nn.ParameterDict]]:
    """
    Create buffers per modality (e.g. "observation.image", "action") containing their mean, std, min, max
    statistics.

    Args: (see Normalize and Unnormalize)

    Returns:
        dict: A dictionary where keys are modalities and values are `nn.ParameterDict` containing
            `nn.Parameters` set to `requires_grad=False`, suitable to not be updated during backpropagation.
    """
    stats_buffers = {}
    for key, ft in features.items():
        norm_mode = norm_map.get(ft.type, NormalizationMode.IDENTITY)
        if norm_mode is NormalizationMode.IDENTITY:
            continue

        assert isinstance(norm_mode, NormalizationMode)

        shape = tuple(ft.shape)

        if ft.type is FeatureType.VISUAL:
            # sanity checks
            assert len(shape) == 3, f"number of dimensions of {key} != 3 ({shape=}"
            c, h, w = shape
            assert c < h and c < w, f"{key} is not channel first ({shape=})"
            # override image shape to be invariant to height and width
            shape = (c, 1, 1)

        # Note: we initialize mean, std, min, max to infinity. They should be overwritten
        # downstream by `stats` or `policy.load_state_dict`, as expected. During forward,
        # we assert they are not infinity anymore.

        buffer = {}
        if norm_mode is NormalizationMode.MEAN_STD:
            mean = torch.ones(shape, dtype=torch.float32) * torch.inf
            std = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "mean": nn.Parameter(mean, requires_grad=False),
                    "std": nn.Parameter(std, requires_grad=False),
                }
            )
        elif norm_mode is NormalizationMode.MIN_MAX:
            min = torch.ones(shape, dtype=torch.float32) * torch.inf
            max = torch.ones(shape, dtype=torch.float32) * torch.inf
            buffer = nn.ParameterDict(
                {
                    "min": nn.Parameter(min, requires_grad=False),
                    "max": nn.Parameter(max, requires_grad=False),
                }
            )

        # TODO(aliberts, rcadene): harmonize this to only use one framework (np or torch)
        if stats:
            if isinstance(stats[key]["mean"], np.ndarray):
                if norm_mode is NormalizationMode.MEAN_STD:
                    buffer["mean"].data = torch.from_numpy(stats[key]["mean"]).to(dtype=torch.float32)
                    buffer["std"].data = torch.from_numpy(stats[key]["std"]).to(dtype=torch.float32)
                elif norm_mode is NormalizationMode.MIN_MAX:
                    buffer["min"].data = torch.from_numpy(stats[key]["min"]).to(dtype=torch.float32)
                    buffer["max"].data = torch.from_numpy(stats[key]["max"]).to(dtype=torch.float32)
            elif isinstance(stats[key]["mean"], torch.Tensor):
                # Note: The clone is needed to make sure that the logic in save_pretrained doesn't see duplicated
                # tensors anywhere (for example, when we use the same stats for normalization and
                # unnormalization). See the logic here
                # https://github.com/huggingface/safetensors/blob/079781fd0dc455ba0fe851e2b4507c33d0c0d407/bindings/python/py_src/safetensors/torch.py#L97.
                if norm_mode is NormalizationMode.MEAN_STD:
                    buffer["mean"].data = stats[key]["mean"].clone().to(dtype=torch.float32)
                    buffer["std"].data = stats[key]["std"].clone().to(dtype=torch.float32)
                elif norm_mode is NormalizationMode.MIN_MAX:
                    buffer["min"].data = stats[key]["min"].clone().to(dtype=torch.float32)
                    buffer["max"].data = stats[key]["max"].clone().to(dtype=torch.float32)
            else:
                type_ = type(stats[key]["mean"])
                raise ValueError(f"np.ndarray or torch.Tensor expected, but type is '{type_}' instead.")

        stats_buffers[key] = buffer
    return stats_buffers


def _no_stats_error_str(name: str) -> str:
    return (
        f"`{name}` is infinity. You should either initialize with `stats` as an argument, or use a "
        "pretrained model."
    )


class Normalize(nn.Module):
    """Normalizes data (e.g. "observation.image") for more stable and faster convergence during training."""

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            shapes (dict): A dictionary where keys are input modalities (e.g. "observation.image") and values
            are their shapes (e.g. `[3,96,96]`]). These shapes are used to create the tensor buffer containing
            mean, std, min, max statistics. If the provided `shapes` contain keys related to images, the shape
            is adjusted to be invariant to height and width, assuming a channel-first (c, h, w) format.
            modes (dict): A dictionary where keys are output modalities (e.g. "observation.image") and values
                are their normalization modes among:
                    - "mean_std": subtract the mean and divide by standard deviation.
                    - "min_max": map to [-1, 1] range.
            stats (dict, optional): A dictionary where keys are output modalities (e.g. "observation.image")
                and values are dictionaries of statistic types and their values (e.g.
                `{"mean": torch.randn(3,1,1)}, "std": torch.randn(3,1,1)}`). If provided, as expected for
                training the model for the first time, these statistics will overwrite the default buffers. If
                not provided, as expected for finetuning or evaluation, the default buffers should to be
                overwritten by a call to `policy.load_state_dict(state_dict)`. That way, initializing the
                dataset is not needed to get the stats, since they are already in the policy state_dict.
        """
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats = stats
        stats_buffers = create_stats_buffers(features, norm_map, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    # TODO(rcadene): should we remove torch.no_grad?
    @torch.no_grad
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        for key, ft in self.features.items():
            if key not in batch:
                # FIXME(aliberts, rcadene): This might lead to silent fail!
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = buffer["mean"]
                std = buffer["std"]
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                axis = -mean.ndim
                ndim = batch[key].ndim
                if mean.shape[axis] != batch[key].shape[axis]:
                    C = mean.shape[0]
                    if ndim == 4:
                        batch[key][:, :C] = (batch[key][:, :C] - mean) / (std + 1e-8)
                    elif ndim == 5:
                        batch[key][:, :, :C] = (batch[key][:, :, :C] - mean) / (std + 1e-8) #batch[image] = B * S * C * H * W
                    elif ndim == 3:
                        batch[key][:C] = (batch[key][:C] - mean) / (std + 1e-8)
                else: 
                    batch[key] = (batch[key] - mean) / (std + 1e-8)
                
            elif norm_mode is NormalizationMode.MIN_MAX:
                min = buffer["min"]
                max = buffer["max"]
                assert not torch.isinf(min).any(), _no_stats_error_str("min")
                assert not torch.isinf(max).any(), _no_stats_error_str("max")
                # normalize to [0,1]
                import pdb; pdb.set_trace()
                axis = -mean.ndim
                if mean.shape[axis] != batch[key].shape[axis]:
                    C = min.shape[0]
                    batch[key][:, :, :C] = (batch[key][:, :, :C] - min) / (max - min + 1e-8)
                else: 
                    batch[key] = (batch[key] - min) / (max - min + 1e-8)
                # normalize to [-1, 1]
                batch[key] = batch[key] * 2 - 1
            else:
                raise ValueError(norm_mode)
        return batch

class Normalize_With_Aug(nn.Module):
    """
    - 일반 키: norm_map[ft.type]에 따라 MEAN_STD / MIN_MAX 정규화
    - action 키: augmented_info( swap / flip_x / flip_y )에 따라 8가지 조합 통계에서 per-sample 선택 정규화
    """

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, Tensor]] | None = None,
        aug_stats: dict[str, dict[str, dict[str, Tensor]]] = None,  # {"action": { "swap0_fx0_fy0": {...}, ... }}
    ):
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats = stats
        self.aug_stats = aug_stats
        # 기본 통계 버퍼 등록(키별 buffer_{key})
        stats_buffers = create_stats_buffers(features, norm_map, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

        # ---- action용 augmented 통계 테이블 등록(선택) ----
        self.has_action_aug_tables = False
        if aug_stats is not None:
            # 조합 순서 고정: idx = (swap<<2) | (fx<<1) | fy
            self.order = [f"swap{s}_fx{fx}_fy{fy}"
                     for s in (0, 1) for fx in (0, 1) for fy in (0, 1)]

            def to_t(x):
                return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)
            mins  = torch.stack([to_t(aug_stats[k]["min"])  for k in self.order], dim=0)  # (8,7)
            maxs  = torch.stack([to_t(aug_stats[k]["max"])  for k in self.order], dim=0)  # (8,7)
            means = torch.stack([to_t(aug_stats[k]["mean"]) for k in self.order], dim=0)  # (8,7)
            stds  = torch.stack([to_t(aug_stats[k]["std"])  for k in self.order], dim=0)  # (8,7)

            # 모델 .to(device) 시 함께 이동되도록 버퍼로 등록
            self.register_buffer("buffer_action_aug_min_table",  mins,  persistent=False)
            self.register_buffer("buffer_action_aug_max_table",  maxs,  persistent=False)
            self.register_buffer("buffer_action_aug_mean_table", means, persistent=False)
            self.register_buffer("buffer_action_aug_std_table",  stds,  persistent=False)
            self.has_action_aug_tables = True

    @torch.no_grad()
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)  # shallow copy

        for key, ft in self.features.items():
            if key not in batch: 
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue
            # -------- 액션: 증강 조합 기반 정규화 --------
            if (key == "action" or key == "dynamic.action") and self.has_action_aug_tables and ("augmented_info" in batch):
                act = batch[key]  # (B,T,7) 또는 (B,7)
                if key == "dynamic.action":
                    info = batch["dynamic.augmented_info"]
                else:
                    info = batch["augmented_info"]
                # 기대 구조:
                # info["xy_swapped"]: (B,) bool tensor
                # info["sign_flipped"]["x"|"y"]: (B,) bool tensor (z는 무시)
                swap = info.get("xy_swapped", None)
                sf   = info.get("sign_flipped", {})
                fx   = sf.get("x", None)
                fy   = sf.get("y", None)

                if (swap is None) or (fx is None) or (fy is None):
                    # 안전장치: 정보 없으면 기본 통계로 처리
                    buffer = getattr(self, "buffer_" + key.replace(".", "_"))
                    batch[key] = self._apply_standard_norm(act, buffer, norm_mode)
                    continue

                # (B,) 인덱스: idx = (swap<<2) | (fx<<1) | fy
                idx = (swap.to(torch.int64) << 2) | (fx.to(torch.int64) << 1) | fy.to(torch.int64)
                buf_device = self.buffer_action_aug_min_table.device
                idx = idx.to(device=buf_device, dtype=torch.long)  # ★ 같은 디바이스, Long으로
                # 테이블에서 per-sample 통계 선택 (버퍼는 이미 올바른 device)
                mins  = self.buffer_action_aug_min_table.index_select(0, idx)   # (B,7)
                maxs  = self.buffer_action_aug_max_table.index_select(0, idx)   # (B,7)
                means = self.buffer_action_aug_mean_table.index_select(0, idx)  # (B,7)
                stds  = self.buffer_action_aug_std_table.index_select(0, idx)   # (B,7)

                # 브로드캐스팅: (B,7) → (B,1,7) for (B,T,7)
                if act.dim() == 3:
                    mins, maxs, means, stds = mins[:, None, :], maxs[:, None, :], means[:, None, :], stds[:, None, :]
                elif act.dim() != 2:
                    raise ValueError(f"Unexpected action shape {act.shape}")

                if norm_mode is NormalizationMode.MEAN_STD:
                    batch[key] = (act - means) / (stds + 1e-8)
                elif norm_mode is NormalizationMode.MIN_MAX:
                    x = (act - mins) / (maxs - mins + 1e-8)
                    batch[key] = x * 2 - 1
                else:
                    raise ValueError(norm_mode)

                continue
            # -------------------------------------------

            # ---- 일반 키: 기존 규칙 ----
            buffer = getattr(self, "buffer_" + key.replace(".", "_"))
            batch[key] = self._apply_standard_norm(batch[key], buffer, norm_mode)

        return batch

    @staticmethod
    def _apply_standard_norm(x: Tensor, buffer: dict[str, Tensor], mode: NormalizationMode) -> Tensor:
        if mode is NormalizationMode.MEAN_STD:
            mean = buffer["mean"]; std = buffer["std"]
            assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
            assert not torch.isinf(std).any(),  _no_stats_error_str("std")
            return (x - mean) / (std + 1e-8)
        elif mode is NormalizationMode.MIN_MAX:
            minv = buffer["min"]; maxv = buffer["max"]
            assert not torch.isinf(minv).any(), _no_stats_error_str("min")
            assert not torch.isinf(maxv).any(), _no_stats_error_str("max")
            x01 = (x - minv) / (maxv - minv + 1e-8)
            return x01 * 2 - 1
        else:
            raise ValueError(mode)

class Unnormalize(nn.Module):
    """
    Similar to `Normalize` but unnormalizes output data (e.g. `{"action": torch.randn(b,c)}`) in their
    original range used by the environment.
    """

    def __init__(
        self,
        features: dict[str, PolicyFeature],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            shapes (dict): A dictionary where keys are input modalities (e.g. "observation.image") and values
            are their shapes (e.g. `[3,96,96]`]). These shapes are used to create the tensor buffer containing
            mean, std, min, max statistics. If the provided `shapes` contain keys related to images, the shape
            is adjusted to be invariant to height and width, assuming a channel-first (c, h, w) format.
            modes (dict): A dictionary where keys are output modalities (e.g. "observation.image") and values
                are their normalization modes among:
                    - "mean_std": subtract the mean and divide by standard deviation.
                    - "min_max": map to [-1, 1] range.
            stats (dict, optional): A dictionary where keys are output modalities (e.g. "observation.image")
                and values are dictionaries of statistic types and their values (e.g.
                `{"mean": torch.randn(3,1,1)}, "std": torch.randn(3,1,1)}`). If provided, as expected for
                training the model for the first time, these statistics will overwrite the default buffers. If
                not provided, as expected for finetuning or evaluation, the default buffers should to be
                overwritten by a call to `policy.load_state_dict(state_dict)`. That way, initializing the
                dataset is not needed to get the stats, since they are already in the policy state_dict.
        """
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats = stats
        # `self.buffer_observation_state["mean"]` contains `torch.tensor(state_dim)`
        stats_buffers = create_stats_buffers(features, norm_map, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

    # TODO(rcadene): should we remove torch.no_grad?
    @torch.no_grad
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)  # shallow copy avoids mutating the input batch
        for key, ft in self.features.items():
            if key not in batch:
                continue

            norm_mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if norm_mode is NormalizationMode.IDENTITY:
                continue

            buffer = getattr(self, "buffer_" + key.replace(".", "_"))

            if norm_mode is NormalizationMode.MEAN_STD:
                mean = buffer["mean"]
                std = buffer["std"]
                assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
                assert not torch.isinf(std).any(), _no_stats_error_str("std")
                batch[key] = batch[key] * std + mean
            elif norm_mode is NormalizationMode.MIN_MAX:
                min = buffer["min"]
                max = buffer["max"]
                assert not torch.isinf(min).any(), _no_stats_error_str("min")
                assert not torch.isinf(max).any(), _no_stats_error_str("max")
                batch[key] = (batch[key] + 1) / 2
                batch[key] = batch[key] * (max - min) + min
            else:
                raise ValueError(norm_mode)
        return batch

class Unnormalize_With_Aug(nn.Module):
    """
    - 일반 키: Norm 모드에 맞춰 표준 역정규화
    - action 키: augmented_info (swap, flip_x, flip_y)로 8가지 테이블 중 하나를 per-sample로 선택해 역정규화
    """

    def __init__(
        self,
        features: dict[str, "PolicyFeature"],
        norm_map: dict[str, NormalizationMode],
        stats: dict[str, dict[str, Tensor]] = None,
        aug_stats: dict[str, dict[str, Tensor]] = None,  # {"action": {"swap0_fx0_fy0": {...}, ...}}
    ):
        super().__init__()
        self.features = features
        self.norm_map = norm_map
        self.stats = stats

        # 기본 통계 버퍼(키별)
        stats_buffers = create_stats_buffers(features, norm_map, stats)
        for key, buffer in stats_buffers.items():
            setattr(self, "buffer_" + key.replace(".", "_"), buffer)

        # action용 증강 테이블(선택)
        self.has_action_aug_tables = False
        if aug_stats is not None and "action" in aug_stats:
            order = [f"swap{s}_fx{fx}_fy{fy}"
                     for s in (0, 1) for fx in (0, 1) for fy in (0, 1)]
            def to_t(x):
                return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)

            mins  = torch.stack([to_t(aug_stats[k]["min"])  for k in order], dim=0)  # (8,7)
            maxs  = torch.stack([to_t(aug_stats[k]["max"])  for k in order], dim=0)  # (8,7)
            means = torch.stack([to_t(aug_stats[k]["mean"]) for k in order], dim=0)  # (8,7)
            stds  = torch.stack([to_t(aug_stats[k]["std"])  for k in order], dim=0)  # (8,7)

            self.register_buffer("buffer_action_aug_min_table",  mins,  persistent=False)
            self.register_buffer("buffer_action_aug_max_table",  maxs,  persistent=False)
            self.register_buffer("buffer_action_aug_mean_table", means, persistent=False)
            self.register_buffer("buffer_action_aug_std_table",  stds,  persistent=False)
            self.has_action_aug_tables = True

    @torch.no_grad()
    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        batch = dict(batch)

        for key, ft in self.features.items():
            if key not in batch:
                continue

            mode = self.norm_map.get(ft.type, NormalizationMode.IDENTITY)
            if mode is NormalizationMode.IDENTITY:
                continue

            # ------ action: augmented_info 기반 per-sample 역정규화 ------
            if key == "action" and self.has_action_aug_tables and ("augmented_info" in batch):
                actn = batch[key]  # (B,T,7) or (B,7)
                info = batch["augmented_info"]

                swap = info.get("xy_swapped", None)
                sf   = info.get("sign_flipped", {})
                fx   = sf.get("x", None)
                fy   = sf.get("y", None)

                if (swap is None) or (fx is None) or (fy is None):
                    # 정보 없으면 기본 버퍼로 역정규화
                    buffer = getattr(self, "buffer_" + key.replace(".", "_"))
                    batch[key] = self._apply_standard_unnorm(actn, buffer, mode)
                    continue

                # 인덱스 비트 패킹
                idx = (swap.to(torch.int64) << 2) | (fx.to(torch.int64) << 1) | fy.to(torch.int64)
                # 버퍼에서 per-sample 통계 선택
                mins  = self.buffer_action_aug_min_table.index_select(0, idx)   # (B,7)
                maxs  = self.buffer_action_aug_max_table.index_select(0, idx)   # (B,7)
                means = self.buffer_action_aug_mean_table.index_select(0, idx)  # (B,7)
                stds  = self.buffer_action_aug_std_table.index_select(0, idx)   # (B,7)

                # 브로드캐스팅 정렬
                if actn.dim() == 3:
                    mins, maxs, means, stds = mins[:, None, :], maxs[:, None, :], means[:, None, :], stds[:, None, :]
                elif actn.dim() != 2:
                    raise ValueError(f"Unexpected action shape {actn.shape}")

                if mode is NormalizationMode.MEAN_STD:
                    # x = z*std + mean
                    batch[key] = actn * stds + means
                elif mode is NormalizationMode.MIN_MAX:
                    # x = (z+1)/2; x = x*(max-min)+min
                    x01 = (actn + 1) / 2
                    batch[key] = x01 * (maxs - mins) + mins
                else:
                    raise ValueError(mode)
                continue
            # -----------------------------------------------------------

            # 일반 키: 표준 역정규화
            buffer = getattr(self, "buffer_" + key.replace(".", "_"))
            batch[key] = self._apply_standard_unnorm(batch[key], buffer, mode)

        return batch

    @staticmethod
    def _apply_standard_unnorm(x: Tensor, buffer: dict[str, Tensor], mode: NormalizationMode) -> Tensor:
        if mode is NormalizationMode.MEAN_STD:
            mean = buffer["mean"]; std = buffer["std"]
            assert not torch.isinf(mean).any(), _no_stats_error_str("mean")
            assert not torch.isinf(std).any(),  _no_stats_error_str("std")
            return x * std + mean
        elif mode is NormalizationMode.MIN_MAX:
            minv = buffer["min"]; maxv = buffer["max"]
            assert not torch.isinf(minv).any(), _no_stats_error_str("min")
            assert not torch.isinf(maxv).any(), _no_stats_error_str("max")
            x01 = (x + 1) / 2
            return x01 * (maxv - minv) + minv
        else:
            raise ValueError(mode)