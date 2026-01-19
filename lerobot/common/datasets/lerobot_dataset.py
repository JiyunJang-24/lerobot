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
import contextlib
import logging
import shutil
from pathlib import Path
from typing import Callable, Tuple, Dict, Any
import cv2
import random
import datasets
import numpy as np
import packaging.version
import PIL.Image
import torch
import torch.utils
import einops
from scipy.spatial.transform import Rotation as R
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.errors import RevisionNotFoundError
from lerobot.common.constants import HF_LEROBOT_HOME
from lerobot.common.datasets.compute_stats import aggregate_stats, compute_episode_stats
from lerobot.common.datasets.image_writer import AsyncImageWriter, write_image
from lerobot.common.datasets.utils import (
    DEFAULT_FEATURES,
    DEFAULT_IMAGE_PATH,
    INFO_PATH,
    TASKS_PATH,
    append_jsonlines,
    backward_compatible_episodes_stats,
    check_delta_timestamps,
    check_timestamps_sync,
    check_version_compatibility,
    create_empty_dataset_info,
    create_lerobot_dataset_card,
    embed_images,
    get_delta_indices,
    get_episode_data_index,
    get_features_from_robot,
    get_hf_features_from_features,
    get_safe_version,
    hf_transform_to_torch,
    is_valid_version,
    load_episodes,
    load_episodes_stats,
    load_info,
    load_stats,
    load_tasks,
    validate_episode_buffer,
    validate_frame,
    write_episode,
    write_episode_stats,
    write_info,
    write_json,
    _get_key_signature,
    _get_tensor_shape_dtype,
)
from lerobot.common.datasets.video_utils import (
    VideoFrame,
    decode_video_frames,
    encode_video_frames,
    get_safe_default_codec,
    get_video_info,
)
from lerobot.common.datasets.camera_utils import (
    PluckerEmbedder,
    remove_extrinsic_camera_axis_correction
)
from lerobot.common.datasets.viz_utils import (
    _rescale_make_motion_basis_axis_rgb_tensor_cam_to_world,
    _make_motion_basis_wrist_axis_rgb_tensor_cam_to_world,
    _make_motion_basis_axis_rgb_tensor_cam_to_world,
    _get_motion_dynamics_basis
)
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.datasets.viz_utils import (
    draw_clipped_arrow_fixed_head, 
    project_world_point_to_pixel_cam_to_world,
    save_rgb_image
)
import re
from collections import defaultdict

_FLOAT = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)'
CODEBASE_VERSION = "v2.1"

def extract_angle(repo_id: str) -> float:
    head = repo_id.split('/', 1)[0]              # 예: 'xyg_10_10_315.0_315.0'
    m = re.search(fr'({_FLOAT})_({_FLOAT})$', head)
    if not m:
        raise ValueError(f"Cannot parse angle from repo_id={repo_id}")
    a1, a2 = float(m.group(1)), float(m.group(2))
    if abs(a1 - a2) > 1e-6:
        print(f"[warn] angles differ: {a1} vs {a2} in {repo_id}; using {a1}")
    # (선택) 0~360 정규화 및 키 안정화
    return round(a1 % 360.0, 6)

def _make_aug_variant_from_base(base_action_stats, swap_xy: bool, flip_x: bool, flip_y: bool):
        """
        base_action_stats: dict with keys ["min","max","mean","std","count"]
        values are array-like (길이 >= 2; 0:x, 1:y, 2:z, ...)
        변환 규칙:
        - swap_xy: x↔y 인덱스 스왑 (min/max/mean/std 전부 스왑)
        - flip_x or flip_y: 그 축만 부호 반전 (min,max 뒤집힘; mean 부호 반전; std는 불변)
        """
        min_vals  = np.array(base_action_stats["min"],  copy=True)
        max_vals  = np.array(base_action_stats["max"],  copy=True)
        mean_vals = np.array(base_action_stats["mean"], copy=True)
        std_vals  = np.array(base_action_stats["std"],  copy=True)
        count_val = base_action_stats["count"]


        # 1) sign flip (축별)
        if flip_x:
            min_vals[0], max_vals[0] = -max_vals[0], -min_vals[0]
            mean_vals[0] = -mean_vals[0]
            # std_vals[0] unchanged

        if flip_y:
            min_vals[1], max_vals[1] = -max_vals[1], -min_vals[1]
            mean_vals[1] = -mean_vals[1]
            # std_vals[1] unchanged

        # 2) x<->y swap (인덱스 0 ↔ 1)
        if swap_xy:
            for arr in (min_vals, max_vals, mean_vals, std_vals):
                arr[[0, 1]] = arr[[1, 0]]

        return {
            "min":   min_vals,
            "max":   max_vals,
            "mean":  mean_vals,
            "std":   std_vals,
            "count": count_val,
        }

class LeRobotDatasetMetadata:
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        revision: str | None = None,
        force_cache_sync: bool = False,
        axis_augmentation: bool = False,
        sign_augmentation: list[bool]= [False, False, False],
    ):
        self.repo_id = repo_id
        self.revision = revision if revision else CODEBASE_VERSION
        self.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id
        self.axis_augmentation = axis_augmentation
        self.sign_augmentation = sign_augmentation
        try:
            if force_cache_sync:
                raise FileNotFoundError
            self.load_metadata()
        except (FileNotFoundError, NotADirectoryError):
            if is_valid_version(self.revision):
                self.revision = get_safe_version(self.repo_id, self.revision)

            (self.root / "meta").mkdir(exist_ok=True, parents=True)
            self.pull_from_repo(allow_patterns="meta/")
            self.load_metadata()

    def load_metadata(self):
        self.info = load_info(self.root)
        check_version_compatibility(self.repo_id, self._version, CODEBASE_VERSION)
        self.tasks, self.task_to_task_index = load_tasks(self.root)
        self.episodes = load_episodes(self.root)
        if self._version < packaging.version.parse("v2.1"):
            self.stats = load_stats(self.root)
            self.episodes_stats = backward_compatible_episodes_stats(self.stats, self.episodes)
        else:
            self.episodes_stats = load_episodes_stats(self.root)
            self.stats = aggregate_stats(list(self.episodes_stats.values()))

        self.aug_stats = {}
        self.build_all_aug_stats()

        # if self.axis_augmentation:
        #     min_vals = np.array(self.stats["action"]["min"].copy())
        #     max_vals = np.array(self.stats["action"]["max"].copy())
        #     mean_vals = np.array(self.stats["action"]["mean"].copy())
        #     std_vals = np.array(self.stats["action"]["std"].copy())
        #     self.aug_stats = {
        #         "action": {
        #             "min": np.concatenate([min_vals[1:2], min_vals[0:1], min_vals[2:]]),
        #             "max":  np.concatenate([max_vals[1:2], max_vals[0:1], max_vals[2:]]),
        #             "mean": np.concatenate([mean_vals[1:2], mean_vals[0:1], mean_vals[2:]]),
        #             "std":  np.concatenate([std_vals[1:2], std_vals[0:1], std_vals[2:]]),
        #             "count": self.stats["action"]["count"],
        #         }
        #     }
        # if self.sign_augmentation != [False, False, False] and not self.axis_augmentation:
        #     self.aug_stats = {}
        #     for key in ["action"]:
        #         min_vals = self.stats[key]["min"].copy()
        #         max_vals = self.stats[key]["max"].copy()
        #         mean_vals = self.stats[key]["mean"].copy()
        #         std_vals = self.stats[key]["std"].copy()
        #         for i, sign_aug in enumerate(self.sign_augmentation):
        #             if sign_aug:
        #                 min_vals[i] = -self.stats[key]["max"][i].copy()
        #                 max_vals[i] = -self.stats[key]["min"][i].copy()
        #                 mean_vals[i] = -self.stats[key]["mean"][i].copy()
        #         self.aug_stats[key] = {
        #             "min": min_vals,
        #             "max": max_vals,
        #             "mean": mean_vals,
        #             "std": std_vals,
        #             "count": self.stats[key]["count"].copy(),
        #         }
        # elif self.sign_augmentation != [False, False, False] and self.axis_augmentation:
        #     for key in ["action"]:
        #         min_vals = self.aug_stats[key]["min"].copy()
        #         max_vals = self.aug_stats[key]["max"].copy()
        #         mean_vals = self.aug_stats[key]["mean"].copy()
        #         std_vals = self.aug_stats[key]["std"].copy()
        #         for i, sign_aug in enumerate(self.sign_augmentation):
        #             if sign_aug:
        #                 min_vals[i] = -self.aug_stats[key]["max"][i]
        #                 max_vals[i] = -self.aug_stats[key]["min"][i]
        #                 mean_vals[i] = -self.aug_stats[key]["mean"][i]
        #         self.aug_stats[key] = {
        #             "min": min_vals,
        #             "max": max_vals,
        #             "mean": mean_vals,
        #             "std": std_vals,
        #             "count": self.stats[key]["count"].copy(),
        #         }

    def build_all_aug_stats(self):
        """self.stats['action'] 기준으로 8가지 조합의 통계를 self.aug_stats에 저장"""
        base = self.stats["action"]
        self.aug_stats = {}  # 전부 재생성

        for swap in (0, 1):         # 0: no swap, 1: x<->y swap
            for fx in (0, 1):       # 0: don't flip x, 1: flip x
                for fy in (0, 1):   # 0: don't flip y, 1: flip y
                    key = f"swap{swap}_fx{fx}_fy{fy}"
                    self.aug_stats[key] = _make_aug_variant_from_base(
                        base_action_stats=base,
                        swap_xy=bool(swap),
                        flip_x=bool(fx),
                        flip_y=bool(fy),
                    )

    def select_aug_stats_by_info(self, info: dict):
        """
        증강 info(dict)로부터 해당 조합 키를 만들고, self.aug_stats에서 해당 통계를 반환.
        info 예:
        {
            "applied_axis": bool,
            "applied_sign": bool,
            "xy_swapped": bool,
            "sign_flipped": {"x": bool, "y": bool, "z": bool}
        }
        """
        swap = 1 if info.get("xy_swapped", False) else 0
        sf = info.get("sign_flipped", {}) or {}
        fx = 1 if sf.get("x", False) else 0
        fy = 1 if sf.get("y", False) else 0
        key = f"swap{swap}_fx{fx}_fy{fy}"
        return self.aug_stats[key]

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    @property
    def _version(self) -> packaging.version.Version:
        """Codebase version used to create this dataset."""
        return packaging.version.parse(self.info["codebase_version"])

    def get_data_file_path(self, ep_index: int) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.data_path.format(episode_chunk=ep_chunk, episode_index=ep_index)
        return Path(fpath)

    def get_video_file_path(self, ep_index: int, vid_key: str) -> Path:
        ep_chunk = self.get_episode_chunk(ep_index)
        fpath = self.video_path.format(episode_chunk=ep_chunk, video_key=vid_key, episode_index=ep_index)
        return Path(fpath)

    def get_episode_chunk(self, ep_index: int) -> int:
        return ep_index // self.chunks_size

    @property
    def data_path(self) -> str:
        """Formattable string for the parquet files."""
        return self.info["data_path"]

    @property
    def video_path(self) -> str | None:
        """Formattable string for the video files."""
        return self.info["video_path"]

    @property
    def robot_type(self) -> str | None:
        """Robot type used in recording this dataset."""
        return self.info["robot_type"]

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.info["fps"]

    @property
    def features(self) -> dict[str, dict]:
        """All features contained in the dataset."""
        return self.info["features"]

    @property
    def image_keys(self) -> list[str]:
        """Keys to access visual modalities stored as images."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "image"]

    @property
    def video_keys(self) -> list[str]:
        """Keys to access visual modalities stored as videos."""
        return [key for key, ft in self.features.items() if ft["dtype"] == "video"]

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access visual modalities (regardless of their storage method)."""
        return [key for key, ft in self.features.items() if ft["dtype"] in ["video", "image"]]

    @property
    def names(self) -> dict[str, list | dict]:
        """Names of the various dimensions of vector modalities."""
        return {key: ft["names"] for key, ft in self.features.items()}

    @property
    def shapes(self) -> dict:
        """Shapes for the different features."""
        return {key: tuple(ft["shape"]) for key, ft in self.features.items()}

    @property
    def total_episodes(self) -> int:
        """Total number of episodes available."""
        return self.info["total_episodes"]

    @property
    def total_frames(self) -> int:
        """Total number of frames saved in this dataset."""
        return self.info["total_frames"]

    @property
    def total_tasks(self) -> int:
        """Total number of different tasks performed in this dataset."""
        return self.info["total_tasks"]

    @property
    def total_chunks(self) -> int:
        """Total number of chunks (groups of episodes)."""
        return self.info["total_chunks"]

    @property
    def chunks_size(self) -> int:
        """Max number of episodes per chunk."""
        return self.info["chunks_size"]

    def get_task_index(self, task: str) -> int | None:
        """
        Given a task in natural language, returns its task_index if the task already exists in the dataset,
        otherwise return None.
        """
        return self.task_to_task_index.get(task, None)

    def add_task(self, task: str):
        """
        Given a task in natural language, add it to the dictionary of tasks.
        """
        if task in self.task_to_task_index:
            raise ValueError(f"The task '{task}' already exists and can't be added twice.")

        task_index = self.info["total_tasks"]
        self.task_to_task_index[task] = task_index
        self.tasks[task_index] = task
        self.info["total_tasks"] += 1

        task_dict = {
            "task_index": task_index,
            "task": task,
        }
        append_jsonlines(task_dict, self.root / TASKS_PATH)

    def save_episode(
        self,
        episode_index: int,
        episode_length: int,
        episode_tasks: list[str],
        episode_stats: dict[str, dict],
    ) -> None:
        self.info["total_episodes"] += 1
        self.info["total_frames"] += episode_length

        chunk = self.get_episode_chunk(episode_index)
        if chunk >= self.total_chunks:
            self.info["total_chunks"] += 1

        self.info["splits"] = {"train": f"0:{self.info['total_episodes']}"}
        self.info["total_videos"] += len(self.video_keys)
        if len(self.video_keys) > 0:
            self.update_video_info()

        write_info(self.info, self.root)

        episode_dict = {
            "episode_index": episode_index,
            "tasks": episode_tasks,
            "length": episode_length,
        }
        self.episodes[episode_index] = episode_dict
        write_episode(episode_dict, self.root)

        self.episodes_stats[episode_index] = episode_stats
        self.stats = aggregate_stats([self.stats, episode_stats]) if self.stats else episode_stats
        write_episode_stats(episode_index, episode_stats, self.root)

    def update_video_info(self) -> None:
        """
        Warning: this function writes info from first episode videos, implicitly assuming that all videos have
        been encoded the same way. Also, this means it assumes the first episode exists.
        """
        for key in self.video_keys:
            if not self.features[key].get("info", None):
                video_path = self.root / self.get_video_file_path(ep_index=0, vid_key=key)
                self.info["features"][key]["info"] = get_video_info(video_path)

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Total episodes: '{self.total_episodes}',\n"
            f"    Total frames: '{self.total_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        root: str | Path | None = None,
        robot: Robot | None = None,
        robot_type: str | None = None,
        features: dict | None = None,
        use_videos: bool = True,
    ) -> "LeRobotDatasetMetadata":
        """Creates metadata for a LeRobotDataset."""
        obj = cls.__new__(cls)
        obj.repo_id = repo_id
        obj.root = Path(root) if root is not None else HF_LEROBOT_HOME / repo_id

        obj.root.mkdir(parents=True, exist_ok=False)

        if robot is not None:
            features = get_features_from_robot(robot, use_videos)
            robot_type = robot.robot_type
            if not all(cam.fps == fps for cam in robot.cameras.values()):
                logging.warning(
                    f"Some cameras in your {robot.robot_type} robot don't have an fps matching the fps of your dataset."
                    "In this case, frames from lower fps cameras will be repeated to fill in the blanks."
                )
        elif features is None:
            raise ValueError(
                "Dataset features must either come from a Robot or explicitly passed upon creation."
            )
        else:
            # TODO(aliberts, rcadene): implement sanity check for features
            features = {**features, **DEFAULT_FEATURES}

            # check if none of the features contains a "/" in their names,
            # as this would break the dict flattening in the stats computation, which uses '/' as separator
            for key in features:
                if "/" in key:
                    raise ValueError(f"Feature names should not contain '/'. Found '/' in feature '{key}'.")

            features = {**features, **DEFAULT_FEATURES}

        obj.tasks, obj.task_to_task_index = {}, {}
        obj.episodes_stats, obj.stats, obj.episodes = {}, {}, {}
        obj.info = create_empty_dataset_info(CODEBASE_VERSION, fps, robot_type, features, use_videos)
        if len(obj.video_keys) > 0 and not use_videos:
            raise ValueError()
        write_json(obj.info, obj.root / INFO_PATH)
        obj.revision = None
        return obj


class LeRobotDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        revision: str | None = None,
        force_cache_sync: bool = False,
        download_videos: bool = True,
        video_backend: str | None = None,
        axis_augmentation: bool = False,
        sign_augmentation: list[bool]= [False, False, False],
    ):
        """
        2 modes are available for instantiating this class, depending on 2 different use cases:

        1. Your dataset already exists:
            - On your local disk in the 'root' folder. This is typically the case when you recorded your
              dataset locally and you may or may not have pushed it to the hub yet. Instantiating this class
              with 'root' will load your dataset directly from disk. This can happen while you're offline (no
              internet connection).

            - On the Hugging Face Hub at the address https://huggingface.co/datasets/{repo_id} and not on
              your local disk in the 'root' folder. Instantiating this class with this 'repo_id' will download
              the dataset from that address and load it, pending your dataset is compliant with
              codebase_version v2.0. If your dataset has been created before this new format, you will be
              prompted to convert it using our conversion script from v1.6 to v2.0, which you can find at
              lerobot/common/datasets/v2/convert_dataset_v1_to_v2.py.


        2. Your dataset doesn't already exists (either on local disk or on the Hub): you can create an empty
           LeRobotDataset with the 'create' classmethod. This can be used for recording a dataset or port an
           existing dataset to the LeRobotDataset format.


        In terms of files, LeRobotDataset encapsulates 3 main things:
            - metadata:
                - info contains various information about the dataset like shapes, keys, fps etc.
                - stats stores the dataset statistics of the different modalities for normalization
                - tasks contains the prompts for each task of the dataset, which can be used for
                  task-conditioned training.
            - hf_dataset (from datasets.Dataset), which will read any values from parquet files.
            - videos (optional) from which frames are loaded to be synchronous with data from parquet files.

        A typical LeRobotDataset looks like this from its root path:
        .
        ├── data
        │   ├── chunk-000
        │   │   ├── episode_000000.parquet
        │   │   ├── episode_000001.parquet
        │   │   ├── episode_000002.parquet
        │   │   └── ...
        │   ├── chunk-001
        │   │   ├── episode_001000.parquet
        │   │   ├── episode_001001.parquet
        │   │   ├── episode_001002.parquet
        │   │   └── ...
        │   └── ...
        ├── meta
        │   ├── episodes.jsonl
        │   ├── info.json
        │   ├── stats.json
        │   └── tasks.jsonl
        └── videos
            ├── chunk-000
            │   ├── observation.images.laptop
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            │   ├── observation.images.phone
            │   │   ├── episode_000000.mp4
            │   │   ├── episode_000001.mp4
            │   │   ├── episode_000002.mp4
            │   │   └── ...
            ├── chunk-001
            └── ...

        Note that this file-based structure is designed to be as versatile as possible. The files are split by
        episodes which allows a more granular control over which episodes one wants to use and download. The
        structure of the dataset is entirely described in the info.json file, which can be easily downloaded
        or viewed directly on the hub before downloading any actual data. The type of files used are very
        simple and do not need complex tools to be read, it only uses .parquet, .json and .mp4 files (and .md
        for the README).

        Args:
            repo_id (str): This is the repo id that will be used to fetch the dataset. Locally, the dataset
                will be stored under root/repo_id.
            root (Path | None, optional): Local directory to use for downloading/writing files. You can also
                set the LEROBOT_HOME environment variable to point to a different location. Defaults to
                '~/.cache/huggingface/lerobot'.
            episodes (list[int] | None, optional): If specified, this will only load episodes specified by
                their episode_index in this list. Defaults to None.
            image_transforms (Callable | None, optional): You can pass standard v2 image transforms from
                torchvision.transforms.v2 here which will be applied to visual modalities (whether they come
                from videos or images). Defaults to None.
            delta_timestamps (dict[list[float]] | None, optional): _description_. Defaults to None.
            tolerance_s (float, optional): Tolerance in seconds used to ensure data timestamps are actually in
                sync with the fps value. It is used at the init of the dataset to make sure that each
                timestamps is separated to the next by 1/fps +/- tolerance_s. This also applies to frames
                decoded from video files. It is also used to check that `delta_timestamps` (when provided) are
                multiples of 1/fps. Defaults to 1e-4.
            revision (str, optional): An optional Git revision id which can be a branch name, a tag, or a
                commit hash. Defaults to current codebase version tag.
            sync_cache_first (bool, optional): Flag to sync and refresh local files first. If True and files
                are already present in the local cache, this will be faster. However, files loaded might not
                be in sync with the version on the hub, especially if you specified 'revision'. Defaults to
                False.
            download_videos (bool, optional): Flag to download the videos. Note that when set to True but the
                video files are already present on local disk, they won't be downloaded again. Defaults to
                True.
            video_backend (str | None, optional): Video backend to use for decoding videos. Defaults to torchcodec when available int the platform; otherwise, defaults to 'pyav'.
                You can also use the 'pyav' decoder used by Torchvision, which used to be the default option, or 'video_reader' which is another decoder of Torchvision.
        """
        super().__init__()
        self.repo_id = repo_id
        self.root = Path(root) if root else HF_LEROBOT_HOME / repo_id
        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.episodes = episodes
        self.tolerance_s = tolerance_s
        self.revision = revision if revision else CODEBASE_VERSION
        self.video_backend = video_backend if video_backend else get_safe_default_codec()
        self.delta_indices = None

        # Unused attributes
        self.image_writer = None
        self.episode_buffer = None

        self.root.mkdir(exist_ok=True, parents=True)

        # Load metadata
        self.meta = LeRobotDatasetMetadata(
            self.repo_id, self.root, self.revision, force_cache_sync=force_cache_sync, axis_augmentation=axis_augmentation, sign_augmentation=sign_augmentation
        )
        if self.episodes is not None and self.meta._version >= packaging.version.parse("v2.1"):
            episodes_stats = [self.meta.episodes_stats[ep_idx] for ep_idx in self.episodes]
            self.stats = aggregate_stats(episodes_stats)

        # Load actual data
        try:
            if force_cache_sync:
                raise FileNotFoundError
            assert all((self.root / fpath).is_file() for fpath in self.get_episodes_file_paths())
            self.hf_dataset = self.load_hf_dataset()
        except (AssertionError, FileNotFoundError, NotADirectoryError):
            self.revision = get_safe_version(self.repo_id, self.revision)
            self.download_episodes(download_videos)
            self.hf_dataset = self.load_hf_dataset()

        self.episode_data_index = get_episode_data_index(self.meta.episodes, self.episodes)

        # Check timestamps
        timestamps = torch.stack(self.hf_dataset["timestamp"]).numpy()
        episode_indices = torch.stack(self.hf_dataset["episode_index"]).numpy()
        ep_data_index_np = {k: t.numpy() for k, t in self.episode_data_index.items()}
        check_timestamps_sync(timestamps, episode_indices, ep_data_index_np, self.fps, self.tolerance_s)

        # Setup delta_indices
        if self.delta_timestamps is not None:
            check_delta_timestamps(self.delta_timestamps, self.fps, self.tolerance_s)
            self.delta_indices = get_delta_indices(self.delta_timestamps, self.fps)

    def push_to_hub(
        self,
        branch: str | None = None,
        tags: list | None = None,
        license: str | None = "apache-2.0",
        tag_version: bool = True,
        push_videos: bool = True,
        private: bool = False,
        allow_patterns: list[str] | str | None = None,
        upload_large_folder: bool = False,
        **card_kwargs,
    ) -> None:
        ignore_patterns = ["images/"]
        if not push_videos:
            ignore_patterns.append("videos/")

        hub_api = HfApi()
        hub_api.create_repo(
            repo_id=self.repo_id,
            private=private,
            repo_type="dataset",
            exist_ok=True,
        )
        if branch:
            hub_api.create_branch(
                repo_id=self.repo_id,
                branch=branch,
                revision=self.revision,
                repo_type="dataset",
                exist_ok=True,
            )

        upload_kwargs = {
            "repo_id": self.repo_id,
            "folder_path": self.root,
            "repo_type": "dataset",
            "revision": branch,
            "allow_patterns": allow_patterns,
            "ignore_patterns": ignore_patterns,
        }
        if upload_large_folder:
            hub_api.upload_large_folder(**upload_kwargs)
        else:
            hub_api.upload_folder(**upload_kwargs)

        if not hub_api.file_exists(self.repo_id, REPOCARD_NAME, repo_type="dataset", revision=branch):
            card = create_lerobot_dataset_card(
                tags=tags, dataset_info=self.meta.info, license=license, **card_kwargs
            )
            card.push_to_hub(repo_id=self.repo_id, repo_type="dataset", revision=branch)

        if tag_version:
            with contextlib.suppress(RevisionNotFoundError):
                hub_api.delete_tag(self.repo_id, tag=CODEBASE_VERSION, repo_type="dataset")
            hub_api.create_tag(self.repo_id, tag=CODEBASE_VERSION, revision=branch, repo_type="dataset")

    def pull_from_repo(
        self,
        allow_patterns: list[str] | str | None = None,
        ignore_patterns: list[str] | str | None = None,
    ) -> None:
        snapshot_download(
            self.repo_id,
            repo_type="dataset",
            revision=self.revision,
            local_dir=self.root,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
        )

    def download_episodes(self, download_videos: bool = True) -> None:
        """Downloads the dataset from the given 'repo_id' at the provided version. If 'episodes' is given, this
        will only download those episodes (selected by their episode_index). If 'episodes' is None, the whole
        dataset will be downloaded. Thanks to the behavior of snapshot_download, if the files are already present
        in 'local_dir', they won't be downloaded again.
        """
        # TODO(rcadene, aliberts): implement faster transfer
        # https://huggingface.co/docs/huggingface_hub/en/guides/download#faster-downloads
        files = None
        ignore_patterns = None if download_videos else "videos/"
        if self.episodes is not None:
            files = self.get_episodes_file_paths()

        self.pull_from_repo(allow_patterns=files, ignore_patterns=ignore_patterns)

    def get_episodes_file_paths(self) -> list[Path]:
        episodes = self.episodes if self.episodes is not None else list(range(self.meta.total_episodes))
        fpaths = [str(self.meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self.meta.video_keys) > 0:
            video_files = [
                str(self.meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self.meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files

        return fpaths

    def load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        if self.episodes is None:
            path = str(self.root / "data")
            hf_dataset = load_dataset("parquet", data_dir=path, split="train")
        else:
            files = [str(self.root / self.meta.get_data_file_path(ep_idx)) for ep_idx in self.episodes]
            hf_dataset = load_dataset("parquet", data_files=files, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def create_hf_dataset(self) -> datasets.Dataset:
        features = get_hf_features_from_features(self.features)
        ft_dict = {col: [] for col in features}
        hf_dataset = datasets.Dataset.from_dict(ft_dict, features=features, split="train")

        # TODO(aliberts): hf_dataset.set_format("torch")
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    @property
    def fps(self) -> int:
        """Frames per second used during data collection."""
        return self.meta.fps

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        return len(self.hf_dataset) if self.hf_dataset is not None else self.meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self.meta.total_episodes

    @property
    def features(self) -> dict[str, dict]:
        return self.meta.features

    @property
    def hf_features(self) -> datasets.Features:
        """Features of the hf_dataset."""
        if self.hf_dataset is not None:
            return self.hf_dataset.features
        else:
            return get_hf_features_from_features(self.features)

    def _get_query_indices(self, idx: int, ep_idx: int) -> tuple[dict[str, list[int | bool]]]:
        ep_start = self.episode_data_index["from"][ep_idx]
        ep_end = self.episode_data_index["to"][ep_idx]
        query_indices = {
            key: [max(ep_start.item(), min(ep_end.item() - 1, idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {  # Pad values outside of current episode range
            f"{key}_is_pad": torch.BoolTensor(
                [(idx + delta < ep_start.item()) | (idx + delta >= ep_end.item()) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self.meta.video_keys:
            if query_indices is not None and key in query_indices:
                timestamps = self.hf_dataset.select(query_indices[key])["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        return {
            key: torch.stack(self.hf_dataset.select(q_idx)[key])
            for key, q_idx in query_indices.items()
            if key not in self.meta.video_keys
        }

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault. This probably happens because a memory reference to the video loader is created in
        the main process and a subprocess fails to access it.
        """
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            video_path = self.root / self.meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, query_ts, self.tolerance_s, self.video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    def _add_padding_keys(self, item: dict, padding: dict[str, list[bool]]) -> dict:
        for key, val in padding.items():
            item[key] = torch.BoolTensor(val)
        return item

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx) -> dict:
        item = self.hf_dataset[idx]
        ep_idx = item["episode_index"].item()
        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self.meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self.image_transforms is not None:
            image_keys = self.meta.camera_keys
            for cam in image_keys:
                item[cam] = self.image_transforms(item[cam])

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self.meta.tasks[task_idx]

        return item

    def __repr__(self):
        feature_keys = list(self.features)
        return (
            f"{self.__class__.__name__}({{\n"
            f"    Repository ID: '{self.repo_id}',\n"
            f"    Number of selected episodes: '{self.num_episodes}',\n"
            f"    Number of selected samples: '{self.num_frames}',\n"
            f"    Features: '{feature_keys}',\n"
            "})',\n"
        )

    def create_episode_buffer(self, episode_index: int | None = None) -> dict:
        current_ep_idx = self.meta.total_episodes if episode_index is None else episode_index
        ep_buffer = {}
        # size and task are special cases that are not in self.features
        ep_buffer["size"] = 0
        ep_buffer["task"] = []
        for key in self.features:
            ep_buffer[key] = current_ep_idx if key == "episode_index" else []
        return ep_buffer

    def _get_image_file_path(self, episode_index: int, image_key: str, frame_index: int) -> Path:
        fpath = DEFAULT_IMAGE_PATH.format(
            image_key=image_key, episode_index=episode_index, frame_index=frame_index
        )
        return self.root / fpath

    def _save_image(self, image: torch.Tensor | np.ndarray | PIL.Image.Image, fpath: Path) -> None:
        if self.image_writer is None:
            if isinstance(image, torch.Tensor):
                image = image.cpu().numpy()
            write_image(image, fpath)
        else:
            self.image_writer.save_image(image=image, fpath=fpath)

    def add_frame(self, frame: dict) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # Convert torch to numpy if needed
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        validate_frame(frame, self.features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # Automatically add frame_index and timestamp to episode buffer
        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        # Add frame features to episode_buffer
        for key in frame:
            if key == "task":
                # Note: we associate the task in natural language to its task index during `save_episode`
                self.episode_buffer["task"].append(frame["task"])
                continue

            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            if self.features[key]["dtype"] in ["image", "video"]:
                img_path = self._get_image_file_path(
                    episode_index=self.episode_buffer["episode_index"], image_key=key, frame_index=frame_index
                )
                if frame_index == 0:
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_image(frame[key], img_path)
                self.episode_buffer[key].append(str(img_path))
            else:
                self.episode_buffer[key].append(frame[key])

        self.episode_buffer["size"] += 1

    def save_episode(self, episode_data: dict | None = None, keep_images: bool | None = False) -> None:
        """
        This will save to disk the current episode in self.episode_buffer.

        Args:
            episode_data (dict | None, optional): Dict containing the episode data to save. If None, this will
                save the current episode in self.episode_buffer, which is filled with 'add_frame'. Defaults to
                None.
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # size and task are special cases that won't be added to hf_dataset
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # Add new tasks to the tasks dictionary
        for task in episode_tasks:
            task_index = self.meta.get_task_index(task)
            if task_index is None:
                self.meta.add_task(task)

        # Given tasks in natural language, find their corresponding task indices
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # index, episode_index, task_index are already processed above, and image and video
            # are processed separately by storing image path and frame info as meta data
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["image", "video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key])

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)
        ep_stats = compute_episode_stats(episode_buffer, self.features)

        if len(self.meta.video_keys) > 0:
            video_paths = self.encode_episode_videos(episode_index)
            for key in self.meta.video_keys:
                episode_buffer[key] = video_paths[key]

        # `meta.save_episode` be executed after encoding the videos
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats)

        ep_data_index = get_episode_data_index(self.meta.episodes, [episode_index])
        ep_data_index_np = {k: t.numpy() for k, t in ep_data_index.items()}
        check_timestamps_sync(
            episode_buffer["timestamp"],
            episode_buffer["episode_index"],
            ep_data_index_np,
            self.fps,
            self.tolerance_s,
        )

        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        # delete images
        if not keep_images:
            img_dir = self.root / "images"
            if img_dir.is_dir():
                shutil.rmtree(self.root / "images")

        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()

    def _save_episode_table(self, episode_buffer: dict, episode_index: int) -> None:
        episode_dict = {key: episode_buffer[key] for key in self.hf_features}
        ep_dataset = datasets.Dataset.from_dict(episode_dict, features=self.hf_features, split="train")
        ep_dataset = embed_images(ep_dataset)
        self.hf_dataset = concatenate_datasets([self.hf_dataset, ep_dataset])
        self.hf_dataset.set_transform(hf_transform_to_torch)
        ep_data_path = self.root / self.meta.get_data_file_path(ep_index=episode_index)
        ep_data_path.parent.mkdir(parents=True, exist_ok=True)
        ep_dataset.to_parquet(ep_data_path)

    def clear_episode_buffer(self) -> None:
        episode_index = self.episode_buffer["episode_index"]
        if self.image_writer is not None:
            for cam_key in self.meta.camera_keys:
                img_dir = self._get_image_file_path(
                    episode_index=episode_index, image_key=cam_key, frame_index=0
                ).parent
                if img_dir.is_dir():
                    shutil.rmtree(img_dir)

        # Reset the buffer
        self.episode_buffer = self.create_episode_buffer()

    def start_image_writer(self, num_processes: int = 0, num_threads: int = 4) -> None:
        if isinstance(self.image_writer, AsyncImageWriter):
            logging.warning(
                "You are starting a new AsyncImageWriter that is replacing an already existing one in the dataset."
            )

        self.image_writer = AsyncImageWriter(
            num_processes=num_processes,
            num_threads=num_threads,
        )

    def stop_image_writer(self) -> None:
        """
        Whenever wrapping this dataset inside a parallelized DataLoader, this needs to be called first to
        remove the image_writer in order for the LeRobotDataset object to be pickleable and parallelized.
        """
        if self.image_writer is not None:
            self.image_writer.stop()
            self.image_writer = None

    def _wait_image_writer(self) -> None:
        """Wait for asynchronous image writer to finish."""
        if self.image_writer is not None:
            self.image_writer.wait_until_done()

    def encode_videos(self) -> None:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        for ep_idx in range(self.meta.total_episodes):
            self.encode_episode_videos(ep_idx)

    def encode_episode_videos(self, episode_index: int) -> dict:
        """
        Use ffmpeg to convert frames stored as png into mp4 videos.
        Note: `encode_video_frames` is a blocking call. Making it asynchronous shouldn't speedup encoding,
        since video encoding with ffmpeg is already using multithreading.
        """
        video_paths = {}
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            video_paths[key] = str(video_path)
            if video_path.is_file():
                # Skip if video is already encoded. Could be the case when resuming data recording.
                continue
            img_dir = self._get_image_file_path(
                episode_index=episode_index, image_key=key, frame_index=0
            ).parent
            encode_video_frames(img_dir, video_path, self.fps, overwrite=True)

        return video_paths

    @classmethod
    def create(
        cls,
        repo_id: str,
        fps: int,
        root: str | Path | None = None,
        robot: Robot | None = None,
        robot_type: str | None = None,
        features: dict | None = None,
        use_videos: bool = True,
        tolerance_s: float = 1e-4,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        video_backend: str | None = None,
    ) -> "LeRobotDataset":
        """Create a LeRobot Dataset from scratch in order to record data."""
        obj = cls.__new__(cls)
        obj.meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id,
            fps=fps,
            root=root,
            robot=robot,
            robot_type=robot_type,
            features=features,
            use_videos=use_videos,
        )
        obj.repo_id = obj.meta.repo_id
        obj.root = obj.meta.root
        obj.revision = None
        obj.tolerance_s = tolerance_s
        obj.image_writer = None

        if image_writer_processes or image_writer_threads:
            obj.start_image_writer(image_writer_processes, image_writer_threads)

        # TODO(aliberts, rcadene, alexander-soare): Merge this with OnlineBuffer/DataBuffer
        obj.episode_buffer = obj.create_episode_buffer()

        obj.episodes = None
        obj.hf_dataset = obj.create_hf_dataset()
        obj.image_transforms = None
        obj.delta_timestamps = None
        obj.delta_indices = None
        obj.episode_data_index = None
        obj.video_backend = video_backend if video_backend is not None else get_safe_default_codec()
        return obj


class MultiLeRobotDataset(torch.utils.data.Dataset):
    """A dataset consisting of multiple underlying `LeRobotDataset`s.

    The underlying `LeRobotDataset`s are effectively concatenated, and this class adopts much of the API
    structure of `LeRobotDataset`.
    """

    def __init__(
        self,
        repo_ids: list[str],
        root: str | Path | None = None,
        episodes: dict | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerances_s: dict | None = None,
        download_videos: bool = True,
        video_backend: str | None = None,
        use_action_avg: bool = False,
        window_size: int | None = None,
        use_dynamic_feature: bool = False,
        num_dynamic_feature: int = 3,
        axis_augmentation: bool = False,
        sign_augmentation: list[bool] = [False, False, False],
        pretrain_dynamic_backbone: bool = False,
        use_plucker: bool = False,
        use_dynamics_basis: bool = False,
        realworld: bool = False,
        apply_basis_scale: bool = False,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.root = Path(root) if root else HF_LEROBOT_HOME
        self.tolerances_s = tolerances_s if tolerances_s else {repo_id: 1e-4 for repo_id in repo_ids}
        # Construct the underlying datasets passing everything but `transform` and `delta_timestamps` which
        # are handled by this class.
        self.axis_augmentation = axis_augmentation # change x, y axis for data augmentation
        self.sign_augmentation = sign_augmentation #[False, False, False]
        self._datasets = [
            LeRobotDataset(
                repo_id,
                root=self.root / repo_id,
                episodes=episodes[repo_id] if episodes else None,
                image_transforms=image_transforms,
                delta_timestamps=delta_timestamps[repo_id],
                tolerance_s=self.tolerances_s[repo_id],
                axis_augmentation=self.axis_augmentation,
                sign_augmentation=self.sign_augmentation,
                download_videos=download_videos,
                video_backend=video_backend,
            )
            for repo_id in repo_ids
        ]
        # Disable any data keys that are not common across all of the datasets. Note: we may relax this
        # restriction in future iterations of this class. For now, this is necessary at least for being able
        # to use PyTorch's default DataLoader collate function.
        self.disabled_features = set()
        intersection_features = set(self._datasets[0].features)
        for ds in self._datasets:
            intersection_features.intersection_update(ds.features)
        if len(intersection_features) == 0:
            raise RuntimeError(
                "Multiple datasets were provided but they had no keys common to all of them. "
                "The multi-dataset functionality currently only keeps common keys."
            )
        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            extra_keys = set(ds.features).difference(intersection_features)
            logging.warning(
                f"keys {extra_keys} of {repo_id} were disabled as they are not contained in all the "
                "other datasets."
            )
            self.disabled_features.update(extra_keys)
        # 2) 추가: 공통 key지만 shape/dtype이 다른 key padding
        # 기준(dataset 0)의 시그니처
        self.compute_pad_specs(intersection_features)

        self.image_transforms = image_transforms
        self.delta_timestamps = delta_timestamps
        self.angle_to_indices = defaultdict(list)
        self.indices_to_angle = {}
        robot_type_to_indices = defaultdict(list)
        for idx, dataset in enumerate(self._datasets):
            robot_type = dataset.meta.robot_type or self.repo_ids[idx]
            # if robot_type != "panda":
                # print("not implementing multiple embodiment dataset for separte normalize")
            # robot_type = "panda"
            robot_type_to_indices[robot_type].append(idx)

        def _aggregate_attr(attr: str, indices: list[int]) -> dict[str, dict[str, np.ndarray]]:
            values = [getattr(self._datasets[i].meta, attr) for i in indices]
            return aggregate_stats(values)
        if len(robot_type_to_indices) == 1:
            self.stats = aggregate_stats([dataset.meta.stats for dataset in self._datasets])
            self.aug_stats = aggregate_stats([dataset.meta.aug_stats for dataset in self._datasets])
        else:
            self.stats = {
                robot_type: _aggregate_attr("stats", indices) for robot_type, indices in robot_type_to_indices.items()
            }
            self.aug_stats = {
                robot_type: _aggregate_attr("aug_stats", indices)
                for robot_type, indices in robot_type_to_indices.items()
            }

        self.use_action_avg = use_action_avg
        self.window_size = window_size
        self.use_dynamic_feature = use_dynamic_feature
        self.num_dynamic_feature = num_dynamic_feature
        self.pretrain_dynamic_backbone = pretrain_dynamic_backbone

        if self.pretrain_dynamic_backbone:
            self.angle_classes = sorted(self.angle_to_indices.keys())        # 예: [0.0, 45.0, 90.0, 135.0, 225.0, 270.0, 315.0]
            self.angle_to_class = {ang: i for i, ang in enumerate(self.angle_classes)}

        self.use_plucker = use_plucker
        self.use_dynamics_basis = use_dynamics_basis
        self.realworld = realworld
        # PLUCKER
        if self.use_plucker:
            self.image_size = 256
            self.plucker_embedder = PluckerEmbedder(img_size=self.image_size, device='cpu')
        else:
            self.plucker_embedder = None
        self.apply_basis_scale = apply_basis_scale
    def augment_action_sequence(
        self,
        action: torch.Tensor,          # (T, 7)
        xy_idx: tuple[int, int] = (0, 1),
        z_idx: int = 2,
        p_axis: float = 0.5,           # ← 축 스왑 확률
        p_sign: float = 0.5,           # ← 부호 반전 확률
        info: dict | None = None,      # ← 주어지면 그 내용대로 재현 적용
    ) -> tuple[torch.Tensor, dict]:
        """
        입력: (T,7) 액션 시퀀스.
        - info is None:
            * self.axis_augmentation=True 이면 p_axis 확률로 x/y 스왑(시퀀스 전체 동일)
            * self.sign_augmentation=[sx,sy,sz]에서 True인 축은 p_sign 확률로 (시퀀스 전체) 부호 반전
            (축별 on/off는 sign_augmentation으로 결정)
        - info is not None:
            * info에 명시된 applied_axis / applied_sign / xy_swapped / sign_flipped을 그대로 재현(확률/설정 무시)
        반환: (aug_action(T,7), info)
        info = {
            "applied_axis": bool,         # 축 스왑을 적용했는지
            "applied_sign": bool,         # 부호 반전을 적용했는지(적어도 한 축)
            "xy_swapped": bool,           # 실제 x/y 스왑 수행 여부
            "sign_flipped": {"x":bool, "y":bool, "z":bool}
        }
        """
        assert torch.is_tensor(action), "action must be a torch.Tensor"
        assert action.dim() == 2 and action.size(-1) == 7, f"Expected (T,7), got {tuple(action.shape)}"

        T, D = action.shape
        x_i, y_i = xy_idx
        assert D > max(x_i, y_i, z_idx), "action last dim must cover x/y/z indices"

        aug = action.clone()
        device = aug.device

        # -------- 1) info가 주어진 경우: 그대로 재현 --------
        if info is not None:
            # 하위 호환: applied_axis / applied_sign 없으면 기존 키로 유추
            applied_axis = bool(info.get("applied_axis", info.get("xy_swapped", False)))
            applied_sign = bool(info.get("applied_sign", False))
            xy_swapped   = bool(info.get("xy_swapped", False))
            sf = info.get("sign_flipped", {}) or {}
            sx = bool(sf.get("x", False))
            sy = bool(sf.get("y", False))
            sz = bool(sf.get("z", False))

            if applied_sign:
                if sx: aug[:, x_i] = -aug[:, x_i]
                if sy: aug[:, y_i] = -aug[:, y_i]
                if sz: aug[:, z_idx] = -aug[:, z_idx]

            if applied_axis and xy_swapped:
                x_vals = aug[:, x_i].clone()
                y_vals = aug[:, y_i].clone()
                aug[:, x_i] = y_vals
                aug[:, y_i] = x_vals


            out_info = {
                "applied_axis": applied_axis,
                "applied_sign": applied_sign,
                "xy_swapped": xy_swapped,
                "sign_flipped": {"x": sx, "y": sy, "z": sz},
            }
            return aug, out_info

        # -------- 2) info가 None인 경우: 확률/설정 기반 --------
        # sign 축 사용 여부 (설정)
        sx, sy, sz = (self.sign_augmentation + [False, False, False])[:3]

        # 각 증강의 적용 여부를 독립적으로 샘플링
        do_sign_x = bool((sx) and (torch.rand((), device=device) < p_sign).item())
        do_sign_y = bool((sy) and (torch.rand((), device=device) < p_sign).item())
        do_sign_z = bool((sz) and (torch.rand((), device=device) < p_sign).item())

        # 축별 on/off에 따라 부호 반전
        flip_x = bool(do_sign_x and sx)
        flip_y = bool(do_sign_y and sy)
        flip_z = bool(do_sign_z and sz)

        if flip_x: aug[:, x_i] = -aug[:, x_i]
        if flip_y: aug[:, y_i] = -aug[:, y_i]
        if flip_z: aug[:, z_idx] = -aug[:, z_idx]

        do_axis = bool(self.axis_augmentation and (torch.rand((), device=device) < p_axis).item())
        xy_swapped = False
        if do_axis:
            x_vals = aug[:, x_i].clone()
            y_vals = aug[:, y_i].clone()
            aug[:, x_i] = y_vals
            aug[:, y_i] = x_vals
            xy_swapped = True

        out_info = {
            "applied_axis": do_axis,
            "applied_sign": bool(flip_x or flip_y or flip_z),
            "xy_swapped": xy_swapped,
            "sign_flipped": {
                "x": flip_x,
                "y": flip_y,
                "z": flip_z,
            },
        }
        return aug, out_info

    def pad_item_inplace(self, item):
        if not hasattr(self, "pad_specs"):
            return item

        for k, spec in self.pad_specs.items():
            if k not in item:
                continue
            x = item[k]
            if not isinstance(x, torch.Tensor):
                continue

            axis = spec["axis"]
            target = spec["target"]

            # axis=-1만 지원(원하면 확장 가능)
            cur = x.shape[axis]
            if cur == target:
                continue
            if cur > target:
                # 이 경우는 데이터가 더 큰데 target이 작다는 뜻이라, 보통 target을 max로 잡으면 안 발생
                # 혹시 발생하면 disable or truncate 정책 선택
                raise ValueError(f"Padding spec target smaller than current for key={k}: cur={cur} target={target}")

            pad_amount = target - cur
            # torch.nn.functional.pad는 last-dim padding에 (0, pad_amount) 형태
            import torch.nn.functional as F
            # F.pad expects pad tuple for last dims: (pad_left, pad_right)
            x_pad = F.pad(x, (0, pad_amount), mode="constant", value=0)
            item[k] = x_pad

        return item
    @property
    def repo_id_to_index(self):
        """Return a mapping from dataset repo_id to a dataset index automatically created by this class.

        This index is incorporated as a data key in the dictionary returned by `__getitem__`.
        """
        return {repo_id: i for i, repo_id in enumerate(self.repo_ids)}

    @property
    def repo_index_to_id(self):
        """Return the inverse mapping if repo_id_to_index."""
        return {v: k for k, v in self.repo_id_to_index}

    @property
    def fps(self) -> int:
        """Frames per second used during data collection.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info["fps"]

    @property
    def video(self) -> bool:
        """Returns True if this dataset loads video frames from mp4 files.

        Returns False if it only loads images from png files.

        NOTE: Fow now, this relies on a check in __init__ to make sure all sub-datasets have the same info.
        """
        return self._datasets[0].meta.info.get("video", False)

    @property
    def features(self) -> datasets.Features:
        features = {}
        for dataset in self._datasets:
            features.update({k: v for k, v in dataset.hf_features.items() if k not in self.disabled_features})
        return features

    @property
    def camera_keys(self) -> list[str]:
        """Keys to access image and video stream from cameras."""
        keys = []
        for key, feats in self.features.items():
            if isinstance(feats, (datasets.Image, VideoFrame)):
                keys.append(key)
        return keys

    @property
    def video_frame_keys(self) -> list[str]:
        """Keys to access video frames that requires to be decoded into images.

        Note: It is empty if the dataset contains images only,
        or equal to `self.cameras` if the dataset contains videos only,
        or can even be a subset of `self.cameras` in a case of a mixed image/video dataset.
        """
        video_frame_keys = []
        for key, feats in self.features.items():
            if isinstance(feats, VideoFrame):
                video_frame_keys.append(key)
        return video_frame_keys

    @property
    def num_frames(self) -> int:
        """Number of samples/frames."""
        return sum(d.num_frames for d in self._datasets)

    @property
    def num_episodes(self) -> int:
        """Number of episodes."""
        return sum(d.num_episodes for d in self._datasets)

    @property
    def tolerance_s(self) -> float:
        """Tolerance in seconds used to discard loaded frames when their timestamps
        are not close enough from the requested frames. It is only used when `delta_timestamps`
        is provided or when loading video frames from mp4 files.
        """
        # 1e-4 to account for possible numerical error
        return 1 / self.fps - 1e-4

    def __len__(self):
        return self.num_frames
    
    def compute_pad_specs(self, intersection_features, sample_idx=0):
        """
        Creates:
        self.pad_specs: dict[key] = {"axis": -1, "target": int}
        and updates:
        self.disabled_features for truly incompatible keys.
        """
        pad_specs = {}
        mismatched_disable = set()

        # collect shapes/dtypes across datasets for each key
        shapes_by_key = defaultdict(list)
        dtypes_by_key = defaultdict(set)

        for repo_id, ds in zip(self.repo_ids, self._datasets, strict=True):
            for k in intersection_features:
                info = _get_tensor_shape_dtype(ds, k, sample_idx=sample_idx)
                if info is None:
                    continue
                shape, dtype = info
                shapes_by_key[k].append((repo_id, shape))
                dtypes_by_key[k].add(dtype)

        for k, entries in shapes_by_key.items():
            # dtype mismatch -> 일단 disable (원하면 promote/cast 규칙도 가능)
            if len(dtypes_by_key[k]) > 1:
                mismatched_disable.add(k)
                logging.warning(f"key '{k}' disabled: dtype mismatch across datasets: {dtypes_by_key[k]}")
                continue

            # shape 분석: last dim만 다르고 나머지는 동일하면 pad 가능
            shapes = [sh for _, sh in entries]
            if len(set(shapes)) == 1:
                continue  # 모두 동일 -> pad 필요 없음

            rank_set = {len(sh) for sh in shapes}
            if len(rank_set) != 1:
                mismatched_disable.add(k)
                logging.warning(f"key '{k}' disabled: rank mismatch across datasets: {entries}")
                continue

            rank = next(iter(rank_set))
            if rank == 0:
                mismatched_disable.add(k)
                logging.warning(f"key '{k}' disabled: scalar mismatch (?) {entries}")
                continue

            # 마지막 축 제외한 prefix가 모두 같은지 확인
            prefixes = {sh[:-1] for sh in shapes}
            if len(prefixes) != 1:
                mismatched_disable.add(k)
                logging.warning(f"key '{k}' disabled: non-last dims differ: {entries}")
                continue

            max_last = max(sh[-1] for sh in shapes)
            pad_specs[k] = {"axis": -1, "target": max_last}
            logging.warning(f"key '{k}' will be padded on axis -1 to target={max_last}. shapes={entries}")

        self.pad_specs = pad_specs
        self.disabled_features.update(mismatched_disable)

    
    def _get_motion_dynamics_basis(self, intrinsic_matrix, cam_to_world):
        """
        intrinsic_matrix: (3,3) or (B,3,3)
        cam_to_world: (4,4) or (B,4,4), T_{w<-c} (world_from_camera)

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
        self,
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
                uv = project_world_point_to_pixel_cam_to_world(K_np, c2w_np, p_world, realworld=realworld)
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

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds. Dataset length is {len(self)}.")
        # Determine which dataset to get an item from based on the index.
        start_idx = 0
        dataset_idx = 0
        for dataset in self._datasets:
            if idx >= start_idx + dataset.num_frames:
                start_idx += dataset.num_frames
                dataset_idx += 1
                continue
            break
        else:
            raise AssertionError("We expect the loop to break out as long as the index is within bounds.")
        item = self._datasets[dataset_idx][idx - start_idx]
        try:
            extrinsic_matrix = item['extrinsic_matrix']
            intrinsic_matrix = item['intrinsic_matrix']
            robot_state = item['observation.state'] 
            img = item['observation.image'] # S * C * H * W

            if self.realworld:
                if self.use_plucker:
                    with torch.no_grad():
                        intrinsic_tensor = intrinsic_matrix.unsqueeze(0).expand(img.shape[0], -1, -1)
                        extrinsic_tensor = extrinsic_matrix.unsqueeze(0).expand(img.shape[0], -1, -1)
                        plucker_data = self.plucker_embedder(intrinsic_tensor, extrinsic_tensor)
                        plucker_tensor = einops.rearrange(plucker_data['plucker'], 's h w c -> s c h w')
                    item['observation.image'] = torch.cat([img, plucker_tensor], dim=1)
                    print("plucker shape:", plucker_tensor.shape)
                    print("image shape:", img.shape)
                elif self.use_dynamics_basis:
                    with torch.no_grad():
                        motion_dynamics_basis = self._get_motion_dynamics_basis(intrinsic_matrix, cam_to_world=extrinsic_matrix).reshape(-1)
                        axis_tensor, origin_xy = self._make_motion_basis_axis_rgb_tensor_cam_to_world(
                            rgb_tensor=img,                  # (B, 3,H,W)
                            motion_dynamics_basis=motion_dynamics_basis,
                            cam_to_world=extrinsic_matrix,                  # cam_pose = cam_to_world (고정)
                            intrinsic_matrix=intrinsic_matrix,
                            robot_eef_abs_poses=robot_state[:, :7],  # eef pose (B, 7)
                            origin_robot=True,
                            origin_fallback="pp",
                            arrow_len=60,
                            return_overlay=True,
                            realworld=True,
                        ) # (B, 3, H, W)
                        try:
                            wrist_img = item['observation.wrist_image']
                            wrist_intrinsic_matrix = item['wrist_intrinsic_matrix']
                            T_tcp_cam = item['wrist_extrinsic_matrix']
                            tip_pose = item['observation.state'] # curr_pos
                            tcp_pose = item['observation.tcp_pose'] # curr_tcp_pose
                            
                            B = wrist_img.shape[0]
                            wrist_out = []
                            for b in range(B):
                                tcp_pose_rv = tcp_pose_euler_to_rv(tcp_pose[b])
                                if not hasattr(self, "_prev_rpy"):
                                    self._prev_rpy = None

                                tcp_pose_rv, curr_rpy = tcp_pose_euler_to_rv(tcp_pose[b], prev_rpy=self._prev_rpy)
                                self._prev_rpy = curr_rpy

                                T_b_tcp = pose6d_to_T(tcp_pose_rv)
                                T_b_tcp = torch.from_numpy(T_b_tcp).to(T_tcp_cam.device).type_as(T_tcp_cam)
                                T_b_cam = T_b_tcp @ T_tcp_cam
                                wrist_extrinsic_matrix = T_b_cam
                                wrist_motion_dynamics_basis = _get_motion_dynamics_basis(wrist_intrinsic_matrix, cam_to_world=wrist_extrinsic_matrix).reshape(-1)
                                wrist_axis_tensor, wrist_origin_xy = _make_motion_basis_axis_rgb_tensor_cam_to_world(
                                    rgb_tensor=wrist_img[b:b+1],               # (B, 3,H,W)
                                    motion_dynamics_basis=wrist_motion_dynamics_basis,
                                    cam_to_world=wrist_extrinsic_matrix,                  # cam_pose = cam_to_world (고정)
                                    intrinsic_matrix=wrist_intrinsic_matrix,
                                    robot_eef_abs_poses=tip_pose[b, :7],  # eef pose (B, 7)
                                    origin_robot=False,
                                    origin_fallback="pp",
                                    arrow_len=60,
                                    return_overlay=True,
                                    realworld=True,
                                    wrist=True,
                                ) # (B, 3, H, W)
                                wrist_out.append(torch.cat([wrist_img[b:b+1], wrist_axis_tensor], dim=1))
                                save_rgb_image(wrist_axis_tensor[0], "tmp_dir/wrist_axis_tensor.png")
                            item['observation.wrist_image'] = torch.cat(wrist_out, dim=0)
                        except:
                            print("No wrist camera info")
                            pass
                    # save_rgb_image(item['observation.image'][0], "tmp_dir/robot_image.png")
                    item['observation.image'] = torch.cat([img, axis_tensor], dim=1)
            else:
                if self.use_plucker:
                    with torch.no_grad():
                        plucker_extrinsic_matrix = remove_extrinsic_camera_axis_correction(extrinsic_matrix)
                        intrinsic_tensor = intrinsic_matrix.unsqueeze(0).expand(img.shape[0], -1, -1)
                        extrinsic_tensor = plucker_extrinsic_matrix.unsqueeze(0).expand(img.shape[0], -1, -1)
                        plucker_data = self.plucker_embedder(intrinsic_tensor, extrinsic_tensor)
                        plucker_tensor = einops.rearrange(plucker_data['plucker'], 's h w c -> s c h w')
                    item['observation.image'] = torch.cat([img, plucker_tensor], dim=1)
                
                elif self.use_dynamics_basis:
                    with torch.no_grad():
                        plucker_extrinsic_matrix = remove_extrinsic_camera_axis_correction(extrinsic_matrix)
                        motion_dynamics_basis = self._get_motion_dynamics_basis(intrinsic_matrix, cam_to_world=plucker_extrinsic_matrix).reshape(-1)
                        axis_tensor, origin_xy = self._make_motion_basis_axis_rgb_tensor_cam_to_world(
                            rgb_tensor=img,                  # (B, 3,H,W)
                            motion_dynamics_basis=motion_dynamics_basis,
                            cam_to_world=plucker_extrinsic_matrix,                  # cam_pose = cam_to_world (고정)
                            intrinsic_matrix=intrinsic_matrix,
                            robot_eef_abs_poses=robot_state[:, -7:],  # eef pose (B, 7)
                            origin_robot=True,
                            origin_fallback="pp",
                            arrow_len=60,  
                            return_overlay=False,
                        ) # (B, 3, H, W)
                    # save_rgb_image(axis_tensor[0], "tmp_dir/axis_tensor.png")
                    # save_rgb_image(item['observation.image'][0], "tmp_dir/robot_image.png")
                    item['observation.image'] = torch.cat([img, axis_tensor], dim=1)                
                
        except Exception as e:
            print(e)
        if self.pretrain_dynamic_backbone:
            ang = self.indices_to_angle[dataset_idx]               # e.g., 270.0
            cls = self.angle_to_class[ang]                         # e.g., 5 (0-based)
            item["angle_class"] = torch.tensor(cls, dtype=torch.long)
        item["action"], augmented_info = self.augment_action_sequence(item["action"])
        item["augmented_info"] = augmented_info
        if self.use_dynamic_feature:
            # 1) 기준 아이템과 같은 angle의 데이터셋 후보 뽑기
            ref_angle = self.indices_to_angle[dataset_idx]
            # candidates = [
            #     i for i in self.angle_to_indices.get(ref_angle, [])
            #     if self._datasets[i].num_frames > self.window_size
            # ]
            # if not candidates:
            #     candidates = [dataset_idx]  # 폴백
            candidates = self.angle_to_indices[ref_angle]
            # 2) 후보 중에서 "데이터셋 인덱스"를 랜덤으로 3번 선택 (중복 허용)
            #    - 중복 허용이므로 len(candidates) < 3 여도 문제 없음
            chosen_ds_idxs = random.choices(candidates, k=self.num_dynamic_feature)

            images, actions, src_indices = [], [], []
            for ds_i in chosen_ds_idxs:
                ds = self._datasets[ds_i]
                # 시작 프레임 랜덤 선택 (window_size 보장)
                max_start = ds.num_frames - self.window_size
                start = random.randrange(max_start) if max_start > 0 else 0
                next_idx = start + self.window_size  # 이미지 비교용
                repo_name_i = self.repo_ids[ds_i]
                img_delta_ts = self.delta_timestamps[repo_name_i]["observation.image"].index(0.0)
                act_delta_ts = self.delta_timestamps[repo_name_i]["action"].index(0.0)

                # 두 시점 이미지를 스택
                img_seq = torch.stack([
                    ds[start]["observation.image"][img_delta_ts],
                    ds[next_idx]["observation.image"][img_delta_ts],
                ])
                images.append(img_seq)

                # 액션 시퀀스 (평균 사용 옵션 유지)
                act_seq = ds[start]["action"][act_delta_ts: act_delta_ts + self.window_size - 1]
                if self.use_action_avg:
                    act_seq = torch.mean(act_seq, dim=0)
                actions.append(act_seq)

                src_indices.append(ds_i)

            item["dynamic.image"] = torch.stack(images)
            item["dynamic.action"] = torch.stack(actions)
            item["dynamic.src_dataset_indices"] = torch.tensor(src_indices)

            # 기존 증강 로직 그대로
            item["dynamic.action"], dynamic_augmented_info = self.augment_action_sequence(
                item["dynamic.action"], info=augmented_info
            )
            item["dynamic.augmented_info"] = dynamic_augmented_info

        item["dataset_index"] = torch.tensor(dataset_idx)
        item = self.pad_item_inplace(item)
        for data_key in self.disabled_features:
            if data_key in item:
                del item[data_key]
        return item

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(\n"
            f"  Repository IDs: '{self.repo_ids}',\n"
            f"  Number of Samples: {self.num_frames},\n"
            f"  Number of Episodes: {self.num_episodes},\n"
            f"  Type: {'video (.mp4)' if self.video else 'image (.png)'},\n"
            f"  Recorded Frames per Second: {self.fps},\n"
            f"  Camera Keys: {self.camera_keys},\n"
            f"  Video Frame Keys: {self.video_frame_keys if self.video else 'N/A'},\n"
            f"  Transformations: {self.image_transforms},\n"
            f")"
        )

def T_inv(T):
    Rm = T[:3,:3]
    t = T[:3, 3]
    Ti = np.eye(4)
    Ti[:3,:3] = Rm.T
    Ti[:3, 3] = -Rm.T @ t
    return Ti

def make_T_tcp_tip(gripper_length=0.23):
    T = np.eye(4)
    T[:3, 3] = np.array([0.0, 0.0, gripper_length])  # TCP 로컬 z축
    return T

def pose6d_to_T(pose6):
        """
        pose6: [x,y,z, rx,ry,rz]
        rvec: Rodrigues axis-angle (rad)
        """
        pose6 = np.asarray(pose6, dtype=np.float64).reshape(6,)
        t = pose6[:3]
        rvec = pose6[3:]
        R, _ = cv2.Rodrigues(rvec)

        T = np.eye(4, dtype=np.float64)
        T[:3,:3] = R
        T[:3, 3] = t
        return T

def pose_posrotvec_to_T(pose_6d: torch.Tensor) -> torch.Tensor:
    """
    pose_6d: (..., 6) torch tensor, [x,y,z, rx,ry,rz] (rotvec)
    return: (..., 4, 4) torch tensor
    """
    assert pose_6d.shape[-1] == 6, f"Expected last dim 6, got {pose_6d.shape}"

    device = pose_6d.device
    dtype = pose_6d.dtype

    # flatten batch dims
    flat = pose_6d.reshape(-1, 6)
    t = flat[:, :3].detach().cpu().numpy()
    rv = flat[:, 3:6].detach().cpu().numpy()

    Rm = R.from_rotvec(rv).as_matrix()  # (N,3,3)

    T = np.tile(np.eye(4)[None, ...], (flat.shape[0], 1, 1))
    T[:, :3, :3] = Rm
    T[:, :3, 3] = t

    T = torch.from_numpy(T).to(device=device, dtype=dtype)
    return T.reshape(*pose_6d.shape[:-1], 4, 4)

def tcp_pose_euler_to_rv(tcp_pose, prev_rpy=None):
    """
    tcp_pose: [x,y,z, roll,pitch,yaw] (rad)
    prev_rpy: 이전 프레임의 [roll,pitch,yaw] (rad) or None
    """
    arr = np.asarray(tcp_pose, dtype=np.float64)
    x, y, z, roll, pitch, yaw = arr

    rpy = np.array([roll, pitch, yaw], dtype=np.float64)
    if prev_rpy is not None:
        rpy = unwrap_rpy(rpy, prev_rpy)

    rv = euler_2_rv(rpy[0], rpy[1], rpy[2])
    return np.concatenate([[x, y, z], rv]), rpy


def euler_2_rv(roll, pitch, yaw, eps=1e-8):
    """
    roll, pitch, yaw: radian
    return: rotation vector (3,)
    """

    # Rotation matrices
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)],
    ])

    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1],
    ])

    R = Rz @ Ry @ Rx

    # rotation matrix -> rotation vector
    trace = np.trace(R)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < eps:
        return np.zeros(3)

    rx = (R[2, 1] - R[1, 2]) / (2 * np.sin(theta))
    ry = (R[0, 2] - R[2, 0]) / (2 * np.sin(theta))
    rz = (R[1, 0] - R[0, 1]) / (2 * np.sin(theta))

    return theta * np.array([rx, ry, rz])

def unwrap_angle(curr, prev):
    """prev 기준으로 curr를 2π 주기에서 가장 가까운 값으로 이동"""
    delta = curr - prev
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
    return prev + delta

def unwrap_rpy(curr_rpy, prev_rpy):
    curr_rpy = np.asarray(curr_rpy, dtype=np.float64)
    prev_rpy = np.asarray(prev_rpy, dtype=np.float64)
    return np.array([
        unwrap_angle(curr_rpy[0], prev_rpy[0]),
        unwrap_angle(curr_rpy[1], prev_rpy[1]),
        unwrap_angle(curr_rpy[2], prev_rpy[2]),
    ], dtype=np.float64)