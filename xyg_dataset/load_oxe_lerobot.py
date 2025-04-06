import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7,"
from pprint import pprint

import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def main():
    base_path = r'/mnt/nfs/CMG/xiejunlin/datasets/Robotics/oxe_lerobot'


if __name__ == "__main__":
    main()
    