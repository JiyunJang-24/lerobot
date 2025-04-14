import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7,"
from pprint import pprint
import numpy as np
import torch
from huggingface_hub import HfApi

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from pathlib import Path
from PIL import Image
from PIL import ImageDraw, ImageFont
from tqdm import tqdm

def main():
    # bridge_dataset_1.0.0_lerobot  fractal20220817_data_0.1.0_lerobot
    base_path = r'/mnt/hdd3/xingyouguang/datasets/robotics/oxe_lerobot'
    save_path = r'./save_images'
    
    os.makedirs(save_path, exist_ok=True)
    
    need_rt1 = False
    need_bridge = True
    
    if need_rt1:
        repo_id = 'xyg/fractal20220817_data'
        rt1_dataset = LeRobotDataset(
            repo_id=repo_id,
            root=f"{base_path}/{repo_id}",
        )
        os.makedirs(f"{save_path}/{repo_id}", exist_ok=True)

        for st in tqdm(rt1_dataset.episode_data_index['from']):
            cur_item = rt1_dataset[st.item()]
            language = cur_item['task']
            if 'fridge' not in language:
                image_numpy = (cur_item['observation.images.image'].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                image_pil = Image.fromarray(image_numpy)
                # pil write text (language)
                # write language to the image as font
                font = ImageFont.truetype("arial.ttf", 18)
                draw = ImageDraw.Draw(image_pil)
                draw.text((10, 10), language, fill=(255, 255, 255), font=font)
                image_pil.save(f"{save_path}/{repo_id}/{st}-{language}.png")
    
    if need_bridge:
        repo_id = 'xyg/bridge_dataset'
        bridge_dataset = LeRobotDataset(
            repo_id=repo_id,
            root=f"{base_path}/{repo_id}",
        )

        os.makedirs(f"{save_path}/{repo_id}", exist_ok=True)
        for st in tqdm(bridge_dataset.episode_data_index['from']):
            cur_item = bridge_dataset[st.item()]
            language = cur_item['task']
            if "put" in language and "plate" in language:
                image_numpy = (cur_item['observation.images.image_0'].cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                image_pil = Image.fromarray(image_numpy)
                # pil write text (language)
                # write language to the image as font
                font = ImageFont.truetype("arial.ttf", 18)
                draw = ImageDraw.Draw(image_pil)
                draw.text((10, 10), language, fill=(255, 255, 255), font=font)
                image_pil.save(f"{save_path}/{repo_id}/{st}-{language}.png")
            
    import ipdb; ipdb.set_trace()
    print('this is a test')


if __name__ == "__main__":
    main()
    