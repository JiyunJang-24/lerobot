import numpy as np
import tensorflow_datasets as tfds
from PIL import Image
from IPython import display
from tqdm import tqdm
import argparse


def dataset2path(dataset_name, base_path=r'/mnt/hdd3/xingyouguang/datasets/robotics/oxe'):
    if dataset_name == 'robo_net':
        version = '1.0.0'
    elif dataset_name == 'language_table':
        version = '0.0.1'
    elif dataset_name == 'bridge_dataset':
        version = '1.0.0'
    else:
        version = '0.1.0'
    return f'{base_path}/{dataset_name}/{version}'


def as_gif(images, path='temp.gif'):
    # Render the images as the gif:
    images[0].save(path, save_all=True, append_images=images[1:], duration=1000, loop=0)
    gif_bytes = open(path,'rb').read()
    return gif_bytes


def check_rt1_dataset(dataset='fractal20220817_data', base_path=r'/mnt/hdd3/xingyouguang/datasets/robotics/oxe', only_show_number=False):
    display_key = 'image'
    builder = tfds.builder_from_directory(builder_dir=dataset2path(dataset, base_path))   # b.info.features['steps'] 

    ds = builder.as_dataset(split='train')
    if only_show_number:
        print(dataset, len(ds))
        return
    
    # 生成一个 txt 文件，将language写入
    with open(f'dataset_description/language_{dataset}.txt', 'w') as f:
        for episode in tqdm(ds):
            cur_step = next(iter(episode['steps']))
            language = cur_step['observation']['natural_language_instruction'].numpy().decode('utf-8')
            f.write(language + '\n')


def check_bridge_dataset(dataset='bridge_dataset', base_path=r'/mnt/hdd3/xingyouguang/datasets/robotics/oxe', only_show_number=False):
    display_key = 'image'
    builder = tfds.builder_from_directory(builder_dir=dataset2path(dataset, base_path))   # b.info.features['steps'] 

    ds = builder.as_dataset(split='train')
    if only_show_number:
        print(dataset, len(ds))
        return
    
    # 生成一个 txt 文件，将language写入
    with open(f'dataset_description/language_{dataset}.txt', 'w') as f:
        for episode in tqdm(ds):
            cur_step = next(iter(episode['steps']))
            language = cur_step['language_instruction'].numpy().decode('utf-8')
            f.write(language + '\n')
            

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='all')
    args = parser.parse_args()
    
    name_dict = {
        'rt1': 'fractal20220817_data',
        'bridge': 'bridge_dataset',
    }
    
    base_path = r'/mnt/hdd3/xingyouguang/datasets/robotics/oxe'
    # base_path = r'/mnt/nfs/CMG/xiejunlin/datasets/Robotics/oxe'
    
    if args.dataset == 'rt1':
        dataset = args.dataset # @param ['fractal20220817_data', 'kuka', 'bridge', 'taco_play', 'jaco_play', 'berkeley_cable_routing', 'roboturk', 'nyu_door_opening_surprising_effectiveness', 'viola', 'berkeley_autolab_ur5', 'toto', 'language_table', 'columbia_cairlab_pusht_real', 'stanford_kuka_multimodal_dataset_converted_externally_to_rlds', 'nyu_rot_dataset_converted_externally_to_rlds', 'stanford_hydra_dataset_converted_externally_to_rlds', 'austin_buds_dataset_converted_externally_to_rlds', 'nyu_franka_play_dataset_converted_externally_to_rlds', 'maniskill_dataset_converted_externally_to_rlds', 'furniture_bench_dataset_converted_externally_to_rlds', 'cmu_franka_exploration_dataset_converted_externally_to_rlds', 'ucsd_kitchen_dataset_converted_externally_to_rlds', 'ucsd_pick_and_place_dataset_converted_externally_to_rlds', 'austin_sailor_dataset_converted_externally_to_rlds', 'austin_sirius_dataset_converted_externally_to_rlds', 'bc_z', 'usc_cloth_sim_converted_externally_to_rlds', 'utokyo_pr2_opening_fridge_converted_externally_to_rlds', 'utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds', 'utokyo_saytap_converted_externally_to_rlds', 'utokyo_xarm_pick_and_place_converted_externally_to_rlds', 'utokyo_xarm_bimanual_converted_externally_to_rlds', 'robo_net', 'berkeley_mvp_converted_externally_to_rlds', 'berkeley_rpt_converted_externally_to_rlds', 'kaist_nonprehensile_converted_externally_to_rlds', 'stanford_mask_vit_converted_externally_to_rlds', 'tokyo_u_lsmo_converted_externally_to_rlds', 'dlr_sara_pour_converted_externally_to_rlds', 'dlr_sara_grid_clamp_converted_externally_to_rlds', 'dlr_edan_shared_control_converted_externally_to_rlds', 'asu_table_top_converted_externally_to_rlds', 'stanford_robocook_converted_externally_to_rlds', 'eth_agent_affordances', 'imperialcollege_sawyer_wrist_cam', 'iamlab_cmu_pickup_insert_converted_externally_to_rlds', 'uiuc_d3field', 'utaustin_mutex', 'berkeley_fanuc_manipulation', 'cmu_food_manipulation', 'cmu_play_fusion', 'cmu_stretch', 'berkeley_gnm_recon', 'berkeley_gnm_cory_hall', 'berkeley_gnm_sac_son']
        check_rt1_dataset(name_dict[dataset], base_path)
    elif args.dataset == 'bridge':
        dataset = args.dataset
        check_bridge_dataset(name_dict[dataset], base_path)
    elif args.dataset == 'all':
        check_rt1_dataset(name_dict['rt1'], base_path, only_show_number=True)
        check_bridge_dataset(name_dict['bridge'], base_path, only_show_number=True)

    print('done')


if __name__ == "__main__":
    main()
    
    
    