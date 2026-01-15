
# IMPORTANT: Set `--dataset.root` to your lerobot-formatted dataset path.
# `--dataset.repo_id`: the two sub-datasets forming the islands; the example below uses camera positions 40%→40% and 60%→60% (Diffusion Policy diversity setting 1).
# Note: You may need to log in to Weights & Biases (wandb) if enabled.
#   --dataset.repo_id=[xyg_20_10_15.0_65.0/v-0.400-0.400_num1,xyg_20_10_15.0_65.0/v-0.600-0.600_num5] \
# --dataset.repo_id=[xyg_10_10_0.0_0.0/v-1.000-1.000_num1,xyg_10_10_0.0_0.0/v-1.000-1.000_num5,xyg_10_10_45.0_45.0/v-1.000-1.000_num1,xyg_10_10_45.0_45.0/v-1.000-1.000_num5,xyg_10_10_90.0_90.0/v-1.000-1.000_num1,xyg_10_10_90.0_90.0/v-1.000-1.000_num5,xyg_10_10_135.0_135.0/v-1.000-1.000_num1,xyg_10_10_135.0_135.0/v-1.000-1.000_num5,xyg_10_10_225.0_225.0/v-1.000-1.000_num1,xyg_10_10_225.0_225.0/v-1.000-1.000_num5,xyg_10_10_270.0_270.0/v-1.000-1.000_num1,xyg_10_10_270.0_270.0/v-1.000-1.000_num5,xyg_10_10_315.0_315.0/v-1.000-1.000_num1,xyg_10_10_315.0_315.0/v-1.000-1.000_num5] \
#   --axis_augmentation=true \
# --sign_augmentation='[false, false, true]'
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd -P)"
REPO_ROOT="${SCRIPT_DIR}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH}"

CUDA_VISIBLE_DEVICES=3 python lerobot/scripts/train.py \
  --dataset.repo_id=[DP_wrist_cam_info_10_10_0.0_0.0/v-1.000-1.000_num1] \
  --dataset.root=/data1/local/shortcut-learning-in-grps/dataset_git/libero_spatial_no_noops_island_1_lerobot \
  --dataset.image_transforms.enable=false \
  --dataset.use_imagenet_stats=false \
  --dataset.split_episodes=false \
  --policy.type=diffusion \
  --policy.n_obs_steps=2 \
  --policy.horizon=16 \
  --policy.n_action_steps=16 \
  --policy.use_robot_state=false \
  --policy.vision_backbone=resnet18 \
  --policy.xyg_resize_shape=[84,84] \
  --policy.use_dynamic_feature=false \
  --policy.use_normalize_for_action=true \
  --policy.image_channel=6 \
  --policy.use_separate_rgb_encoder_per_camera=true \
  --policy.use_wrist_image=true \
  --steps=90000 \
  --save_freq=1000 \
  --batch_size=4 \
  --wandb.enable=false \
  --num_workers=0 \
  --use_plucker=false \
  --use_dynamics_basis=true \
  --apply_basis_scale=false
# Training checkpoints will be saved under: lerobot/outputs/train/202x-xx-xx/xx-xx-xx_diffusion