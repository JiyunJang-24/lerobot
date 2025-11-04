
# IMPORTANT: Set `--dataset.root` to your lerobot-formatted dataset path.
# `--dataset.repo_id`: the two sub-datasets forming the islands; the example below uses camera positions 40%→40% and 60%→60% (Diffusion Policy diversity setting 1).
# Note: You may need to log in to Weights & Biases (wandb) if enabled.
#   --dataset.repo_id=[xyg_20_10_15.0_65.0/v-0.400-0.400_num1,xyg_20_10_15.0_65.0/v-0.600-0.600_num5] \
  # --dataset.repo_id=[xyg_10_10_0.0_0.0/v-1.000-1.000_num1,xyg_10_10_0.0_0.0/v-1.000-1.000_num5,xyg_10_10_45.0_45.0/v-1.000-1.000_num1,xyg_10_10_45.0_45.0/v-1.000-1.000_num5,xyg_10_10_90.0_90.0/v-1.000-1.000_num1,xyg_10_10_90.0_90.0/v-1.000-1.000_num5,xyg_10_10_135.0_135.0/v-1.000-1.000_num1,xyg_10_10_135.0_135.0/v-1.000-1.000_num5,xyg_10_10_225.0_225.0/v-1.000-1.000_num1,xyg_10_10_225.0_225.0/v-1.000-1.000_num5,xyg_10_10_270.0_270.0/v-1.000-1.000_num1,xyg_10_10_270.0_270.0/v-1.000-1.000_num5,xyg_10_10_315.0_315.0/v-1.000-1.000_num1,xyg_10_10_315.0_315.0/v-1.000-1.000_num5] \

CUDA_VISIBLE_DEVICES=1 python lerobot/scripts/train_dynamic.py \
  --pretrained_path="/data1/local/shortcut-learning-in-grps/lerobot/outputs/train/2025-11-01/06-23-32_DP_ex1_angle_from_0_135_225/checkpoints/045000/pretrained_model" \
  --dataset.repo_id=[xyg_10_10_45.0_45.0/v-1.000-1.000_num1,xyg_10_10_45.0_45.0/v-1.000-1.000_num5,xyg_10_10_90.0_90.0/v-1.000-1.000_num1,xyg_10_10_90.0_90.0/v-1.000-1.000_num5,xyg_10_10_270.0_270.0/v-1.000-1.000_num1,xyg_10_10_270.0_270.0/v-1.000-1.000_num5,xyg_10_10_315.0_315.0/v-1.000-1.000_num1,xyg_10_10_315.0_315.0/v-1.000-1.000_num5] \
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
  --policy.use_dynamic_feature=true \
  --policy.num_dynamic_feature=3 \
  --policy.train_dynamic_with_frozen_dp=true \
  --use_action_avg=true \
  --window_size=5 \
  --steps=45000 \
  --save_freq=5000 \
  --batch_size=32 \
  --wandb.enable=true \
  --wandb.project=libero_DP \
  --wandb.disable_artifact=true \
  --wandb.entity=DynamicVLA \
  --job_name=Dynamic_DP_45_90_275_315_from_frozen_DP_ex1_angle_from_0_135_225_cond_denoise_action_direct_action

# Training checkpoints will be saved under: lerobot/outputs/train/202x-xx-xx/xx-xx-xx_diffusion