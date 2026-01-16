
# IMPORTANT: Set `--dataset.root` to your lerobot-formatted dataset path.
# `--dataset.repo_id`: the two sub-datasets forming the islands; the example below uses camera positions 40%→40% and 60%→60% (Diffusion Policy diversity setting 1).
# Note: You may need to log in to Weights & Biases (wandb) if enabled.


# CUDA_VISIBLE_DEVICES=0 python lerobot/scripts/train.py \
#   --dataset.repo_id=[yujin/ur5_plush_pickup] \
#   --dataset.root=/home/vai/Desktop/yujin/shortcut-learning-in-grps/dataset_git \
#   --dataset.image_transforms.enable=false \
#   --dataset.use_imagenet_stats=false \
#   --dataset.split_episodes=false \
#   --policy.type=diffusion \
#   --policy.n_obs_steps=2 \
#   --policy.horizon=16 \
#   --policy.n_action_steps=16 \
#   --policy.use_robot_state=false \
#   --policy.vision_backbone=resnet18 \
#   --policy.xyg_resize_shape=[84,84] \
#   --policy.use_language=false \
#   --steps=50000 \
#   --save_freq=5000 \
#   --batch_size=128 \
#   --wandb.enable=true \
#   --wandb.project=realworld_DP \
#   --wandb.disable_artifact=true \
#   --wandb.entity=DynamicVLA \
#   --job_name=realworld_DP \
#   --use_plucker=false \
#   --use_dynamics_basis=false \
#   --realworld true
# # # # Training checkpoints will be saved under: lerobot/outputs/train/202x-xx-xx/xx-xx-xx_diffusion

# CUDA_VISIBLE_DEVICES=0 python lerobot/scripts/train.py \
#   --dataset.repo_id=[yujin/ur5_plush_pickup] \
#   --dataset.root=/home/vai/Desktop/yujin/shortcut-learning-in-grps/dataset_git \
#   --dataset.image_transforms.enable=false \
#   --dataset.use_imagenet_stats=false \
#   --dataset.split_episodes=false \
#   --policy.type=diffusion \
#   --policy.n_obs_steps=2 \
#   --policy.horizon=16 \
#   --policy.n_action_steps=16 \
#   --policy.use_robot_state=false \
#   --policy.vision_backbone=resnet18 \
#   --policy.xyg_resize_shape=[84,84] \
#   --policy.use_language=false \
#   --policy.image_channel=6 \
#   --steps=50000 \
#   --save_freq=5000 \
#   --batch_size=128 \
#   --wandb.enable=true \
#   --wandb.project=realworld_DP \
#   --wandb.disable_artifact=true \
#   --wandb.entity=DynamicVLA \
#   --job_name=realworld_DP_basis \
#   --use_plucker=false \
#   --use_dynamics_basis=true \
#   --realworld true

  CUDA_VISIBLE_DEVICES=0 python lerobot/scripts/train.py \
  --dataset.repo_id=[yujin/ur5_plush_pickup] \
  --dataset.root=/home/vai/Desktop/yujin/shortcut-learning-in-grps/dataset_git \
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
  --policy.use_language=false \
  --policy.image_channel=9 \
  --steps=50000 \
  --save_freq=5000 \
  --batch_size=128 \
  --wandb.enable=true \
  --wandb.project=realworld_DP \
  --wandb.disable_artifact=true \
  --wandb.entity=DynamicVLA \
  --job_name=realworld_DP_pluker \
  --use_plucker=true \
  --use_dynamics_basis=false \
  --realworld true