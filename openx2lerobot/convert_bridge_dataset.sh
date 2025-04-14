python openx_rlds.py \
    --raw-dir /mnt/hdd3/xingyouguang/datasets/robotics/oxe/bridge_dataset/1.0.0 \
    --local-dir /mnt/hdd3/xingyouguang/datasets/robotics/oxe_lerobot \
    --repo-id xyg/bridge_dataset \
    --use-videos \
    --image-writer-process 32 \
    --image-writer-threads 32 \
    --filters True


# python openx_rlds.py \
#     --raw-dir /mnt/hdd3/xingyouguang/datasets/robotics/oxe/bridge_dataset/1.0.0 \
#     --local-dir /mnt/hdd3/xingyouguang/datasets/robotics/oxe_lerobot \
#     --repo-id xyg/bridge_dataset \
#     --use-videos \
#     --push-to-hub