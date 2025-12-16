conda create -y -n lerobot python=3.10
conda activate lerobot
pip install -e .
pip install torch==2.6.0 torchvision==0.21.0 torchcodec==0.2.1
pip install datasets==3.4.1 huggingface-hub==0.29.2 draccus==0.10.0 diffusers==0.32.2 transformers==4.50.3