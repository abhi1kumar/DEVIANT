#===============================================================================
# Training Scripts
#===============================================================================

# ==== KITTI Val 1 Split ====
# GUP Net
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/config_run_201_a100_v0_1.yaml
# DEVIANT
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/run_221.yaml

# ==== KITTI Full Split  ====
# DEVIANT
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/run_250.yaml

# ==== Waymo Val Split  ====
# Change val_split_name from 'val' to 'val_small' in waymo configs for quicker validation performance. We had used val_small.
# GUP Net
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/run_1050.yaml
# DEVIANT
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/run_1051.yaml


#===============================================================================
# Ablation Studies
#===============================================================================
# GUP Net without Scale Augmentation
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/run_246.yaml
# DEVIANT without Scale Augmentation
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/run_247.yaml

# DCNN
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/run_260_1.yaml

# GUP Net with (bigger) DLA102 and DLA169 backbones 
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/gup_dla102.yaml
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/gup_dla169.yaml

# GUP Net vs DEVIANT on ResNet-18 backbone
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/gup_resnet18.yaml
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/dev_resnet18.yaml
