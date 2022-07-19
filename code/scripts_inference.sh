#===============================================================================
# Inference Scripts
#===============================================================================

# ==== KITTI Val 1 Split ====
# GUP Net
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/config_run_201_a100_v0_1.yaml --resume_model output/config_run_201_a100_v0_1/checkpoints/checkpoint_epoch_140.pth -e
# DEVIANT
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/run_221.yaml                  --resume_model output/run_221/checkpoints/checkpoint_epoch_140.pth -e

# ==== KITTI Full Split  ====
# DEVIANT
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/run_250.yaml                  --resume_model output/run_250/checkpoints/checkpoint_epoch_140.pth -e

# ==== Waymo Val Split  ====
# GUP Net
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/run_1050.yaml                 --resume_model output/run_1050/checkpoints/checkpoint_epoch_30.pth -e
# DEVIANT
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/run_1051.yaml                 --resume_model output/run_1051/checkpoints/checkpoint_epoch_30.pth -e

# === nuScenes Val Cross Dataset Evaluation on KITTI Val 1 ===
# Change eval_dataset to "nusc_kitti" and resolution to [672, 384] in config files
# GUP Net
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/config_run_201_a100_v0_1.yaml --resume_model output/config_run_201_a100_v0_1/checkpoints/checkpoint_epoch_140.pth -e
# DEVIANT
CUDA_VISIBLE_DEVICES=0 python -u tools/train_val.py --config=experiments/run_221.yaml                  --resume_model output/run_221/checkpoints/checkpoint_epoch_140.pth -e
