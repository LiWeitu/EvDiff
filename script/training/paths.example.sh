# Copy to paths.local.sh and edit paths. Run launchers from any directory.
export MODEL_NAME="/path/to/sd3-medium"
export PLACES365_PATH="/path/to/places365/train"  # contains gt/
# Keep REDS in the H5 paths: the baseline reader enables merging by that name.
export REDS_SMALL1="/path/to/REDS/small1"  # contains event/*.h5
export REDS_SMALL2="/path/to/REDS/small2"
export BS_ERGB_PATH="/path/to/bs_ergb/mem"
export STEP1_CHECKPOINT="/path/to/step1/checkpoint-N"  # transformer + vae safetensors
export E2VIDPLUS_CHECKPOINT="/path/to/EVREAL/pretrained/E2VID+/model.pth"
export STEP2_VOXEL_VAE="/path/to/step2/checkpoint-N/voxel_vae.pth"
export WANDB_MODE=offline
