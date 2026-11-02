"""Project-wide settings."""

from pathlib import Path

# Repository root (the folder that contains app.py).
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Trained weights (state_dict), stored with Git LFS.
MODEL_PATH = PROJECT_ROOT / "segmentation_model.pth"

# Input size the network was trained on.
IMAGE_SIZE = (512, 512)

# ImageNet normalization statistics used during training.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# File types accepted by the uploader.
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "tif", "tiff"]

# --- Image size limits -------------------------------------------------------

# Images with more pixels than this are rejected before decoding
# (about 7000 x 7000). Protects the app from running out of memory.
MAX_IMAGE_PIXELS = 50_000_000

# Large images are downscaled so their longest side is at most this value.
# The model works at 512 x 512 anyway, so this only affects display quality.
MAX_IMAGE_SIDE = 2048

# Images smaller than this (longest side) are upscaled for display, and the
# user is warned that the prediction may be less accurate.
MIN_IMAGE_SIDE = 512
