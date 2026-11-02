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
