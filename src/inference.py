"""Loading the model and predicting segmentation masks."""

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from .classes import NUM_CLASSES
from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from .model import UNet

_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        T.Resize(IMAGE_SIZE),
    ]
)


def load_model(model_path: str | Path, device: str = "cpu") -> UNet:
    """Build the U-Net and load trained weights from a state_dict file."""
    model = UNet(n_classes=NUM_CLASSES)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def preprocess(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to a normalized (1, 3, H, W) tensor."""
    return _transform(image.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def predict_mask(model: UNet, image: Image.Image) -> np.ndarray:
    """Return an (H, W) array of class ids for the given image."""
    device = next(model.parameters()).device
    logits = model(preprocess(image).to(device))
    return logits.argmax(dim=1).squeeze(0).cpu().numpy()
