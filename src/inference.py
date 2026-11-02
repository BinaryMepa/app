"""Loading the model and predicting segmentation masks."""

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from .classes import NUM_CLASSES
from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from .model import UNet

# Resize first (on the PIL image), so huge images never become huge tensors.
_transform = T.Compose(
    [
        T.Resize(IMAGE_SIZE),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
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
def predict_mask(
    model: UNet,
    image: Image.Image,
    output_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return an (H, W) array of class ids for the given image.

    The model always works at 512 x 512. The mask is then resized to
    `output_size` (width, height), or to the image size if not given,
    so it lines up with the original image and keeps its aspect ratio.
    """
    device = next(model.parameters()).device
    logits = model(preprocess(image).to(device))
    mask = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    output_size = output_size or image.size
    if (mask.shape[1], mask.shape[0]) != tuple(output_size):
        # NEAREST keeps class ids intact (no blending between classes).
        mask = np.asarray(Image.fromarray(mask).resize(output_size, Image.NEAREST))
    return mask
