"""Turning class-id masks into color images."""

import numpy as np

from .classes import CLASSES


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """'#FFB6C1' -> (255, 182, 193)."""
    value = hex_color.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


# Row i is the RGB color of class i.
PALETTE = np.array([hex_to_rgb(cls.color) for cls in CLASSES], dtype=np.uint8)


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map an (H, W) mask of class ids to an (H, W, 3) RGB image."""
    return PALETTE[mask]
