"""Safe loading of user images of any size and format."""

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import MAX_IMAGE_PIXELS, MAX_IMAGE_SIDE, MIN_IMAGE_SIDE

# Let Pillow raise an error early for absurdly large files ("decompression bombs").
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


class ImageLoadError(ValueError):
    """Raised with a user-friendly message when an image can't be used."""


@dataclass
class LoadedImage:
    image: Image.Image            # RGB image, ready for the model
    original_size: tuple[int, int]  # (width, height) of the uploaded file

    @property
    def was_downscaled(self) -> bool:
        return self.image.size != self.original_size

    @property
    def is_small(self) -> bool:
        return max(self.original_size) < MIN_IMAGE_SIDE


def load_image(file) -> LoadedImage:
    """Open an uploaded file and return an RGB image of a safe size."""
    try:
        image = Image.open(file)
    except Image.DecompressionBombError:
        raise ImageLoadError(_too_large_message()) from None
    except (UnidentifiedImageError, OSError):
        raise ImageLoadError(
            "Не удалось прочитать файл. Поддерживаются обычные изображения "
            "JPG, PNG и TIFF (1, 3 или 4 канала)."
        ) from None

    width, height = image.size
    if width * height > MAX_IMAGE_PIXELS:
        raise ImageLoadError(_too_large_message(width, height))

    try:
        image = ImageOps.exif_transpose(image)  # fix rotated photos
        if image.mode.startswith("I;16"):
            image = image.convert("I")          # 16-bit -> 32-bit int, resizable
        original_size = image.size
        image.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))  # only ever shrinks
        image = to_rgb(image)
    except (OSError, ValueError):
        raise ImageLoadError("Файл изображения поврежден или имеет неподдерживаемый формат.") from None

    return LoadedImage(image=image, original_size=original_size)


def to_rgb(image: Image.Image) -> Image.Image:
    """Convert any Pillow image mode to 8-bit RGB."""
    # 16/32-bit integer or float images (common for satellite TIFFs):
    # stretch the values to 0-255 instead of clipping them.
    if image.mode in ("I", "F"):
        values = np.asarray(image, dtype=np.float32)
        low, high = np.percentile(values, (1, 99))
        scaled = np.clip((values - low) / max(high - low, 1e-6), 0, 1) * 255
        image = Image.fromarray(scaled.astype(np.uint8), mode="L")

    # Transparent images: put them on a white background.
    if image.mode in ("RGBA", "LA", "PA") or (image.mode == "P" and "transparency" in image.info):
        image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, image)

    return image.convert("RGB")


def display_size(size: tuple[int, int], min_side: int = MIN_IMAGE_SIDE) -> tuple[int, int]:
    """Size used to show the image and mask: small images are upscaled, aspect ratio kept."""
    width, height = size
    scale = max(1.0, min_side / max(width, height))
    return max(1, round(width * scale)), max(1, round(height * scale))


def _too_large_message(width: int | None = None, height: int | None = None) -> str:
    size = f" ({width}×{height})" if width and height else ""
    max_side = int(MAX_IMAGE_PIXELS ** 0.5)
    return (
        f"Изображение слишком большое{size}. Максимум около "
        f"{max_side}×{max_side} пикселей. Уменьшите его и попробуйте снова."
    )
