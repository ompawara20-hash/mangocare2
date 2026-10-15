"""Decode and sanity-check uploads. Trusts decoded content, never filename/MIME."""
import io
from PIL import Image, ImageOps, UnidentifiedImageError
from ..config import settings


class ImageError(Exception):
    def __init__(self, code: str, message: str, status: int = 400):
        self.code, self.message, self.status = code, message, status


def load_image(data: bytes) -> Image.Image:
    if not data:
        raise ImageError("NO_FILE", "No file content was received.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise ImageError("FILE_TOO_LARGE", f"File exceeds {settings.max_upload_mb} MB.", 413)
    Image.MAX_IMAGE_PIXELS = settings.max_pixels  # decompression-bomb guard
    try:
        with Image.open(io.BytesIO(data)) as probe:
            if probe.format not in {"JPEG", "PNG", "WEBP"}:
                raise ImageError("UNSUPPORTED_TYPE", "Only JPEG, PNG and WebP images are supported.", 415)
            probe.verify()
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)  # fix phone-camera rotation
        img = img.convert("RGB")
    except ImageError:
        raise
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError, SyntaxError):
        raise ImageError("CORRUPT_IMAGE", "The file could not be read as a valid image.", 422)
    if min(img.size) < settings.min_side:
        raise ImageError("IMAGE_TOO_SMALL", f"Image is too small (min {settings.min_side}px per side).", 422)
    return img
