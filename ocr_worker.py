"""Run one image through the existing recognizer in an isolated process."""

from enum import IntEnum
from io import BytesIO
from pathlib import Path
import sys


IMAGE_FORMATS = {"PNG", "JPEG", "WEBP", "BMP"}
MAX_IMAGE_PIXELS = 16_000_000


class WorkerExit(IntEnum):
    UNSUPPORTED_IMAGE = 10
    INVALID_IMAGE = 11
    IMAGE_TOO_LARGE = 12
    RECOGNITION_FAILED = 13


def recognize_image(data: bytes, tessdata: Path) -> str | WorkerExit:
    from PIL import Image

    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in IMAGE_FORMATS or getattr(image, "is_animated", False):
                return WorkerExit.UNSUPPORTED_IMAGE
            if image.width * image.height > MAX_IMAGE_PIXELS:
                return WorkerExit.IMAGE_TOO_LARGE
            image.verify()
    except Image.DecompressionBombError:
        return WorkerExit.IMAGE_TOO_LARGE
    except (OSError, EOFError, SyntaxError, ValueError):
        return WorkerExit.INVALID_IMAGE

    import cv2
    from captcha import CODE, load_image, read_code

    try:
        pixels = load_image(BytesIO(data))
    except (OSError, EOFError, SyntaxError, ValueError, cv2.error):
        return WorkerExit.INVALID_IMAGE

    text, _ = read_code(pixels, tessdata, debug_dir=None)
    if text is None or not CODE.fullmatch(text):
        return WorkerExit.RECOGNITION_FAILED
    return text


def main() -> int:
    result = recognize_image(sys.stdin.buffer.read(), Path(sys.argv[1]))
    if isinstance(result, WorkerExit):
        return result
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
