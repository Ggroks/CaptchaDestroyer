"""Read one image locally and optionally compare the result with a known answer."""

import argparse
from collections import defaultdict
from pathlib import Path
import re
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageOps
from tesserocr import PSM, PyTessBaseAPI

from digit_ocr import recognize_glyphs


DIGITS = "0123456789"
CODE = re.compile(r"[0-9]{4}")
UNREADABLE = "Не удалось распознать CAPTCHA"
POSITIONS = ("первой", "второй", "третьей", "четвёртой")
TESSDATA = Path(__file__).resolve().parent / "tessdata"


def load_image(path):
    with Image.open(path) as image:
        rgba = ImageOps.exif_transpose(image).convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        rgb = Image.alpha_composite(background, rgba).convert("RGB")
    width, height = rgb.size
    scale = min(3, 160 / height, 1600 / width)
    return cv2.resize(
        np.asarray(rgb),
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA,
    )


def clean_mask(mask):
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = max(4, round(mask.size * 0.0002))
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return (keep[labels] * 255).astype(np.uint8)


def pad(image):
    return cv2.copyMakeBorder(image, 18, 18, 18, 18, cv2.BORDER_CONSTANT, value=255)


def glyph_strips(mask):
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    boxes = sorted(
        (x, y, width, height)
        for x, y, width, height, area in stats[1:]
        if height >= mask.shape[0] / 3 and area >= mask.size * 0.002
    )
    if len(boxes) != 4:
        return
    if any(left[0] + left[2] > right[0] for left, right in zip(boxes, boxes[1:])):
        return

    for target_height in (36, 60):
        strip = []
        for x, y, width, height in boxes:
            glyph = mask[y : y + height, x : x + width]
            target_width = max(1, round(width * target_height / height))
            resized = cv2.resize(
                glyph, (target_width, target_height), interpolation=cv2.INTER_AREA
            )
            strip.append(resized)
            strip.append(np.zeros((target_height, 4), dtype=np.uint8))
        yield target_height, pad(cv2.bitwise_not(np.hstack(strip)))


def variants(rgb):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    border = np.concatenate((gray[0], gray[-1], gray[:, 0], gray[:, -1]))
    if np.median(border) < 128:
        gray = cv2.bitwise_not(gray)

    smooth = cv2.medianBlur(gray, 3)
    contrast = cv2.createCLAHE(clipLimit=2.0).apply(smooth)
    yield "gray", "gray", pad(smooth)

    _, otsu = cv2.threshold(smooth, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    block = max(3, min(51, (gray.shape[0] // 3) | 1))
    adaptive = cv2.adaptiveThreshold(
        contrast, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 9
    )

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[..., 1]
    chroma = rgb.max(axis=2) - rgb.min(axis=2)
    color = ((saturation >= 38) & (chroma >= 22)).astype(np.uint8) * 255
    _, strong_color = cv2.threshold(
        cv2.medianBlur(chroma, 3), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    strong_color = cv2.bitwise_and(strong_color, color)
    _, dark = cv2.threshold(smooth, 127, 255, cv2.THRESH_BINARY_INV)
    closing = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 5))

    for family, name, mask in (
        ("gray", "otsu", otsu),
        ("gray", "adaptive", adaptive),
        ("gray", "dark", dark),
        ("color", "hsv", color),
        ("color", "rgb", strong_color),
    ):
        cleaned = clean_mask(mask)
        coverage = np.count_nonzero(cleaned) / cleaned.size
        if not 0.005 <= coverage <= 0.65:
            continue
        x, y, width, height = cv2.boundingRect(cleaned)
        cropped = cv2.bitwise_not(cleaned[y : y + height, x : x + width])
        yield family, name, pad(cropped)

        if name in ("dark", "hsv"):
            joined = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, closing)
            for target_height, strip in glyph_strips(joined):
                yield family, f"{name}-{target_height}", strip


def recognize(rgb, tessdata, debug_dir):
    # Keep one score per family and code so similar filters do not multiply votes.
    evidence = defaultdict(dict)
    with PyTessBaseAPI(path=str(tessdata), lang="eng") as ocr:
        ocr.SetVariable("tessedit_char_whitelist", DIGITS)
        for family, name, pixels in variants(rgb):
            image = Image.fromarray(pixels)
            if debug_dir is not None:
                image.save(debug_dir / f"{name}.png")
            for mode in (PSM.SINGLE_LINE, PSM.SINGLE_WORD, PSM.RAW_LINE):
                ocr.SetPageSegMode(mode)
                ocr.SetImage(image)
                code = "".join(ocr.GetUTF8Text().split())
                if CODE.fullmatch(code):
                    confidence = ocr.MeanTextConf()
                    if confidence > 0:
                        evidence[code][family] = max(evidence[code].get(family, 0), confidence)
    return evidence


def describe(evidence):
    if not evidence:
        return None, UNREADABLE

    def score(code):
        return sum(25 + confidence for confidence in evidence[code].values())

    ranked = sorted(evidence, key=score, reverse=True)
    best = ranked[0]
    if len(ranked) > 1 and score(best) == score(ranked[1]):
        return None, UNREADABLE

    close = [other for other in ranked[1:] if score(best) - score(other) <= 35]
    confidence = max(evidence[best].values())
    enough_support = len(evidence[best]) >= 2 or confidence >= 90
    if enough_support and confidence >= 55 and not close:
        return best, best

    differing = [
        index
        for index in range(len(POSITIONS))
        if any(other[index] != best[index] for other in close)
    ]
    return best, low_confidence(best, differing)


def low_confidence(code, indices):
    differing = [POSITIONS[index] for index in indices]
    detail = ""
    if differing:
        preposition = "во" if differing[0] == POSITIONS[1] else "в"
        if len(differing) == 1:
            detail = f", сомнение {preposition} {differing[0]} цифре"
        else:
            positions = f"{', '.join(differing[:-1])} и {differing[-1]}"
            detail = f", сомнение {preposition} {positions} цифрах"
    return f"{code} (уверенность низкая{detail})"


def read_code(image, tessdata, debug_dir):
    code, message = describe(recognize(image, tessdata, debug_dir))
    if code is not None and message == code:
        return code, message

    with PyTessBaseAPI(path=str(tessdata), lang="eng") as ocr:
        ocr.SetVariable("tessedit_char_whitelist", DIGITS)
        separate, uncertain = recognize_glyphs(image, ocr, debug_dir)
    if separate is None:
        return code, message
    if code is None:
        return separate, low_confidence(separate, uncertain)

    # Refine a weak line reading when the separate reading agrees on three places.
    differences = [index for index, digit in enumerate(code) if digit != separate[index]]
    if len(differences) == 1 and differences[0] not in uncertain:
        return separate, low_confidence(separate, differences)
    return code, message


def main():
    parser = argparse.ArgumentParser(description="Распознавание четырёх цифр на изображении")
    parser.add_argument("image", type=Path, help="путь к изображению")
    parser.add_argument("--expected", metavar="0000", help="эталон для проверки результата")
    parser.add_argument("--debug-dir", type=Path, help="папка для промежуточных изображений")
    parser.add_argument("--tessdata", type=Path, default=TESSDATA, help="папка с eng.traineddata")
    args = parser.parse_args()
    if args.expected is not None and not CODE.fullmatch(args.expected):
        parser.error("эталон должен содержать ровно четыре цифры от 0 до 9")

    try:
        image = load_image(args.image)
    except (OSError, ValueError, Image.DecompressionBombError, cv2.error) as error:
        print(UNREADABLE)
        print(error, file=sys.stderr)
        return 1

    try:
        debug_dir = None
        if args.debug_dir is not None:
            args.debug_dir.mkdir(parents=True, exist_ok=True)
            debug_dir = Path(tempfile.mkdtemp(prefix="captcha-", dir=args.debug_dir))
            print(debug_dir, file=sys.stderr)
        code, message = read_code(image, args.tessdata, debug_dir)
    except (OSError, RuntimeError, cv2.error) as error:
        print(UNREADABLE)
        print(error, file=sys.stderr)
        return 1

    print(message)
    if code is None:
        return 1
    if args.expected is not None:
        if code != args.expected:
            print("Результат расходится с эталоном", file=sys.stderr)
            return 2
        print("Результат совпал с эталоном", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    sys.exit(main())
