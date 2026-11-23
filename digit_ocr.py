"""Recognize separated color glyphs when recognition of the whole line fails."""

import os
from pathlib import Path
from string import digits

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tesserocr import PSM


TEMPLATE_FONTS = (
    "arial", "arialbd", "arialbi", "ariblk", "candarali", "calibri", "calibrib",
    "comic", "comicbd", "comicz", "consola", "consolab", "consolaz", "corbelb",
    "corbell", "corbelz", "cour", "courbd", "framd", "framdit", "georgiab",
    "georgiai", "impact", "Inkfree", "malgunbd", "msyi", "segoeprb", "segoeui",
    "seguibl", "seguisb", "timesbd", "timesbi", "verdana", "verdanab", "verdanaz",
)
HOG = cv2.HOGDescriptor((40, 40), (16, 16), (8, 8), (8, 8), 9)
KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def find_components(chroma, mask, kernel):
    closed = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    for label, (x, y, width, height, area) in enumerate(stats[1:], start=1):
        if height < chroma.shape[0] * 0.12 or area < chroma.size * 0.004:
            continue
        region = np.s_[y : y + height, x : x + width]
        component = labels[region] == label
        values = np.where(component, chroma[region], 0)
        yield (x, y, width, height), area, (values, component.astype(np.uint8) * 255)


def overlap_fraction(a, b):
    x, y, width, height = a
    u, v, other_width, other_height = b
    across = max(0, min(x + width, u + other_width) - max(x, u))
    down = max(0, min(y + height, v + other_height) - max(y, v))
    return across * down / min(width * height, other_width * other_height)


def hue_groups(rgb, chroma):
    hue = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[..., 0].astype(np.int16)
    histogram = np.bincount(hue[chroma >= 23], minlength=180)
    histogram = sum(np.roll(histogram, step) for step in range(-2, 3))
    centers = []
    radius = 12
    while histogram.max() >= max(30, chroma.size * 0.0015):
        peak = int(histogram.argmax())
        distance = np.abs(hue - peak)
        centers.append(np.minimum(distance, 180 - distance) <= radius)
        histogram[(np.arange(peak - radius, peak + radius + 1) % 180)] = 0

    # Separate touching digits when their hue differs. A hue can contain several digits.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for threshold in (13, 17, 21):
        components = [
            component
            for band in centers
            for component in find_components(chroma, band & (chroma >= threshold), kernel)
        ]
        distinct = []
        for component in sorted(components, key=lambda item: item[1], reverse=True):
            if all(overlap_fraction(component[0], saved[0]) < 0.55 for saved in distinct):
                distinct.append(component)
        yield distinct


def ordered_glyphs(groups):
    candidates = []
    for components in groups:
        if len(components) != 4:
            continue
        ordered = sorted(components, key=lambda item: item[0][0])
        centers = [x + width / 2 for (x, _, width, _), _, _ in ordered]
        if all(left < right for left, right in zip(centers, centers[1:])):
            candidates.append([glyph for _, _, glyph in ordered])
    return list(zip(*candidates))


def extract_glyphs(rgb):
    chroma = cv2.medianBlur(rgb.max(axis=2) - rgb.min(axis=2), 3)
    by_chroma = ordered_glyphs(
        list(find_components(chroma, chroma >= threshold, KERNEL))
        for threshold in (13, 16, 19, 23)
    )
    return by_chroma or ordered_glyphs(hue_groups(rgb, chroma))


def describe_shape(image):
    x, y, width, height = cv2.boundingRect(image)
    extent = max(width, height)
    if extent == 0:
        return np.zeros(HOG.getDescriptorSize(), dtype=np.float32)
    out_width = max(1, round(32 * (width / extent) ** 0.65))
    out_height = max(1, round(32 * (height / extent) ** 0.65))
    resized = cv2.resize(
        image[y : y + height, x : x + width],
        (out_width, out_height),
        interpolation=cv2.INTER_AREA,
    )
    centered = np.zeros((40, 40), dtype=np.uint8)
    left, top = (40 - out_width) // 2, (40 - out_height) // 2
    centered[top : top + out_height, left : left + out_width] = resized
    descriptor = HOG.compute(centered).ravel()
    return descriptor / max(float(np.linalg.norm(descriptor)), 1e-6)


def rotate(image, angle):
    height, width = image.shape
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    return cv2.warpAffine(image, transform, (width, height))


def template_images(digit):
    fonts = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for name in TEMPLATE_FONTS:
        try:
            font = ImageFont.truetype(str(fonts / f"{name}.ttf"), 55)
        except OSError:
            continue
        image = Image.new("L", (86, 86))
        ImageDraw.Draw(image).text((43, 43), digit, anchor="mm", font=font, fill=255)
        yield cv2.threshold(np.asarray(image), 127, 255, cv2.THRESH_BINARY)[1]

    for style in (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_TRIPLEX):
        image = np.zeros((86, 86), dtype=np.uint8)
        (width, height), _ = cv2.getTextSize(digit, style, 1.8, 4)
        cv2.putText(image, digit, ((86 - width) // 2, (86 + height) // 2), style, 1.8, 255, 4)
        yield image


def template_styles(image, digit):
    thin = cv2.erode(image, KERNEL)
    thick = cv2.dilate(image, KERNEL)
    extra = cv2.dilate(image, KERNEL, iterations=2)
    yield from (image, extra, thick - thin, extra - thin)

    x, y, width, height = cv2.boundingRect(image)
    if digit == "0":
        slashed = image.copy()
        cv2.line(slashed, (x + width // 5, y + height * 4 // 5),
                 (x + width * 4 // 5, y + height // 5), 255, 2)
        yield slashed
    elif digit == "7":
        for part in (0.46, 0.57, 0.65):
            row = y + round(height * part)
            columns = np.flatnonzero(image[row])
            if not columns.size:
                continue
            center = round(columns.mean())
            for length in (0.65, 0.85):
                half = round(length * width / 2)
                crossed = image.copy()
                cv2.line(crossed, (center - half, row), (center + half, row), 255, max(2, height // 10))
                yield crossed
                yield cv2.dilate(crossed, KERNEL, iterations=2)


def shape_matches(glyphs):
    queries = []
    for versions in glyphs:
        samples = [describe_shape(mask) for _, mask in versions]
        values = versions[0][0]
        cutoff = np.percentile(values[values > 0], 93) * 0.55
        contrast = (values >= cutoff).astype(np.uint8) * 255
        samples.append(describe_shape(contrast))
        queries.append(samples)
    shapes = np.array(queries)
    by_digit = []
    for digit in digits:
        templates = np.stack([
            describe_shape(rotate(style, angle))
            for image in template_images(digit)
            for style in template_styles(image, digit)
            for angle in range(-48, 49, 12)
        ])
        similarity = shapes @ templates.T
        by_digit.append(similarity.max(axis=(1, 2)))
    return np.stack(by_digit, axis=1)


def has_upper_loop(values):
    # A closed counter can distinguish a 9 from a 3 with a similar outline.
    upper = np.percentile(values[values > 0], 93)
    support = 0
    for ratio in (0.28, 0.35, 0.42):
        mask = (values >= upper * ratio).astype(np.uint8)
        _, top, width, height = cv2.boundingRect(mask)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None:
            continue
        holes = [
            contour
            for contour, parent in zip(contours, hierarchy[0, :, 3])
            if parent >= 0 and cv2.contourArea(contour) >= width * height * 0.009
        ]
        if len(holes) == 1:
            _, y, _, hole_height = cv2.boundingRect(holes[0])
            center = (y + hole_height / 2 - top) / height
            if 0.15 <= center <= 0.42 and hole_height <= height * 0.45:
                support += 1
    return support >= 2


def read_glyph(versions, ocr):
    scores = np.zeros(10, dtype=np.int32)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ocr.SetPageSegMode(PSM.SINGLE_CHAR)
    for values, _ in versions:
        upper = np.percentile(values[values > 0], 93)
        for ratio in (0.22, 0.38, 0.55, 0.71, 0.85):
            raw = (values >= upper * ratio).astype(np.uint8) * 255
            closed = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel)
            thinned = cv2.erode(closed, kernel)
            for mask in (raw, closed, thinned):
                padded = np.pad(mask, 25)
                best = np.zeros(10, dtype=np.int32)
                for angle in (-40, -25, -12, 0, 12, 25, 40):
                    ocr.SetImage(Image.fromarray(255 - rotate(padded, angle)))
                    value = ocr.GetUTF8Text().strip()
                    if len(value) == 1 and value in digits:
                        confidence = ocr.MeanTextConf()
                        if confidence >= 30:
                            best[int(value)] = max(best[int(value)], confidence)
                scores += best
    return scores


def recognize_glyphs(rgb, ocr, debug_dir):
    glyphs = extract_glyphs(rgb)
    if not glyphs:
        return None, ()

    result = []
    uncertain = []
    for index, (versions, shapes) in enumerate(zip(glyphs, shape_matches(glyphs))):
        if debug_dir is not None:
            Image.fromarray(255 - versions[0][1]).save(debug_dir / f"glyph-{index + 1}.png")
        second, first = np.argsort(shapes)[-2:]
        margin = shapes[first] - shapes[second]
        if {first, second} == {3, 9} and margin <= 0.025 and has_upper_loop(versions[0][0]):
            result.append("9")
            uncertain.append(index)
            continue
        if shapes[first] >= 0.76 and margin >= 0.025:
            result.append(str(first))
            if margin < 0.06:
                uncertain.append(index)
            continue

        shape_choice = first
        scores = read_glyph(versions, ocr)
        second, first = np.argsort(scores)[-2:]
        if scores[first] < 100 or scores[first] == scores[second]:
            return None, ()
        # A close OCR alternative can have a clearer match to the digit's shape.
        if (
            shapes[shape_choice] >= 0.80
            and shapes[shape_choice] - shapes[first] >= 0.01
            and scores[shape_choice] >= 0.70 * scores[first]
        ):
            result.append(str(shape_choice))
            uncertain.append(index)
            continue
        result.append(str(first))
        if scores[first] < 1.6 * scores[second] or shapes[first] < shapes.max() - 0.03:
            uncertain.append(index)
    return "".join(result), uncertain
