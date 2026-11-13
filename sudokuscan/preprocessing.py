from __future__ import annotations

import cv2
import numpy as np


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Flatten slow illumination changes while retaining printed strokes."""
    h, w = gray.shape[:2]
    sigma = max(8.0, 0.025 * min(h, w))
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    normalized = gray.astype(np.float32) / np.maximum(background.astype(np.float32), 1.0)
    normalized *= float(np.median(background))
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(normalized)


def preprocess_image(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return original grayscale, normalized grayscale and adaptive binary image."""
    if image is None or image.size == 0:
        raise ValueError("Input image is empty")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    normalized = normalize_illumination(denoised)

    h, w = gray.shape[:2]
    block = max(15, int(round(min(h, w) * 0.025)))
    if block % 2 == 0:
        block += 1
    block = min(block, 151)
    binary = cv2.adaptiveThreshold(
        normalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, 7,
    )
    return gray, normalized, binary


def resize_for_detection(image: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale < 1.0:
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    else:
        resized = image.copy()
    return resized, scale
