from __future__ import annotations

import cv2
import numpy as np
from typing import Any


def correct_plate_perspective(image: Any) -> np.ndarray:
    """Attempt perspective correction for tilted plate regions."""
    if image is None:
        return np.zeros((64, 128), dtype=np.uint8)
    img = np.array(image)
    if img.size == 0:
        return np.zeros((64, 128), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    best_quad = None
    best_area = 0
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            area = cv2.contourArea(approx)
            if area > best_area:
                best_area = area
                best_quad = approx
    if best_quad is None or best_area < 500:
        return img
    pts = best_quad.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxWidth = int(max(widthA, widthB))
    maxHeight = int(max(heightA, heightB))
    if maxWidth < 10 or maxHeight < 10:
        return img
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1],
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warped


def generate_plate_variants(image: Any) -> list[Any]:
    """Create multiple preprocessing variants for OCR from a plate crop."""
    if image is None:
        return [np.zeros((64, 128), dtype=np.uint8)]
    img = np.array(image)
    if img.size == 0:
        return [np.zeros((64, 128), dtype=np.uint8)]
    if len(img.shape) == 3:
        base = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        base = img
    variants = [base]
    upscaled = cv2.resize(base, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    variants.append(upscaled)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    variants.append(clahe.apply(upscaled))
    variants.append(cv2.bilateralFilter(upscaled, 9, 75, 75))
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    variants.append(cv2.filter2D(clahe.apply(upscaled), -1, kernel))
    variants.append(cv2.adaptiveThreshold(upscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 7))
    _, otsu = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(otsu)
    return variants


def preprocess_plate_image(image: Any) -> np.ndarray:
    """Preprocess a cropped plate image for OCR."""
    if image is None:
        return np.zeros((64, 128), dtype=np.uint8)
    img = np.array(image)
    if img.size == 0:
        return np.zeros((64, 128), dtype=np.uint8)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    gray = cv2.filter2D(gray, -1, kernel)
    gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 7)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return gray
