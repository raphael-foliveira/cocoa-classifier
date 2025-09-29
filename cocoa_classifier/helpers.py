import cv2
from pathlib import Path
import numpy as np


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def convert_to_bgr(img_lab: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_lab, cv2.COLOR_LAB2BGR)


def convert_to_gray(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


def convert_to_lab(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
