from pathlib import Path

from cv2.typing import MatLike
from .helpers import get_blurred_gray
import cv2
from .segment_params import SegmentParams
from .bean_segmenter import segment_beans
from .feature_contourer import contour_features
import numpy as np


def load_training_samples(
    data_dir: Path,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
]:
    feature_vectors, class_labels = [], []
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError(f"No class folders found in {data_dir}")

    for idx, cls in enumerate(classes):
        for img_path in sorted((data_dir / cls).glob("*.*")):
            image = cv2.imdecode(
                np.fromfile(str(img_path), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                continue

            params = SegmentParams(min_area=300, open_ksize=3)
            _, contours = segment_beans(image, params)

            if not contours:
                threshold = _find_threshold(image)
                contours = _find_contours(threshold)

            if contours:
                contour = max(contours, key=cv2.contourArea)
                features = contour_features(image, contour)
                feature_vectors.append(features)
                class_labels.append(idx)

    if not feature_vectors:
        raise RuntimeError("No training samples extracted. Check images.")
    return np.vstack(feature_vectors), np.array(class_labels), classes


def _find_threshold(image: np.ndarray) -> np.ndarray:
    blur = get_blurred_gray(image)
    _, threshold = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return threshold


def _find_contours(image: np.ndarray) -> MatLike:
    _, contours = cv2.findContours(
        image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    return contours
