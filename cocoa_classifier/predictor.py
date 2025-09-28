from typing import Any, TypedDict
import cv2
from cv2.typing import MatLike
import numpy as np
from .segment_params import SegmentParams
from .bean_segmenter import segment_single_bean, segment_beans
from .feature_contourer import contour_features


class PredictionResultRow(TypedDict):
    idx: int
    x: int
    y: int
    w: int
    h: int
    pred_class: str
    confidence: float


def predict(
    image: MatLike,
    model: Any,
    classes: list[str],
    min_area: int,
    max_area: int,
    open_ksize: int,
    single_bean: bool = False,
):
    params = SegmentParams(min_area=min_area, max_area=max_area, open_ksize=open_ksize)
    contours = _get_contours(image, params, single_bean)

    results: list[PredictionResultRow] = []
    overlay = image.copy()
    for i, cnt in enumerate(contours):
        features = contour_features(image, cnt)
        probability = model.predict_proba([features])[0]
        yhat = int(np.argmax(probability))
        label = classes[yhat]
        confidence = float(probability[yhat])

        x, y, width, height = cv2.boundingRect(cnt)
        cv2.rectangle(
            overlay,
            (x, y),
            (x + width, y + height),
            (0, 255, 0),
            2,
        )
        text = f"{label} {confidence:.2f}"
        cv2.putText(
            overlay,
            text,
            (x, y + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

        results.append(
            {
                "idx": i,
                "x": x,
                "y": y,
                "w": width,
                "h": height,
                "pred_class": label,
                "confidence": confidence,
            }
        )

    return overlay, results


def _get_contours(
    image: MatLike, params: SegmentParams, single_bean: bool
) -> list[np.ndarray]:
    if single_bean:
        return segment_single_bean(image, params)
    return segment_beans(image, params)
