from pathlib import Path
import csv
import json
import cv2
import numpy as np
from joblib import load
from .segment_params import SegmentParams
from .bean_segmenter import segment_beans
from .feature_contourer import contour_features


def predict(
    image_path: str,
    model_dir: Path,
    out_dir: Path,
    min_area: int,
    max_area: int,
    open_ksize: int,
):
    model = load(model_dir / "model.pkl")
    with open(model_dir / "classes.json", "r") as f:
        classes = json.load(f)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image {image_path}")

    params = SegmentParams(min_area=min_area, max_area=max_area, open_ksize=open_ksize)
    _, contours = segment_beans(image, params)

    results = []
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

    out_img = out_dir / f"{Path(image_path).stem}_annotated.png"
    cv2.imwrite(str(out_img), overlay)

    out_csv = out_dir / "predictions.csv"
    if results:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"Wrote {out_img}")
    print(f"Wrote {out_csv}")
