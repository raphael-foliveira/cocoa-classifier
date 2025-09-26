import argparse
import json
import math
import csv
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from joblib import dump, load


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class SegmentParams:
    min_area: int = 600
    max_area: int = 2_000_000
    open_ksize: int = 5
    sure_bg_dilate: int = 5
    distance_thresh: float = 0.25


def segment_beans(
    img_bgr: np.ndarray, params: SegmentParams
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Return binary mask and list of contours for each bean after watershed splitting."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = th if th.mean() > 127 else cv2.bitwise_not(th)

    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (params.open_ksize, params.open_ksize)
    )
    opened = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)

    kbg = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (params.sure_bg_dilate, params.sure_bg_dilate)
    )
    sure_bg = cv2.dilate(opened, kbg, iterations=2)

    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    dist_norm = dist / dist.max() if dist.max() > 0 else dist
    sure_fg = (dist_norm > params.distance_thresh).astype(np.uint8) * 255

    unknown = cv2.subtract(sure_bg, sure_fg)

    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    cv2.watershed(img_bgr, markers)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    contours = []
    for lbl in range(2, markers.max() + 1):
        comp = (markers == lbl).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if params.min_area <= area <= params.max_area:
                contours.append(c)
                cv2.drawContours(
                    mask,
                    [c],
                    -1,
                    (255, 255, 255),
                    thickness=-1,
                )

    return mask, contours


def contour_features(img_bgr: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Extracts shape and color features for a single contour."""
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(
        mask,
        [contour],
        -1,
        (255, 255, 255),
        -1,
    )

    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect = w / (h + 1e-6)
    circularity = (4 * math.pi * area) / ((perim + 1e-6) ** 2)
    hull = cv2.convexHull(contour)
    solidity = area / (cv2.contourArea(hull) + 1e-6)

    ecc = 0.0
    if len(contour) >= 5:
        (cx, cy), (MA, ma), angle = cv2.fitEllipse(contour)
        a = max(MA, ma) / 2.0
        b = min(MA, ma) / 2.0
        if a > 1e-6:
            ecc = math.sqrt(1.0 - (b**2) / (a**2))

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    def masked_stats(ch):
        vals = ch[mask == 255]
        return (float(vals.mean()), float(vals.std())) if len(vals) > 0 else (0.0, 0.0)

    Lm, Ls = masked_stats(lab[:, :, 0])
    Am, As = masked_stats(lab[:, :, 1])
    Bm, Bs = masked_stats(lab[:, :, 2])

    features = [area, perim, aspect, circularity, solidity, ecc, Lm, Ls, Am, As, Bm, Bs]
    return np.array(features, dtype=np.float32)


def load_training_samples(data_dir: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    X, y = [], []
    classes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if not classes:
        raise RuntimeError(f"No class folders found in {data_dir}")

    for idx, cls in enumerate(classes):
        for img_path in sorted((data_dir / cls).glob("*.*")):
            img = cv2.imdecode(
                np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if img is None:
                continue

            params = SegmentParams(min_area=300, open_ksize=3)
            _, cnts = segment_beans(img, params)

            if not cnts:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (5, 5), 0)
                _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cnts, _ = cv2.findContours(
                    th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

            if cnts:
                cnt = max(cnts, key=cv2.contourArea)
                features = contour_features(img, cnt)
                X.append(features)
                y.append(idx)

    if not X:
        raise RuntimeError("No training samples extracted. Check images.")
    return np.vstack(X), np.array(y), classes


def train(data_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    X, y, classes = load_training_samples(data_dir)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, C=10.0, gamma="scale")),
        ]
    )

    class_counts = np.unique(y, return_counts=True)[1]
    n_splits = min(5, max(2, int(np.min(class_counts))))
    if n_splits > 1:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
        print(f"CV accuracy: mean={scores.mean():.3f} ± {scores.std():.3f}")

    pipe.fit(X, y)

    dump(pipe, out_dir / "model.pkl")
    with open(out_dir / "classes.json", "w") as f:
        json.dump(classes, f, indent=2)


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

    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image {image_path}")

    params = SegmentParams(min_area=min_area, max_area=max_area, open_ksize=open_ksize)
    mask, contours = segment_beans(img, params)

    results = []
    overlay = img.copy()
    for i, cnt in enumerate(contours):
        features = contour_features(img, cnt)
        proba = model.predict_proba([features])[0]
        yhat = int(np.argmax(proba))
        label = classes[yhat]
        conf = float(proba[yhat])

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{label} {conf:.2f}"
        cv2.putText(
            overlay,
            text,
            (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            text,
            (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        results.append(
            {
                "idx": i,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "pred_class": label,
                "confidence": conf,
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


def main():
    ap = argparse.ArgumentParser(description="Cocoa Beans Classifier (OpenCV + Python)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_train = sub.add_parser("train", help="Train an SVM on single-bean images")
    ap_train.add_argument(
        "--data-dir", required=True, type=Path, help="Folder with class subfolders"
    )
    ap_train.add_argument(
        "--out-dir", required=True, type=Path, help="Where to save the model"
    )

    ap_pred = sub.add_parser(
        "predict", help="Predict classes for beans in a multi-bean image"
    )
    ap_pred.add_argument("--image", required=True, type=str, help="Input image path")
    ap_pred.add_argument(
        "--model-dir", required=True, type=Path, help="Folder with model assets"
    )
    ap_pred.add_argument("--out-dir", required=True, type=Path, help="Output folder")
    ap_pred.add_argument("--min-area", type=int, default=600, help="Min contour area")
    ap_pred.add_argument(
        "--max-area", type=int, default=2_000_000, help="Max contour area"
    )
    ap_pred.add_argument(
        "--open-ksize", type=int, default=5, help="Morphology opening kernel size"
    )

    args = ap.parse_args()
    if args.cmd == "train":
        train(args.data_dir, args.out_dir)
    elif args.cmd == "predict":
        predict(
            args.image,
            args.model_dir,
            args.out_dir,
            args.min_area,
            args.max_area,
            args.open_ksize,
        )


if __name__ == "__main__":
    main()
