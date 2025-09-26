import argparse
import json
import math
import csv
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from joblib import dump, load


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# -------------------------
# Image preprocessing
# -------------------------
def clahe_lab(img_bgr: np.ndarray) -> np.ndarray:
    """Apply CLAHE to L channel in Lab for gentle contrast normalization."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    Lc = clahe.apply(L)
    labc = cv2.merge([Lc, A, B])
    return cv2.cvtColor(labc, cv2.COLOR_LAB2BGR)


# -------------------------
# LBP texture (basic, 8-neighbor, radius=1, 256-bin hist)
# -------------------------
def lbp_8u(gray: np.ndarray) -> np.ndarray:
    """Compute LBP image with 8 neighbors, radius 1, basic pattern, returns uint8 codes [0..255]."""
    h, w = gray.shape
    padded = cv2.copyMakeBorder(gray, 1, 1, 1, 1, cv2.BORDER_REFLECT)
    center = padded[1 : h + 1, 1 : w + 1]
    codes = np.zeros_like(center, dtype=np.uint8)
    offsets = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
        (1, 0),
        (1, -1),
        (0, -1),
    ]  # clockwise
    for i, (dy, dx) in enumerate(offsets):
        nbr = padded[1 + dy : h + 1 + dy, 1 + dx : w + 1 + dx]
        codes |= (nbr >= center).astype(np.uint8) << (7 - i)
    return codes


def lbp_hist(gray: np.ndarray, mask: np.ndarray, bins: int = 256) -> np.ndarray:
    lbp = lbp_8u(gray)
    hist = cv2.calcHist([lbp], [0], mask, [bins], [0, 256]).flatten()
    if hist.sum() > 0:
        hist = hist / hist.sum()
    return hist


# -------------------------
# Segmentation
# -------------------------
@dataclass
class SegmentParams:
    min_area: int = 600  # ignore tiny specks
    max_area: int = 2_000_000  # ignore unrealistically large blobs
    open_ksize: int = 5  # morphology opening kernel size
    sure_bg_dilate: int = 5  # dilation size for sure background
    distance_thresh: float = 0.25  # watershed split sensitivity (relative to max dist)


def segment_beans(
    img_bgr: np.ndarray, params: SegmentParams
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Return binary mask and list of contours for each bean after watershed splitting."""
    img = clahe_lab(img_bgr)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive/Otsu threshold
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Beans are darker or lighter depending on background; pick the mode with fewer foreground pixels
    if th.mean() > 127:
        fg = th
    else:
        fg = cv2.bitwise_not(th)

    # Morphological opening to remove noise
    k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (params.open_ksize, params.open_ksize)
    )
    opened = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)

    # Sure background
    kbg = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (params.sure_bg_dilate, params.sure_bg_dilate)
    )
    sure_bg = cv2.dilate(opened, kbg, iterations=2)

    # Distance transform for sure foreground
    dist = cv2.distanceTransform(opened, cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist_norm = dist / dist.max()
    else:
        dist_norm = dist
    sure_fg = (dist_norm > params.distance_thresh).astype(np.uint8) * 255

    # Unknown region
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # make sure background is 1
    markers[unknown == 255] = 0

    # Watershed
    img_ws = img_bgr.copy()
    cv2.watershed(img_ws, markers)

    # Build mask per label (labels >=2 are objects)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    contours = []
    for lbl in range(2, markers.max() + 1):
        comp = (markers == lbl).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < params.min_area or area > params.max_area:
                continue
            contours.append(c)
            cv2.drawContours(mask, [c], -1, (255, 255, 255), thickness=-1)

    return mask, contours


# -------------------------
# Feature extraction per contour
# -------------------------
def contour_features(
    img_bgr: np.ndarray, contour: np.ndarray
) -> Tuple[np.ndarray, Dict[str, float]]:
    mask = np.zeros(img_bgr.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    area = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    x, y, w, h = cv2.boundingRect(contour)
    aspect = w / (h + 1e-6)
    circularity = (4 * math.pi * area) / ((perim + 1e-6) ** 2)
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull) + 1e-6
    solidity = area / hull_area

    # Ellipse eccentricity (if enough points)
    ecc = 0.0
    if len(contour) >= 5:
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (MA, ma), angle = (
            ellipse  # major/minor axes lengths (OpenCV returns (major, minor) order is arbitrary)
        )
        a = max(MA, ma) / 2.0
        b = min(MA, ma) / 2.0
        if a > 1e-6:
            ecc = math.sqrt(1.0 - (b * b) / (a * a))

    # Color stats in Lab
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    def masked_stats(ch):
        vals = ch[mask == 255].astype(np.float32)
        if len(vals) == 0:
            return 0.0, 0.0
        return float(vals.mean()), float(vals.std())

    Lm, Ls = masked_stats(L)
    Am, As = masked_stats(A)
    Bm, Bs = masked_stats(B)

    # Texture LBP histogram on grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hist = lbp_hist(gray, mask, bins=256)

    features = [area, perim, aspect, circularity, solidity, ecc, Lm, Ls, Am, As, Bm, Bs]
    feat_names = [
        "area",
        "perimeter",
        "aspect",
        "circularity",
        "solidity",
        "eccentricity",
        "L_mean",
        "L_std",
        "a_mean",
        "a_std",
        "b_mean",
        "b_std",
    ]
    # append texture hist
    for i in range(256):
        features.append(hist[i])
        feat_names.append(f"lbp_{i}")

    feat_arr = np.array(features, dtype=np.float32)
    meta = {name: float(val) for name, val in zip(feat_names, feat_arr)}
    return feat_arr, meta


# -------------------------
# Loading training data
# -------------------------
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
            # segment largest contour (assumes single-bean image)
            params = SegmentParams(min_area=300, max_area=1_000_000, open_ksize=3)
            mask, cnts = segment_beans(img, params)
            if not cnts:
                # fallback: try simple largest contour from edges
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, th = cv2.threshold(
                    cv2.GaussianBlur(gray, (5, 5), 0),
                    0,
                    255,
                    cv2.THRESH_BINARY + cv2.THRESH_OTSU,
                )
                cnts, _ = cv2.findContours(
                    th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
            if not cnts:
                continue
            cnt = max(cnts, key=cv2.contourArea)
            f, _ = contour_features(img, cnt)
            X.append(f)
            y.append(idx)
    if not X:
        raise RuntimeError("No training samples could be extracted. Check your images.")
    return np.vstack(X), np.array(y), classes


# -------------------------
# Train
# -------------------------
def train(data_dir: Path, out_dir: Path):
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    X, y, classes = load_training_samples(data_dir)

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svm", SVC(kernel="rbf", probability=True, C=10.0, gamma="scale")),
        ]
    )
    cv = StratifiedKFold(
        n_splits=min(5, np.unique(y, return_counts=True)[1].min()),
        shuffle=True,
        random_state=42,
    )
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print(f"CV accuracy: mean={scores.mean():.3f} ± {scores.std():.3f}")

    pipe.fit(X, y)

    dump(pipe, out_dir / "model.pkl")
    with open(out_dir / "classes.json", "w") as f:
        json.dump(classes, f, indent=2)
    feat_names = [
        "area",
        "perimeter",
        "aspect",
        "circularity",
        "solidity",
        "eccentricity",
        "L_mean",
        "L_std",
        "a_mean",
        "a_std",
        "b_mean",
        "b_std",
    ] + [f"lbp_{i}" for i in range(256)]
    with open(out_dir / "feat_meta.json", "w") as f:
        json.dump({"feature_names": feat_names, "version": 1}, f, indent=2)


# -------------------------
# Predict on multi-bean image
# -------------------------
def predict(
    image_path: str,
    model_dir: Path,
    out_dir: Path,
    min_area: int,
    max_area: int,
    open_ksize: int,
):
    model = load(Path(model_dir) / "model.pkl")
    classes = json.load(open(Path(model_dir) / "classes.json", "r"))
    out_dir = Path(out_dir)
    ensure_dir(out_dir)

    img = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image {image_path}")

    params = SegmentParams(min_area=min_area, max_area=max_area, open_ksize=open_ksize)
    mask, cnts = segment_beans(img, params)

    results = []
    overlay = img.copy()
    for i, cnt in enumerate(cnts):
        f, meta = contour_features(img, cnt)
        proba = model.predict_proba([f])[0]
        yhat = int(np.argmax(proba))
        label = classes[yhat]
        conf = float(proba[yhat])

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            overlay,
            f"{label} {conf:.2f}",
            (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            f"{label} {conf:.2f}",
            (x, y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

        row = {
            "idx": i,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "pred_class": label,
            "confidence": conf,
        }
        row.update(meta)
        # also include per-class probabilities
        for ci, cname in enumerate(classes):
            row[f"proba_{cname}"] = float(proba[ci])
        results.append(row)

    # Save annotated image
    out_img = out_dir / "annotated.png"
    cv2.imwrite(str(out_img), overlay)

    # Save CSV
    out_csv = out_dir / "predictions.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(results[0].keys())
            if results
            else ["idx", "x", "y", "w", "h", "pred_class", "confidence"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Wrote {out_img}")
    print(f"Wrote {out_csv}")


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Cocoa Beans Classifier (OpenCV + Python)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_train = sub.add_parser(
        "train", help="Train an SVM on single-bean images in class folders"
    )
    ap_train.add_argument(
        "--data-dir", required=True, type=str, help="Folder with class subfolders"
    )
    ap_train.add_argument(
        "--out-dir", required=True, type=str, help="Where to save the model"
    )

    ap_pred = sub.add_parser(
        "predict", help="Predict classes for beans in a multi-bean image"
    )
    ap_pred.add_argument("--image", required=True, type=str, help="Input image path")
    ap_pred.add_argument(
        "--model-dir",
        required=True,
        type=str,
        help="Folder with model.pkl and classes.json",
    )
    ap_pred.add_argument(
        "--out-dir",
        required=True,
        type=str,
        help="Output folder for annotated image + CSV",
    )
    ap_pred.add_argument(
        "--min-area", type=int, default=600, help="Min contour area to keep"
    )
    ap_pred.add_argument(
        "--max-area", type=int, default=2_000_000, help="Max contour area to keep"
    )
    ap_pred.add_argument(
        "--open-ksize", type=int, default=5, help="Morphology opening kernel size"
    )

    args = ap.parse_args()
    if args.cmd == "train":
        train(
            data_dir=Path(args.data_dir),
            out_dir=Path(args.out_dir),
        )
    elif args.cmd == "predict":
        predict(
            image_path=args.image,
            model_dir=Path(args.model_dir),
            out_dir=Path(args.out_dir),
            min_area=args.min_area,
            max_area=args.max_area,
            open_ksize=args.open_ksize,
        )


if __name__ == "__main__":
    main()
