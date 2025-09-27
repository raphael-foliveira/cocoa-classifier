import argparse
import csv
import numpy as np
import cv2
import json
from joblib import load
from pathlib import Path
from cocoa_classifier.trainer import train
from cocoa_classifier.predictor import predict


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
        model_dir = args.model_dir
        image_path = args.image
        model = load(model_dir / "model.pkl")
        with open(model_dir / "classes.json", "r") as f:
            classes = json.load(f)
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not read image {image_path}")

        overlay, results = predict(
            image,
            model,
            classes,
            args.min_area,
            args.max_area,
            args.open_ksize,
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


if __name__ == "__main__":
    main()
