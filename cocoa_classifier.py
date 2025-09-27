import argparse
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
