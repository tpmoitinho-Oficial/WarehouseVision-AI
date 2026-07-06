from __future__ import annotations

import argparse
from datetime import datetime

import mlflow
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/dataset.yaml")
    parser.add_argument("--model", default="yolov8m.pt")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    mlflow.set_experiment("yolov8_almoxarifado")
    with mlflow.start_run(run_name=f"train_{datetime.now():%Y%m%d_%H%M%S}"):
        model = YOLO(args.model)
        results = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)
        mlflow.log_params(vars(args))
        mlflow.log_artifact(args.data)
        print(results)


if __name__ == "__main__":
    main()
