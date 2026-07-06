from __future__ import annotations

import mlflow
from prometheus_client import Counter, Histogram

DETECTIONS = Counter("almox_detections_total", "Total de detecções", ["class_name"])
LATENCY = Histogram("almox_pipeline_latency_seconds", "Latência do pipeline")


class MLOpsMonitor:
    def __init__(self, tracking_uri: str):
        mlflow.set_tracking_uri(tracking_uri)

    def log_metrics(self, metrics: dict) -> None:
        with mlflow.start_run(nested=True):
            for key, value in metrics.items():
                mlflow.log_metric(key, float(value))
