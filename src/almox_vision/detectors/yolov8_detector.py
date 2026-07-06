from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: list[float]


class YOLOv8Detector:
    """Detector de objetos YOLOv8 para o domínio de almoxarifado alimentar."""

    CLASS_NAMES = {
        0: "produto_vencido",
        1: "sem_epi",
        2: "acesso_nao_autorizado",
        3: "produto_fora_posicao",
        4: "caixa_aberta",
        5: "temperatura_anomala",
        6: "empilhamento_incorreto",
        7: "pragas",
    }

    def __init__(self, model_path: str, conf: float = 0.50, iou: float = 0.45, imgsz: int = 640):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz

    def detect(self, frame: np.ndarray) -> list[Detection]:
        results: list[Any] = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                detections.append(
                    Detection(
                        class_id=class_id,
                        class_name=self.CLASS_NAMES.get(class_id, str(class_id)),
                        confidence=float(box.conf[0]),
                        bbox=[float(x) for x in box.xyxy[0].tolist()],
                    )
                )
        return detections
