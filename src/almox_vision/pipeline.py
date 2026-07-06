from __future__ import annotations

import asyncio

import cv2
import numpy as np

from almox_vision.alerts.dispatcher import AlertDispatcher
from almox_vision.detectors.yolov8_detector import YOLOv8Detector
from almox_vision.validators.qwen_validator import QwenVLValidator


class AlmoxarifadoPipeline:
    def __init__(self, config: dict):
        self.detector = YOLOv8Detector(
            config["yolo_model_path"],
            conf=config.get("conf_threshold", 0.50),
            iou=config.get("iou_threshold", 0.45),
            imgsz=config.get("img_size", 640),
        )
        self.validator = QwenVLValidator(config.get("qwen_model_id", "Qwen/Qwen-VL-Chat"))
        channels = config.get("alert_channels", {})
        self.alerter = AlertDispatcher(channels.get("teams_webhook"))

    async def process_frame(self, frame: np.ndarray, camera_id: str) -> list[dict]:
        alerts = []
        for det in self.detector.detect(frame):
            crop = self._crop(frame, det.bbox)
            validation = await self.validator.validate(frame, crop, det.class_name, det.confidence)
            if validation.confirmed:
                alert = await self.alerter.dispatch(det.__dict__, validation.reason, camera_id)
                alerts.append(alert)
        return alerts

    async def process_camera(self, camera_url: str, camera_id: str) -> None:
        cap = cv2.VideoCapture(camera_url)
        while True:
            ok, frame = cap.read()
            if not ok:
                await asyncio.sleep(1)
                continue
            await self.process_frame(frame, camera_id)

    @staticmethod
    def _crop(frame: np.ndarray, bbox: list[float]) -> np.ndarray:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h, w = frame.shape[:2]
        return frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
