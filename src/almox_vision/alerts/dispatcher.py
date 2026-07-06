from __future__ import annotations

from datetime import datetime, timezone

import httpx


class AlertDispatcher:
    SEVERITY = {
        "produto_vencido": "CRITICA",
        "pragas": "CRITICA",
        "temperatura_anomala": "CRITICA",
        "sem_epi": "ALTA",
        "acesso_nao_autorizado": "ALTA",
        "caixa_aberta": "ALTA",
        "empilhamento_incorreto": "ALTA",
        "produto_fora_posicao": "MEDIA",
    }

    def __init__(self, teams_webhook: str | None = None):
        self.teams_webhook = teams_webhook

    async def dispatch(self, detection: dict, reason: str, camera_id: str) -> dict:
        alert = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "camera_id": camera_id,
            "class": detection["class_name"],
            "confidence": detection["confidence"],
            "severity": self.SEVERITY.get(detection["class_name"], "MEDIA"),
            "reason": reason,
            "bbox": detection["bbox"],
        }
        if self.teams_webhook:
            await self._send_teams(alert)
        return alert

    async def _send_teams(self, alert: dict) -> None:
        async with httpx.AsyncClient(timeout=10) as client:
            await client.post(self.teams_webhook, json={"text": str(alert)})
