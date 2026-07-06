from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ValidationResult:
    confirmed: bool
    reason: str


class QwenVLValidator:
    """Validador semântico de segundo estágio.

    Este arquivo deixa uma interface pronta para produção. Em ambiente com GPU,
    carregue o Qwen-VL pelo Transformers e implemente a chamada real em validate().
    """

    PROMPTS = {
        "produto_vencido": "Confirme se há produto vencido ou deteriorado na imagem.",
        "sem_epi": "Confirme se há funcionário sem EPI obrigatório.",
        "acesso_nao_autorizado": "Confirme se há pessoa sem identificação em área restrita.",
        "produto_fora_posicao": "Confirme se há produto fora da posição correta.",
        "caixa_aberta": "Confirme se há embalagem aberta ou violada.",
        "temperatura_anomala": "Confirme se há indicação visual de temperatura fora da faixa.",
        "empilhamento_incorreto": "Confirme se há empilhamento inseguro.",
        "pragas": "Confirme se há evidência de pragas.",
    }

    def __init__(self, model_id: str = "Qwen/Qwen-VL-Chat"):
        self.model_id = model_id

    async def validate(
        self,
        full_frame: np.ndarray,
        crop: np.ndarray,
        class_name: str,
        confidence: float,
    ) -> ValidationResult:
        prompt = self.PROMPTS.get(class_name, "Confirme a detecção visual.")
        # Placeholder seguro para desenvolvimento/testes.
        confirmed = confidence >= 0.70
        reason = f"Validação automática inicial: {prompt} confiança={confidence:.2f}."
        return ValidationResult(confirmed=confirmed, reason=reason)
