from __future__ import annotations

from fastapi import FastAPI

app = FastAPI(title="MBI Master Vision API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "mbi-master-vision-almoxarifado"}


@app.get("/classes")
def classes() -> dict:
    return {
        "classes": [
            "produto_vencido",
            "sem_epi",
            "acesso_nao_autorizado",
            "produto_fora_posicao",
            "caixa_aberta",
            "temperatura_anomala",
            "empilhamento_incorreto",
            "pragas",
        ]
    }
