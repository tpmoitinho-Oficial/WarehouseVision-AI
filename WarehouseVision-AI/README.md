# MBI Master Vision — Pipeline Inteligente de Visão Computacional para Almoxarifados

Repositório profissional baseado na dissertação **Pipeline Inteligente de Visão Computacional para Gestão de Almoxarifados e Centros de Distribuição de Alimentos**.

## Objetivo
Sistema de visão computacional em dois estágios:

1. **YOLOv8** para detecção em tempo real.
2. **Qwen-VL** para validação semântica e redução de falsos positivos.

Inclui MLOps com MLflow, Evidently AI, Prometheus, Grafana, FastAPI, Docker e estrutura pronta para fine-tuning e produção.

## Classes do projeto

| ID | Classe | Criticidade |
|---|---|---|
| C01 | produto_vencido | Crítica |
| C02 | sem_epi | Alta |
| C03 | acesso_nao_autorizado | Alta |
| C04 | produto_fora_posicao | Média |
| C05 | caixa_aberta | Alta |
| C06 | temperatura_anomala | Crítica |
| C07 | empilhamento_incorreto | Alta |
| C08 | pragas | Crítica |

## Arquitetura

```text
Câmeras RTSP -> YOLOv8 -> Qwen-VL -> Alertas -> Dashboard -> MLOps
                         |                     |
                         v                     v
                     PostgreSQL            MLflow/Evidently
```

## Como instalar

```bash
git clone https://github.com/SEU-USUARIO/mbi-master-vision-almoxarifado.git
cd mbi-master-vision-almoxarifado
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

## Executar API

```bash
uvicorn src.almox_vision.api.main:app --host 0.0.0.0 --port 8000 --reload
```

## Treinar YOLOv8

```bash
python scripts/train_yolo.py --data configs/dataset.yaml --model yolov8m.pt --epochs 150
```

## Rodar pipeline

```bash
python scripts/run_pipeline.py --config configs/config.yaml
```

## Docker

```bash
docker compose up -d --build
```

## Estrutura

```text
src/almox_vision/     Código principal
detectors/            YOLOv8
validators/           Qwen-VL
alerts/               Alertas Teams/E-mail
mlops/                MLflow, drift, retreinamento
api/                  FastAPI
docs/                 Documentação técnica
configs/              Configurações YAML
monitoring/           Prometheus/Grafana
tests/                Testes automatizados
```

## Resultados esperados da pesquisa

| Métrica | Só YOLO | YOLO + Qwen-VL |
|---|---:|---:|
| Precisão | 71,3% | 93,4% |
| Falsos Positivos | 28,7% | 6,6% |
| F1-Score | 77,6% | 87,2% |
| mAP@0.5 | 68,4% | 82,1% |

## Autor

Thiago Passaline Moitinho — MBI Computer Vision Master
