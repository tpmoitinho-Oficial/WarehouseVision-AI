# Arquitetura Técnica

O projeto usa arquitetura em camadas:

1. **Aquisição:** câmeras RTSP.
2. **Detecção:** YOLOv8 com threshold configurável.
3. **Validação:** Qwen-VL como filtro semântico.
4. **Alertas:** Teams, e-mail e API.
5. **MLOps:** MLflow, Evidently AI, Prometheus e Grafana.
6. **Retreinamento:** coleta de falsos positivos/falsos negativos e novo fine-tuning.
