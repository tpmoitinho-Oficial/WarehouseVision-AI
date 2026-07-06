# Guia de Instalação

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src
uvicorn src.almox_vision.api.main:app --reload
```
