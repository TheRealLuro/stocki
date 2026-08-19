# model

FastAPI service that serves the exported ONNX model through ONNX Runtime.

```bash
pip install -r requirements.txt
uvicorn main:app --reload            # http://127.0.0.1:8000/docs
```

The ONNX file defaults to `../model_training/artifacts/model.onnx`; override with the
`MODEL_PATH` environment variable. It is loaded once at startup — a load failure does not
crash the service, so `/health` can report it.

## Input layout

A *sequence* is one contiguous segment of stock data, **one row per timestep**:
`[[f0, f1, ...], [f0, f1, ...], ...]` with shape `(timesteps, features)`. The service
transposes to the channels-first layout the model expects. The number of features must
match the model's; a mismatch is a `422` naming the expected count.

## Endpoints

| method | path | body | returns |
| --- | --- | --- | --- |
| GET | `/health` | — | `200` + `model_loaded: true` when the model loaded, `503` + `false` otherwise |
| POST | `/predict` | `{"sequence": [[...], ...]}` | `{"prediction": [...]}` |
| POST | `/predict-many` | `{"sequences": [[[...], ...], ...]}` | `{"predictions": [[...], ...], "count": N}` |

```bash
curl localhost:8000/health
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"sequence": [[0,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,1]]}'
```

Status codes: `422` for malformed or wrong-shaped input, `503` when the model is not
loaded.
