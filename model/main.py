"""Prediction API for the exported stock CNN.

    uvicorn main:app --reload            # from the model/ directory
    MODEL_PATH=/path/to/model.onnx uvicorn main:app

Endpoints
    GET  /health        -> whether the ONNX model loaded successfully
    POST /predict       -> one segment  -> one prediction
    POST /predict-many  -> N segments   -> N predictions
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from inference import OnnxPredictor

# Path to the ONNX file produced by model_training/export_onnx.py.
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "model_training" / "artifacts" / "model.onnx"
MODEL_PATH = Path(os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))

# Populated at startup; stays None when loading fails so /health can report it.
predictor: OnnxPredictor | None = None
load_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor, load_error
    try:
        predictor = OnnxPredictor(MODEL_PATH)
        load_error = None
        print(f"loaded model: {predictor.info()}")
    except Exception as exc:  # keep serving so /health can report the failure
        predictor, load_error = None, f"{type(exc).__name__}: {exc}"
        print(f"failed to load model from {MODEL_PATH} -- {load_error}")
    yield
    predictor = None


app = FastAPI(
    title="stocki prediction API",
    description="Runs a 1D-CNN stock-price model served through ONNX Runtime.",
    version="0.1.0",
    lifespan=lifespan,
)


# --- request / response schemas --------------------------------------
# A "sequence" is a contiguous segment of stock data: one row per timestep,
# one column per input feature -- shape (timesteps, features).
Sequence2D = list[list[float]]


class PredictRequest(BaseModel):
    sequence: Sequence2D = Field(
        ...,
        min_length=1,
        description="Contiguous segment, shape (timesteps, features).",
    )


class PredictManyRequest(BaseModel):
    sequences: list[Sequence2D] = Field(
        ...,
        min_length=1,
        description="One or more segments, each shape (timesteps, features).",
    )


class PredictResponse(BaseModel):
    prediction: list[float]


class PredictManyResponse(BaseModel):
    predictions: list[list[float]]
    count: int


class HealthResponse(BaseModel):
    # the model_* field names below would otherwise collide with pydantic's
    # reserved "model_" namespace
    model_config = ConfigDict(protected_namespaces=())

    model_loaded: bool
    model_path: str
    detail: dict | None = None
    error: str | None = None


def require_predictor() -> OnnxPredictor:
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail=f"model not loaded from {MODEL_PATH}" + (f" -- {load_error}" if load_error else ""),
        )
    return predictor


# --- endpoints -------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> JSONResponse:
    """True when the API successfully loaded the model, false (503) otherwise."""
    loaded = predictor is not None
    body = HealthResponse(
        model_loaded=loaded,
        model_path=str(MODEL_PATH),
        detail=predictor.info() if loaded else None,
        error=load_error,
    )
    return JSONResponse(status_code=200 if loaded else 503, content=body.model_dump())


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    """Run a single segment through the model."""
    model = require_predictor()
    try:
        return PredictResponse(prediction=model.predict(request.sequence))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/predict-many", response_model=PredictManyResponse)
def predict_many(request: PredictManyRequest) -> PredictManyResponse:
    """Run several segments through the model in one call."""
    model = require_predictor()
    try:
        predictions = model.predict_many(request.sequences)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return PredictManyResponse(predictions=predictions, count=len(predictions))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 8000)))
