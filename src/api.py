from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import load_bundle, predict_from_mapping

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

_bundle: dict[str, Any] | None = None


class PredictRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    aod_550: float = Field(..., ge=0, le=5)
    t2m_celsius: float = Field(..., ge=-50, le=65)
    wind_speed_10m: float = Field(..., ge=0, le=100)
    r2m: float = Field(..., ge=0, le=100)
    blh: float = Field(..., ge=0, le=10000)
    hour: int = Field(..., ge=0, le=23)
    month: int = Field(..., ge=1, le=12)
    season: int = Field(..., ge=0, le=4)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _bundle
    _bundle = load_bundle()
    logger.info("Loaded model bundle during API startup")
    yield


app = FastAPI(title="VayuDrishti API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_loaded": _bundle is not None,
        "bundle_version": _bundle.get("bundle_version") if _bundle else None,
    }


@app.post("/predict")
def predict(payload: PredictRequest) -> dict[str, Any]:
    if _bundle is None:
        raise HTTPException(status_code=503, detail="Model bundle is not loaded")
    try:
        prediction = predict_from_mapping(payload.model_dump(), _bundle)
        logger.info("Prediction completed for lat=%s lon=%s", payload.latitude, payload.longitude)
        return {"prediction": {"pm2_5": prediction}, "model_version": _bundle.get("model_version")}
    except ValueError as exc:
        logger.exception("Prediction validation failed")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
