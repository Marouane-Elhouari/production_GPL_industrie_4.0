import logging
import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.schemas import (BatchRequest, BatchResponse, ModelInfo,
                         PredictRequest, PredictResponse)

logger = logging.getLogger("duty-api")
logging.basicConfig(level=logging.INFO)

state = {}


def env_bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes")


@asynccontextmanager
async def lifespan(app):
    path = os.getenv("MODEL_PATH", "models/duty_model.joblib")
    if not os.path.exists(path):
        raise RuntimeError(f"Modele introuvable: {path}")
    state["artifact"] = joblib.load(path)
    art = state["artifact"]
    logger.info("Modele charge: %s (%s) | %d features", art["model_name"], art["version"], len(art["features"]))
    yield
    state.clear()


app = FastAPI(
    title="API d'estimation du duty de HT-1 (capteur virtuel)",
    description="Estime le duty de l'echangeur HT-1 a partir des mesures du procede (chaine GPL, simulation DWSIM).",
    version="1.1.0",
    lifespan=lifespan,
)


def artifact():
    art = state.get("artifact")
    if art is None:
        raise HTTPException(status_code=503, detail="Modele non charge")
    return art


def check_row(row, art):
    errors, warnings = [], []
    strict = env_bool("STRICT_RANGES", True)
    margin = float(os.getenv("RANGE_MARGIN", "0.05"))

    unknown = sorted(set(row) - set(art["features"]))
    if unknown:
        warnings.append(f"features ignorees (inconnues du modele): {unknown}")
    missing = [f for f in art["features"] if f not in row]
    if missing:
        errors.append(f"features manquantes: {missing}")

    for feat in art["features"]:
        if feat not in row:
            continue
        value = row[feat]
        lo, hi = art["ranges"][feat]
        tol = margin * (hi - lo)
        if not (lo - tol <= value <= hi + tol):
            msg = f"'{feat}'={value} hors du domaine d'entrainement [{lo:.4g}, {hi:.4g}]"
            (errors if strict else warnings).append(msg)
    return errors, warnings


def to_frame(rows, art):
    return pd.DataFrame([{f: r[f] for f in art["features"]} for r in rows], columns=art["features"])


@app.get("/health")
def health():
    art = artifact()
    return {"status": "ok", "model_version": art["version"]}


@app.get("/model-info", response_model=ModelInfo)
def model_info():
    art = artifact()
    return ModelInfo(model_name=art["model_name"], model_version=art["version"],
                     target=art["target"], features=art["features"],
                     ranges=art["ranges"], example=art["example"], metrics=art["metrics"])


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    art = artifact()
    errors, warnings = check_row(req.features, art)
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    duty = float(art["model"].predict(to_frame([req.features], art))[0])
    return PredictResponse(
        duty_HT1_kW=round(duty, 2),
        model_name=art["model_name"],
        model_version=art["version"],
        test_r2=round(art["metrics"]["test_r2"], 5),
        test_mae_kW=round(art["metrics"]["test_mae_kw"], 2),
        warnings=warnings,
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    art = artifact()
    all_errors = []
    for i, row in enumerate(req.items):
        errors, _ = check_row(row, art)
        all_errors += [f"ligne {i}: {e}" for e in errors]
    if all_errors:
        raise HTTPException(status_code=422, detail=all_errors[:50])
    preds = art["model"].predict(to_frame(req.items, art))
    return BatchResponse(duty_HT1_kW=[round(float(p), 2) for p in preds], model_version=art["version"])
