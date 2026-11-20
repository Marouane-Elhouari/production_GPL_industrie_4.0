
import json 
import torch
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException

from pydantic import (BatchRequest, BatchResponse, ModelInfo,
                         PredictRequest, PredictResponse)
MODEL_DIR = Path(__file__).resolve().parent.parent /'artifacts'
MODEL_PATH_1 = MODEL_DIR / 'ridge_modelamélioré.joblib'
MODEL_PATH_2 = MODEL_DIR /'gnn_sans_aretes.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import os
_model_1 = None
_model_2= None
def env_bool(name, default):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes")
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



def load_model_1():
    global _model_1
    if _model_1 is not None :
        return _model_1
    model = joblib.load(MODEL_PATH_1)
    _model_1 = model
    return _model_1

def predict_by_one(req : PredictRequest) -> list : 
    if _model_1 is None :
        raise HTTPException(status_code = 422 , detail = 'manquante de model')
    art = _model_1
    errors , wariningd = check_row(req.features , art )
    if errors : 
        raise HTTPException(status_code = 422 , detail = errors)
    duty = float(art.predict([req.features] , art))[0])
    return PredictResponse(
    duty_HT1_KW  = round(duty , 2) ,

    model_name = art['model_name'] , 
    warnings = Warning , 
        
        )

def predict_batch(req: BatchRequest):
    
    art = _model_1
    all_errors = []
    for i, row in enumerate(req.items):
        errors, _ = check_row(row, art)
        all_errors += [f"ligne {i}: {e}" for e in errors]
    if all_errors:
        raise HTTPException(status_code=422, detail=all_errors[:50])
    preds = art["model"].predict((req.items, art))
    return BatchResponse(duty_HT1_kW=[round(float(p), 2) for p in preds], model_version=art["version"])


def load_model_2():
    global _model_2
    if _model_2 is not None :
        return _model_2
    model = joblib.load(MODEL_PATH_2)
    _model_2 = model

    return _model_2


