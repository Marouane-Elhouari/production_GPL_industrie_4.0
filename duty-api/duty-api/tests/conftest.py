import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["reflux_ratio", "feed_total_kg_h", "P_cond_barg", "T16_C", "Q_cond_kW", "Q_reb_kW"]


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame({
        "reflux_ratio": rng.uniform(2.0, 3.6, n),
        "feed_total_kg_h": rng.uniform(166680, 203720, n),
        "P_cond_barg": rng.uniform(18.0, 19.0, n),
        "T16_C": rng.uniform(137, 154, n),
        "Q_cond_kW": rng.uniform(16000, 32000, n),
        "Q_reb_kW": rng.uniform(3000, 23000, n),
    })
    y = 7500 + 120 * (X["T16_C"] - 144) + 0.02 * (X["feed_total_kg_h"] - 185000)
    model = make_pipeline(StandardScaler(), LinearRegression()).fit(X, y)
    artifact = {
        "model": model,
        "model_name": "test_linear",
        "features": FEATURES,
        "target": "duty_HT1",
        "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in FEATURES},
        "example": {c: float(X[c].median()) for c in FEATURES},
        "metrics": {"test_r2": 0.99, "test_mae_kw": 1.0, "cv_r2": 0.99},
        "version": "test-0001",
    }
    path = tmp_path_factory.mktemp("model") / "duty_model.joblib"
    joblib.dump(artifact, path)
    return str(path)


@pytest.fixture()
def client(model_path, monkeypatch):
    from fastapi.testclient import TestClient
    from app.main import app
    monkeypatch.setenv("MODEL_PATH", model_path)
    with TestClient(app) as c:
        yield c
