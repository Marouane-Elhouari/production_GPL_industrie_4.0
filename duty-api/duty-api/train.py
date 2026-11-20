import argparse
import hashlib
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

SEED = 42
TARGET = "duty_HT1"
CORE_FEATURES = ["reflux_ratio", "feed_total_kg_h", "P_cond_barg", "T16_C", "T15a_C"]
EXCLUDED = [TARGET, "Q_HT1_kW", "converged", "status", "mass_balance_err", "P_reb_barg", "P_top_barg"]
MIN_STD = 1e-4
ALPHAS = np.logspace(-4, 3, 30)


def candidates(n_features):
    models = {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=ALPHAS)),
        "hist_gb": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=SEED),
    }
    if n_features <= 25:
        models["ridge_poly2"] = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                                              StandardScaler(), RidgeCV(alphas=ALPHAS))
    if n_features <= 8:
        models["ridge_poly3"] = make_pipeline(StandardScaler(), PolynomialFeatures(3, include_bias=False),
                                              StandardScaler(), RidgeCV(alphas=ALPHAS))
    return models


def read_clean(csv_path):
    df = pd.read_csv(csv_path)
    if "converged" in df.columns:
        df = df[df["converged"] == 1]
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    return df


def select_features(df, mode):
    if mode == "core":
        return [c for c in CORE_FEATURES if c in df.columns]
    numeric = df.select_dtypes(include=[np.number])
    return [c for c in numeric.columns if c not in EXCLUDED and numeric[c].std() >= MIN_STD]


def file_hash(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="models/duty_model.joblib")
    parser.add_argument("--mode", choices=["all", "core"], default="all")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = read_clean(args.csv)
    if TARGET not in df.columns:
        raise SystemExit(f"Colonne cible '{TARGET}' absente du CSV")
    features = select_features(df, args.mode)
    df = df[features + [TARGET]].dropna().drop_duplicates().reset_index(drop=True)
    print(f"Mode: {args.mode} | {len(df)} lignes | {len(features)} features")
    print("Features:", features)
    X, y = df[features], df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=SEED)
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

    cv_scores = {}
    for name, model in candidates(len(features)).items():
        cv_scores[name] = cross_val_score(model, X_train, y_train, cv=kf, scoring="r2").mean()
        print(f"  {name:12s} CV R2 = {cv_scores[name]:.5f}")

    best = max(cv_scores, key=cv_scores.get)
    holdout_model = clone(candidates(len(features))[best]).fit(X_train, y_train)
    pred = holdout_model.predict(X_test)
    metrics = {
        "model_name": best,
        "mode": args.mode,
        "cv_r2": float(cv_scores[best]),
        "test_r2": float(r2_score(y_test, pred)),
        "test_mae_kw": float(mean_absolute_error(y_test, pred)),
        "test_rmse_kw": float(np.sqrt(mean_squared_error(y_test, pred))),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "all_cv_r2": {k: float(v) for k, v in cv_scores.items()},
    }
    print(f"Meilleur modele: {best} | test R2 = {metrics['test_r2']:.5f} "
          f"| MAE = {metrics['test_mae_kw']:.2f} kW | RMSE = {metrics['test_rmse_kw']:.2f} kW")

    final_model = clone(candidates(len(features))[best]).fit(X, y)
    artifact = {
        "model": final_model,
        "model_name": best,
        "features": features,
        "target": TARGET,
        "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in features},
        "example": {c: float(X[c].median()) for c in features},
        "metrics": metrics,
        "version": f"{datetime.now():%Y%m%d}-{file_hash(args.csv)}",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(artifact, args.out)
    with open(os.path.splitext(args.out)[0] + "_metrics.json", "w", encoding="utf-8") as fh:
        json.dump({"version": artifact["version"], "features": features,
                   "ranges": artifact["ranges"], **metrics}, fh, indent=2)
    print("Modele sauvegarde:", args.out, "| version:", artifact["version"])


if __name__ == "__main__":
    main()
