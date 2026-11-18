import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import optuna
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED = 42
N_SPLITS = 5
TEST_SIZE = 0.2
VAL_SIZE = 0.15
N_JOBS = -1
N_TRIALS = 20

TARGET_COL = "duty_HT1"
LEAK_COLS = ["Q_HT1_kW"]
DROP_COLS = ["converged", "status", "mass_balance_err", "P_reb_barg", "P_top_barg"]
MIN_STD = 1e-4

GRAPH_TOPK = 3
GRAPH_MIN_CORR = 0.30

GNN_HIDDEN = 32
GNN_LAYERS = 2
GNN_DROPOUT = 0.1
GNN_EPOCHS = 600
GNN_LR = 3e-3
GNN_WD = 1e-4
GNN_BATCH = 32
GNN_PATIENCE = 60

DATA_CSV = r"C:\Users\pc\Desktop\project\simulation chaine porduciton de ciment\data_final_v4_clean.csv"
OUT_DIR = r"C:\Users\pc\Desktop\project\simulation chaine porduciton de ciment\resultats_ml"

TREE_MODELS = ("XGBoost", "LightGBM", "CatBoost")

CYAN, PURPLE, YELLOW = "#0CF7AF", "#D826F8", "#F7C53E"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)


def save_fig(name):
    path = os.path.join(OUT_DIR, name)
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print("  figure:", path)


def load_data():
    if not os.path.exists(DATA_CSV):
        raise SystemExit(f"CSV introuvable: {DATA_CSV}")
    raw = pd.read_csv(DATA_CSV)
    print(f"CSV brut: {raw.shape[0]} lignes, {raw.shape[1]} colonnes")
    if "status" in raw.columns:
        print("Statuts:", raw["status"].value_counts().to_dict())
    df = raw.copy()
    if "converged" in df.columns:
        df = df[df["converged"] == 1]
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    df = df.drop(columns=[c for c in LEAK_COLS + DROP_COLS if c in df.columns])
    df = df.select_dtypes(include=[np.number]).dropna().drop_duplicates().reset_index(drop=True)
    if TARGET_COL not in df.columns:
        raise SystemExit(f"Colonne cible '{TARGET_COL}' absente du CSV")
    low_var = [c for c in df.columns if c != TARGET_COL and df[c].std() < MIN_STD]
    if low_var:
        print("Colonnes quasi constantes retirees:", low_var)
        df = df.drop(columns=low_var)
    print(f"Dataset utilisable: {df.shape[0]} lignes, {df.shape[1] - 1} features\n")
    if len(df) < 500:
        print("[attention] moins de 500 points : les resultats ML/GNN seront peu stables.\n")
    return df


def eda(df):
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "nunique": df.nunique(),
        "mean": df.mean(),
        "std": df.std(),
        "min": df.min(),
        "max": df.max(),
    })
    print("=== EDA ===")
    print(summary.round(4).to_string())
    print("Doublons:", int(df.duplicated().sum()), "\n")

    plt.figure(figsize=(7, 4))
    sns.histplot(df[TARGET_COL], kde=True, color=CYAN)
    plt.xlabel(f"{TARGET_COL} (kW)")
    plt.title(f"Distribution de {TARGET_COL}")
    save_fig("eda_target.png")

    corr_target = df.corr()[TARGET_COL].drop(TARGET_COL)
    top = corr_target.abs().sort_values(ascending=False).head(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for col, ax in zip(top, axes.flatten()):
        ax.scatter(df[col], df[TARGET_COL], s=14, alpha=0.7, color=PURPLE)
        ax.set_xlabel(col)
        ax.set_ylabel(TARGET_COL)
        ax.set_title(f"corr = {corr_target[col]:+.3f}")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Features les plus liees a {TARGET_COL}")
    save_fig("eda_scatter_top.png")

    corr = df.corr()
    size = max(8, 0.55 * len(corr))
    plt.figure(figsize=(size, size * 0.85))
    sns.heatmap(corr, cmap="RdYlGn", center=0, annot=len(corr) <= 14,
                fmt=".2f", linewidths=0.4, square=True)
    plt.title("Correlations (Pearson)")
    save_fig("eda_heatmap.png")


def build_feature_graph(X_train, y_train):
    feats = list(X_train.columns)
    n = len(feats)
    corr = X_train.corr(method="spearman").abs().values.copy()
    np.fill_diagonal(corr, 0.0)
    adj = np.zeros((n, n))
    for i in range(n):
        for j in np.argsort(-corr[i])[:GRAPH_TOPK]:
            if corr[i, j] >= GRAPH_MIN_CORR:
                adj[i, j] = adj[j, i] = corr[i, j]
    G = nx.from_numpy_array(adj)
    G = nx.relabel_nodes(G, {i: f for i, f in enumerate(feats)})
    corr_target = {f: abs(X_train[f].corr(y_train, method="spearman")) for f in feats}
    nx.set_node_attributes(G, corr_target, "corr_target")
    return G, adj


def draw_graph(G, filename="graphe_features.png"):
    fig, ax = plt.subplots(figsize=(14, 11))
    pos = nx.spring_layout(G, seed=SEED, weight="weight", k=1.5 / np.sqrt(max(len(G), 1)))
    node_vals = [G.nodes[n]["corr_target"] for n in G.nodes]
    widths = [0.5 + 4.0 * G[u][v]["weight"] for u, v in G.edges]
    nodes = nx.draw_networkx_nodes(G, pos, node_color=node_vals, cmap="viridis",
                                   vmin=0, vmax=1, node_size=1100, ax=ax)
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.45, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
    cbar = plt.colorbar(nodes, ax=ax, shrink=0.7)
    cbar.set_label(f"|Spearman| avec {TARGET_COL}")
    ax.set_title("Graphe des relations entre features "
                 f"(top-{GRAPH_TOPK} voisins, |corr| >= {GRAPH_MIN_CORR})")
    ax.set_axis_off()
    save_fig(filename)

    edges = sorted(((u, v, d["weight"]) for u, v, d in G.edges(data=True)), key=lambda t: -t[2])
    pd.DataFrame(edges, columns=["feature_a", "feature_b", "abs_spearman"]).to_csv(
        os.path.join(OUT_DIR, "graphe_aretes.csv"), index=False)
    print("Aretes les plus fortes:")
    for u, v, w in edges[:8]:
        print(f"  {u} -- {v}: {w:.3f}")
    print(f"Noeuds: {G.number_of_nodes()} | aretes: {G.number_of_edges()}\n")


def build_model(name, params):
    if name == "XGBoost":
        return XGBRegressor(**params, random_state=SEED, n_jobs=N_JOBS, tree_method="hist")
    if name == "LightGBM":
        return LGBMRegressor(**params, random_state=SEED, n_jobs=N_JOBS, verbose=-1)
    if name == "CatBoost":
        return CatBoostRegressor(**params, random_seed=SEED, thread_count=N_JOBS,
                                 verbose=0, allow_writing_files=False)
    raise ValueError(name)


def suggest_params(name, trial):
    if name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        }
    if name == "LightGBM":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 31),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 20),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        }
    return {
        "iterations": trial.suggest_int("iterations", 100, 500, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
        "depth": trial.suggest_int("depth", 3, 6),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
    }


def make_objective(name, X, y):
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    def objective(trial):
        model = build_model(name, suggest_params(name, trial))
        return cross_val_score(model, X, y, cv=kf, scoring="r2").mean()

    return objective


class FeatureGraphNet(nn.Module):
    def __init__(self, n_nodes, adj_norm, hidden, n_layers, dropout):
        super().__init__()
        self.register_buffer("A", adj_norm)
        self.node_w = nn.Parameter(torch.randn(n_nodes, hidden) * 0.1)
        self.node_b = nn.Parameter(torch.zeros(n_nodes, hidden))
        self.lins = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(n_nodes * hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        h = x.unsqueeze(-1) * self.node_w + self.node_b
        for lin, norm in zip(self.lins, self.norms):
            m = torch.matmul(self.A, lin(h))
            h = h + self.drop(torch.relu(norm(m)))
        return self.head(h.flatten(1)).squeeze(-1)


def normalize_adjacency(adj):
    a = adj + np.eye(adj.shape[0])
    d = a.sum(axis=1)
    return a / np.sqrt(np.outer(d, d))


def train_gnn(X_train, y_train, X_test, adj, label):
    set_seed()
    sx = StandardScaler().fit(X_train.values)
    sy = StandardScaler().fit(y_train.values.reshape(-1, 1))
    Xs = sx.transform(X_train.values).astype(np.float32)
    ys = sy.transform(y_train.values.reshape(-1, 1)).ravel().astype(np.float32)
    Xt = torch.tensor(sx.transform(X_test.values), dtype=torch.float32)

    X_tr, X_va, y_tr, y_va = train_test_split(Xs, ys, test_size=VAL_SIZE, random_state=SEED)
    X_tr, X_va = torch.tensor(X_tr), torch.tensor(X_va)
    y_tr, y_va = torch.tensor(y_tr), torch.tensor(y_va)

    a_norm = torch.tensor(normalize_adjacency(adj), dtype=torch.float32)
    model = FeatureGraphNet(Xs.shape[1], a_norm, GNN_HIDDEN, GNN_LAYERS, GNN_DROPOUT)
    opt = torch.optim.AdamW(model.parameters(), lr=GNN_LR, weight_decay=GNN_WD)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=20)
    loss_fn = nn.MSELoss()

    best_val, best_state, wait = float("inf"), None, 0
    hist = {"train": [], "val": []}
    for epoch in range(GNN_EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr))
        run = 0.0
        for i in range(0, len(X_tr), GNN_BATCH):
            idx = perm[i:i + GNN_BATCH]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            loss = loss_fn(model(X_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
            run += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(X_va), y_va).item()
        sched.step(val)
        hist["train"].append(run / len(X_tr))
        hist["val"].append(val)
        if val < best_val - 1e-5:
            best_val = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= GNN_PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(Xt).numpy()
    pred = sy.inverse_transform(pred.reshape(-1, 1)).ravel()
    print(f"  {label}: {len(hist['val'])} epoques, meilleure val MSE (std) = {best_val:.5f}")
    return pred, hist


def score(name, y_true, y_pred, cv_r2=np.nan):
    return {
        "model": name,
        "cv_R2": cv_r2,
        "test_R2": r2_score(y_true, y_pred),
        "test_MAE_kW": mean_absolute_error(y_true, y_pred),
        "test_RMSE_kW": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def plot_predictions(y_test, preds, results):
    r2 = {r["model"]: r["test_R2"] for r in results}
    names = list(preds.keys())
    ncols = 3
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows), constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()
    lo, hi = float(y_test.min()), float(y_test.max())
    for name, ax in zip(names, axes):
        ax.scatter(y_test, preds[name], s=18, alpha=0.7, color=CYAN)
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.3)
        ax.set_xlabel("duty HT-1 reel (kW)")
        ax.set_ylabel("duty HT-1 predit (kW)")
        ax.set_title(f"{name}\nR2 = {r2[name]:.4f}")
        ax.grid(True, alpha=0.3)
    for ax in axes[len(names):]:
        ax.axis("off")
    fig.suptitle("Predictions vs realite (jeu de test)", fontweight="bold")
    save_fig("predictions_vs_actual.png")


def plot_comparison(results):
    df = pd.DataFrame(results).sort_values("test_R2")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    axes[0].barh(df["model"], df["test_R2"], color=PURPLE)
    axes[0].set_xlabel("R2 (test)")
    axes[0].set_title("R2")
    axes[1].barh(df["model"], df["test_MAE_kW"], color=YELLOW)
    axes[1].set_xlabel("MAE (kW)")
    axes[1].set_title("Erreur absolue moyenne")
    save_fig("comparaison_modeles.png")


def plot_gnn_history(histories):
    plt.figure(figsize=(8, 5))
    for label, h in histories.items():
        plt.plot(h["val"], label=f"{label} (val)")
        plt.plot(h["train"], "--", alpha=0.6, label=f"{label} (train)")
    plt.yscale("log")
    plt.xlabel("Epoque")
    plt.ylabel("MSE (cible standardisee)")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.title("Courbes d'apprentissage GNN")
    save_fig("gnn_apprentissage.png")


def plot_importance(model, X_test, y_test, name):
    imp = permutation_importance(model, X_test, y_test, n_repeats=15,
                                 random_state=SEED, scoring="r2")
    s = pd.Series(imp.importances_mean, index=X_test.columns).sort_values().tail(12)
    plt.figure(figsize=(8, 6))
    plt.barh(s.index, s.values, color=CYAN)
    plt.xlabel("Baisse de R2 (permutation)")
    plt.title(f"Importance des features - {name}")
    save_fig("importance_features.png")


def main():
    set_seed()
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_data()
    eda(df)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}\n")

    G, adj = build_feature_graph(X_train, y_train)
    draw_graph(G)

    results, preds, tuned = [], {}, {}

    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-4, 3, 30)))
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    cv_ridge = cross_val_score(ridge, X_train, y_train, cv=kf, scoring="r2").mean()
    ridge.fit(X_train, y_train)
    preds["Ridge"] = ridge.predict(X_test)
    results.append(score("Ridge", y_test, preds["Ridge"], cv_ridge))

    for name in TREE_MODELS:
        print(f"Optimisation {name} ({N_TRIALS} essais)...")
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=SEED))
        study.optimize(make_objective(name, X_train, y_train), n_trials=N_TRIALS)
        print(f"  CV R2 = {study.best_value:.5f} | params = {study.best_params}")
        model = build_model(name, study.best_params).fit(X_train, y_train)
        tuned[name] = model
        preds[name] = model.predict(X_test)
        results.append(score(name, y_test, preds[name], study.best_value))

    print("\nEntrainement GNN...")
    histories = {}
    preds["GNN (graphe)"], histories["graphe"] = train_gnn(
        X_train, y_train, X_test, adj, "GNN avec graphe de correlations")
    results.append(score("GNN (graphe)", y_test, preds["GNN (graphe)"]))
    preds["GNN (sans aretes)"], histories["sans aretes"] = train_gnn(
        X_train, y_train, X_test, np.zeros_like(adj), "GNN sans aretes (ablation)")
    results.append(score("GNN (sans aretes)", y_test, preds["GNN (sans aretes)"]))

    table = pd.DataFrame(results).sort_values("test_R2", ascending=False)
    print("\n=== RESULTATS (jeu de test) ===")
    print(table.round(5).to_string(index=False))
    table.to_csv(os.path.join(OUT_DIR, "resultats_modeles.csv"), index=False)

    plot_predictions(y_test, preds, results)
    plot_comparison(results)
    plot_gnn_history(histories)
    best_tree = max(TREE_MODELS, key=lambda n: r2_score(y_test, preds[n]))
    plot_importance(tuned[best_tree], X_test, y_test, best_tree)

    gain = r2_score(y_test, preds["GNN (graphe)"]) - r2_score(y_test, preds["GNN (sans aretes)"])
    print(f"\nApport du graphe (R2 GNN graphe - R2 GNN sans aretes): {gain:+.5f}")
    print("Resultats dans:", OUT_DIR)


if __name__ == "__main__":
    main()