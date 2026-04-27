"""
feature_selection_v6.py — Sequential Feature Selection con LightGBM (v6).

Estrategia: greedy forward SFS sobre el train set (80% temporal).
Evalua k en [K_MIN, K_MAX] y reporta la curva de score vs k.

Uso standalone:
    python feature_selection_v6.py

Uso desde otro modulo:
    from feature_selection_v6 import run_sfs, SFS_CANDIDATE_FEATURES
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

# ---------------------------------------------------------------------------
# Configuracion
# ---------------------------------------------------------------------------

K_MIN       = 6
K_MAX       = 15
N_SPLITS    = 5
TRAIN_RATIO = 0.8

# LGBM liviano para SFS — menos estimadores para velocidad
LGBM_SFS_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    num_leaves=15,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    objective="multiclass",
    num_class=3,
    random_state=42,
    verbose=-1,
)

# 22 candidatas: 10 originales + 12 nuevas
SFS_CANDIDATE_FEATURES = [
    # Originales — Binance
    "log_return_lag1",
    "log_return_lag4",
    "volatility_24h",
    "high_low_range",
    "log_return_roll_std72",
    # Originales — The Graph
    "price_divergence",
    "volumeUSD_roll_std72",
    # Originales — Dune
    "transfer_count",
    "whale_volume_ratio",
    # Originales — Dollar bar
    "dollar_bar_duration",
    # Nuevas — tecnicas Binance
    "rsi_14",
    "macd_histogram",
    "bb_width",
    "atr_14",
    "volume_ratio",
    "return_skew_72h",
    "log_return_lag8",
    "log_return_lag12",
    # Nuevas — The Graph
    "fee_apr_proxy",
    "tvl_change_pct",
    # Nuevas — Dune
    "unique_senders",
    "net_flow_proxy",
    # Indicadores clasicos
    "stoch_k",
    "stoch_d",
    "cci_20",
    "williams_r",
    "roc_10",
    "obv_momentum",
    "mfi_14",
    "ema_ratio",
    "adx_14",
    "donchian_pos",
    "vwap_dev",
]

LABEL_MAP = {-1: 0, 0: 1, 1: 2}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _remap(y: np.ndarray) -> np.ndarray:
    return np.array([LABEL_MAP[int(v)] for v in y], dtype=np.int64)


def _cv_score(X: np.ndarray, y: np.ndarray, tscv: TimeSeriesSplit) -> float:
    scores = cross_val_score(
        LGBMClassifier(**LGBM_SFS_PARAMS),
        X, y,
        cv=tscv,
        scoring="f1_macro",
        n_jobs=-1,
    )
    return float(scores.mean())


# ---------------------------------------------------------------------------
# Greedy forward SFS
# ---------------------------------------------------------------------------

def _greedy_forward_sfs(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    tscv: TimeSeriesSplit,
    k_max: int,
) -> list[tuple[str, float]]:
    """
    Greedy forward SFS.
    En cada paso agrega la feature que maximiza f1_macro con CV.
    Retorna lista de (feature_name, score_con_k_features) en orden de adicion.
    """
    selected: list[int] = []
    remaining: list[int] = list(range(len(feature_names)))
    addition_order: list[tuple[str, float]] = []

    for step in range(k_max):
        best_score = -np.inf
        best_idx   = None

        for feat_idx in remaining:
            candidate = selected + [feat_idx]
            score = _cv_score(X[:, candidate], y, tscv)
            if score > best_score:
                best_score = score
                best_idx   = feat_idx

        selected.append(best_idx)
        remaining.remove(best_idx)
        addition_order.append((feature_names[best_idx], best_score))

        print(f"  Paso {step + 1:2d}/{k_max}: +{feature_names[best_idx]:<30}  "
              f"f1_macro={best_score:.4f}")

    return addition_order


# ---------------------------------------------------------------------------
# Interfaz publica
# ---------------------------------------------------------------------------

def run_sfs(df: pd.DataFrame, labels: pd.Series) -> dict:
    """
    Corre greedy forward SFS sobre el train set (80% temporal).

    Parametros
    ----------
    df     : DataFrame con features, indexado temporalmente.
    labels : Serie con labels {-1, 0, 1} alineada con df.

    Retorna
    -------
    dict con:
        results        -> {k: {"score": float, "features": list[str]}}
        addition_order -> [(feature, score), ...] en orden de adicion
        best_k         -> k optimo
        best_score     -> f1_macro del mejor k
        best_features  -> lista de features del mejor k
    """
    available = [f for f in SFS_CANDIDATE_FEATURES if f in df.columns]
    missing   = [f for f in SFS_CANDIDATE_FEATURES if f not in df.columns]

    print(f"\n  [SFS] Candidatas disponibles: {len(available)}/{len(SFS_CANDIDATE_FEATURES)}")
    if missing:
        print(f"  [SFS] Ausentes (no estan en el dataset): {missing}")

    # Split temporal — solo train
    n         = len(df)
    train_end = int(n * TRAIN_RATIO)
    X_raw = df[available].iloc[:train_end].values.astype(np.float32)
    y     = _remap(labels.iloc[:train_end].values)

    # Imputacion + escalado (fit solo sobre train)
    imp    = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = scaler.fit_transform(imp.fit_transform(X_raw))

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    print(f"  [SFS] Train: {train_end} eventos | k=[{K_MIN}, {K_MAX}] | "
          f"CV: TimeSeriesSplit({N_SPLITS})\n")

    addition_order = _greedy_forward_sfs(X, y, available, tscv, K_MAX)

    # Construir results por k en [K_MIN, K_MAX]
    results: dict[int, dict] = {}
    for k in range(K_MIN, K_MAX + 1):
        results[k] = {
            "score":    addition_order[k - 1][1],
            "features": [name for name, _ in addition_order[:k]],
        }

    best_k = max(results, key=lambda k: results[k]["score"])

    print(f"\n  [SFS] Mejor k={best_k} | f1_macro={results[best_k]['score']:.4f}")
    print(f"  [SFS] Features: {results[best_k]['features']}")

    return {
        "results":           results,
        "addition_order":    addition_order,
        "best_k":            best_k,
        "best_score":        results[best_k]["score"],
        "best_features":     results[best_k]["features"],
        "available_features": available,
    }


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_sfs_curve(sfs_output: dict, save_path: str | None = None) -> None:
    results = sfs_output["results"]
    ks      = sorted(results.keys())
    scores  = [results[k]["score"] for k in ks]
    best_k  = sfs_output["best_k"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ks, scores, "o-", color="steelblue", linewidth=2, markersize=7)
    ax.axvline(best_k, color="crimson", linestyle="--", linewidth=1.5,
               label=f"optimo k={best_k} (f1={results[best_k]['score']:.4f})")
    ax.set_xlabel("Numero de features (k)")
    ax.set_ylabel("f1_macro (CV train, TimeSeriesSplit 5-fold)")
    ax.set_title("SFS Greedy Forward — Score vs k (LightGBM)")
    ax.set_xticks(ks)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  [SFS] Plot guardado: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Standalone
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    BASE_DIR     = Path(__file__).parent
    parquet_path = BASE_DIR / "data" / "processed" / "features_4h.parquet"

    # Cargar o reconstruir dataset
    if parquet_path.exists():
        df_raw = pd.read_parquet(parquet_path)
        print(f"  [SFS] Dataset cargado: {df_raw.shape}")
    else:
        print("  [SFS] features_4h.parquet no encontrado. Reconstruyendo...")
        from process_data_v5 import build_dataset_4h
        df_raw = build_dataset_4h()

    # Triple Barrier para labels
    from triple_barrier_v4 import build_labels_triple_barrier

    df_idx = df_raw.copy()
    df_idx["datetime"] = pd.to_datetime(df_idx["datetime"])
    df_idx = df_idx.set_index("datetime").sort_index()

    events = build_labels_triple_barrier(df_idx, freq="4h")
    if events.empty:
        print("  [SFS] ERROR: Triple Barrier no produjo eventos.")
        sys.exit(1)

    labels = events["bin"].astype(int).rename("label")
    joined = df_idx.join(labels, how="inner").dropna(subset=["label"])

    label_series = joined["label"]
    feature_df   = joined.drop(columns=["label"])

    # Correr SFS
    sfs_output = run_sfs(feature_df, label_series)

    # Guardar plot
    plot_sfs_curve(sfs_output, save_path=str(BASE_DIR / "sfs_curve.png"))

    # Resumen final
    print("\n" + "=" * 65)
    print("RESUMEN SFS — Score por k")
    print("=" * 65)
    for k, v in sorted(sfs_output["results"].items()):
        marker = "  <- OPTIMO" if k == sfs_output["best_k"] else ""
        print(f"  k={k:2d} | f1={v['score']:.4f} | {v['features']}{marker}")

    print("\nOrden de adicion de features:")
    for i, (feat, score) in enumerate(sfs_output["addition_order"], 1):
        print(f"  {i:2d}. {feat:<30}  acum f1={score:.4f}")
