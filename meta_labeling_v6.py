"""
meta_labeling_v6.py — Pipeline de dos modelos con meta-labeling.

Modelo 1 (M1): binario — activo {+-1} vs neutral {0}
Modelo 2 (M2): binario — sube {+1} vs baja {-1}, entrenado solo en eventos activos

SFS separado para cada modelo usando LGBM como estimador.
Pares finales soportados: LGBM, XGB, MLP (mismo algoritmo para M1 y M2).

Flujo de inferencia:
    P(+1) = P(activo) * P(sube | activo)
    P(-1) = P(activo) * P(baja  | activo)
    P(0)  = 1 - P(activo)
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

TRAIN_RATIO = 0.8
K_MIN       = 6
K_MAX_M1    = 15
K_MAX_M2    = 22
N_SPLITS    = 5

# Candidatas (33 en total: 10 originales + 12 SFS v6 + 11 clasicos)
SFS_CANDIDATE_FEATURES = [
    "log_return_lag1", "log_return_lag4", "volatility_24h", "high_low_range",
    "log_return_roll_std72", "price_divergence", "volumeUSD_roll_std72",
    "transfer_count", "whale_volume_ratio", "dollar_bar_duration",
    "rsi_14", "macd_histogram", "bb_width", "atr_14", "volume_ratio",
    "return_skew_72h", "log_return_lag8", "log_return_lag12",
    "fee_apr_proxy", "tvl_change_pct", "unique_senders", "net_flow_proxy",
    "stoch_k", "stoch_d", "cci_20", "williams_r", "roc_10",
    "obv_momentum", "mfi_14", "ema_ratio", "adx_14", "donchian_pos", "vwap_dev",
    # Features de direccion (M2)
    "body_ratio", "upper_wick_ratio", "lower_wick_ratio",
    "bb_pct_b", "close_ema20_ratio",
    "log_return_lag2", "log_return_lag3", "log_return_lag6",
    "rsi_divergence",
    # Microestructura (LdP cap. 19)
    "bvc_imbalance",
    "amihud_illiquidity",
    "roll_spread",
]

LGBM_SFS_PARAMS = dict(
    n_estimators=200, learning_rate=0.05, max_depth=4, num_leaves=15,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
    objective="binary", random_state=42, verbose=-1,
)

LGBM_FINAL_PARAMS = dict(
    n_estimators=1000, learning_rate=0.01, max_depth=4, num_leaves=15,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
    objective="binary", random_state=42, verbose=-1,
)

XGB_FINAL_PARAMS = dict(
    n_estimators=1000, learning_rate=0.01, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0,
    objective="binary:logistic", eval_metric="logloss",
    random_state=42, tree_method="hist",
)


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def make_meta_labels(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Convierte labels originales {-1, 0, +1}:
      meta      : 1 si activo (±1), 0 si neutral
      direction : 1 si sube (+1), 0 si baja (-1)  — NaN si neutral
    """
    y = np.array(y, dtype=float)
    meta      = (y != 0).astype(int)
    direction = np.where(y == 1, 1, np.where(y == -1, 0, np.nan))
    return meta, direction


# ---------------------------------------------------------------------------
# Greedy forward SFS (binario)
# ---------------------------------------------------------------------------

def _cv_score_binary(X: np.ndarray, y: np.ndarray, tscv: TimeSeriesSplit) -> float:
    scores = cross_val_score(
        LGBMClassifier(**LGBM_SFS_PARAMS), X, y,
        cv=tscv, scoring="f1_macro", n_jobs=-1,
    )
    return float(scores.mean())


def _greedy_sfs(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    tscv: TimeSeriesSplit,
    k_max: int,
    label: str = "",
) -> list[tuple[str, float]]:
    selected: list[int] = []
    remaining: list[int] = list(range(len(feature_names)))
    order: list[tuple[str, float]] = []

    for step in range(k_max):
        best_score, best_idx = -np.inf, None
        for idx in remaining:
            score = _cv_score_binary(X[:, selected + [idx]], y, tscv)
            if score > best_score:
                best_score, best_idx = score, idx
        selected.append(best_idx)
        remaining.remove(best_idx)
        order.append((feature_names[best_idx], best_score))
        print(f"  [{label}] Paso {step+1:2d}/{k_max}: "
              f"+{feature_names[best_idx]:<30}  f1={best_score:.4f}")

    return order


def _run_sfs(
    X_raw: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    label: str,
    k_max: int = K_MAX_M1,
) -> dict:
    imp    = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    X = scaler.fit_transform(imp.fit_transform(X_raw))

    tscv  = TimeSeriesSplit(n_splits=N_SPLITS)
    order = _greedy_sfs(X, y, feature_names, tscv, k_max, label=label)

    results = {
        k: {"score": order[k-1][1], "features": [n for n, _ in order[:k]]}
        for k in range(K_MIN, k_max + 1)
    }
    best_k = max(results, key=lambda k: results[k]["score"])

    print(f"\n  [{label}] Mejor k={best_k} | f1={results[best_k]['score']:.4f}")
    print(f"  [{label}] Features: {results[best_k]['features']}\n")

    return {
        "results": results, "addition_order": order,
        "best_k": best_k, "best_features": results[best_k]["features"],
    }


# ---------------------------------------------------------------------------
# SFS publico
# ---------------------------------------------------------------------------

def run_sfs_m1(df: pd.DataFrame, labels: pd.Series) -> dict:
    """SFS para M1: activo vs neutral. Usa todos los eventos de train."""
    available = [f for f in SFS_CANDIDATE_FEATURES if f in df.columns]
    missing   = [f for f in SFS_CANDIDATE_FEATURES if f not in df.columns]
    if missing:
        print(f"  [SFS-M1] Ausentes: {missing}")

    n = len(df)
    train_end = int(n * TRAIN_RATIO)

    meta, _ = make_meta_labels(labels.values)

    X_raw = df[available].iloc[:train_end].values.astype(np.float32)
    y_m1  = meta[:train_end]

    print(f"\n  [SFS-M1] Train: {train_end} eventos | "
          f"activos={y_m1.sum()} ({y_m1.mean()*100:.1f}%) | "
          f"candidatas={len(available)} | k_max={K_MAX_M1}\n")

    sfs = _run_sfs(X_raw, y_m1, available, label="SFS-M1", k_max=K_MAX_M1)
    sfs["available"] = available
    return sfs


def run_sfs_m2(df: pd.DataFrame, labels: pd.Series) -> dict:
    """SFS para M2: sube vs baja. Solo eventos activos de train."""
    available = [f for f in SFS_CANDIDATE_FEATURES if f in df.columns]

    n = len(df)
    train_end = int(n * TRAIN_RATIO)

    y_all = labels.values
    meta, direction = make_meta_labels(y_all)

    # Filtrar solo eventos activos en train
    active_mask = (meta[:train_end] == 1)
    X_active = df[available].iloc[:train_end][active_mask].values.astype(np.float32)
    y_m2     = direction[:train_end][active_mask].astype(int)

    print(f"\n  [SFS-M2] Eventos activos en train: {len(y_m2)} | "
          f"sube={y_m2.sum()} ({y_m2.mean()*100:.1f}%) | "
          f"candidatas={len(available)} | k_max={K_MAX_M2}\n")

    sfs = _run_sfs(X_active, y_m2, available, label="SFS-M2", k_max=K_MAX_M2)
    sfs["available"] = available
    return sfs


# ---------------------------------------------------------------------------
# Plot curvas SFS
# ---------------------------------------------------------------------------

def plot_sfs_curves(sfs_m1: dict, sfs_m2: dict, save_path: str | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, sfs, title in [
        (axes[0], sfs_m1, "M1 — Activo vs Neutral"),
        (axes[1], sfs_m2, "M2 — Sube vs Baja"),
    ]:
        ks     = sorted(sfs["results"].keys())
        scores = [sfs["results"][k]["score"] for k in ks]
        best_k = sfs["best_k"]

        ax.plot(ks, scores, "o-", color="steelblue", linewidth=2, markersize=6)
        ax.axvline(best_k, color="crimson", linestyle="--", linewidth=1.5,
                   label=f"k={best_k} (f1={sfs['results'][best_k]['score']:.4f})")
        ax.set_title(f"SFS — {title}")
        ax.set_xlabel("k features")
        ax.set_ylabel("f1_macro CV")
        ax.set_xticks(ks)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Meta-Labeling v6 — Curvas SFS", fontsize=13)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  [SFS] Plot guardado: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# MetaPipeline
# ---------------------------------------------------------------------------

class MetaPipeline:
    """
    Pipeline de dos modelos con meta-labeling.

    M1: activo vs neutral  (todos los eventos)
    M2: sube   vs baja     (solo eventos activos)

    predict_proba retorna (n, 3): [P(-1), P(0), P(+1)]
    compatible con simulate_v4.
    """

    def __init__(
        self,
        model_type:   str,
        features_m1:  list[str],
        features_m2:  list[str],
        threshold:    float = 0.5,
    ):
        self.model_type  = model_type
        self.features_m1 = features_m1
        self.features_m2 = features_m2
        self.threshold   = threshold
        self.m1 = None
        self.m2 = None
        self.imp_m1    = SimpleImputer(strategy="median")
        self.scaler_m1 = StandardScaler()
        self.imp_m2    = SimpleImputer(strategy="median")
        self.scaler_m2 = StandardScaler()

    # --- constructores de modelos ---

    def _make_model(self):
        if self.model_type == "lgbm":
            return LGBMClassifier(**LGBM_FINAL_PARAMS)
        if self.model_type == "xgb":
            import xgboost as xgb
            return xgb.XGBClassifier(**XGB_FINAL_PARAMS, early_stopping_rounds=50)
        if self.model_type == "mlp":
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(
                hidden_layer_sizes=(32, 32, 16),
                activation="relu",
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=42,
                learning_rate_init=1e-3,
            )
        raise ValueError(f"model_type desconocido: {self.model_type}")

    # --- fit ---

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val:   pd.DataFrame,
        y_val:   np.ndarray,
    ) -> "MetaPipeline":
        meta_tr,  dir_tr  = make_meta_labels(y_train)
        meta_val, dir_val = make_meta_labels(y_val)

        # M1 — todos los eventos
        X_tr_m1  = self.scaler_m1.fit_transform(
            self.imp_m1.fit_transform(X_train[self.features_m1].values))
        X_val_m1 = self.scaler_m1.transform(
            self.imp_m1.transform(X_val[self.features_m1].values))

        self.m1 = self._make_model()
        self._fit_model(self.m1, X_tr_m1, meta_tr, X_val_m1, meta_val)

        # M2 — solo eventos activos
        active_tr  = meta_tr  == 1
        active_val = meta_val == 1

        X_tr_m2  = self.scaler_m2.fit_transform(
            self.imp_m2.fit_transform(X_train[self.features_m2].values[active_tr]))
        X_val_m2 = self.scaler_m2.transform(
            self.imp_m2.transform(X_val[self.features_m2].values[active_val]))
        y_tr_dir  = dir_tr[active_tr].astype(int)
        y_val_dir = dir_val[active_val].astype(int)

        self.m2 = self._make_model()
        self._fit_model(self.m2, X_tr_m2, y_tr_dir, X_val_m2, y_val_dir)

        return self

    def _fit_model(self, model, X_tr, y_tr, X_val, y_val):
        import lightgbm as lgb
        if self.model_type == "lgbm":
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        elif self.model_type == "xgb":
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        else:
            model.fit(X_tr, y_tr)  # MLP sklearn maneja early stopping internamente

    # --- predict ---

    def predict_proba(
        self,
        X: pd.DataFrame,
        side: pd.Series | None = None,
    ) -> np.ndarray:
        """
        Retorna (n, 3): [P(-1), P(0), P(+1)]

        Si se provee `side` (EMA crossover signal, +1 o -1 por barra):
          - M2 predice P(apuesta correcta | activo)
          - La direccion de la apuesta la da el side (EMA)
          - P(+1) = P(activo) * P(correct)  cuando side=+1
          - P(+1) = P(activo) * P(wrong)    cuando side=-1   (y viceversa)

        Sin side: comportamiento anterior (M2 predice direccion directamente).
        """
        X_m1 = self.scaler_m1.transform(
            self.imp_m1.transform(X[self.features_m1].values))
        X_m2 = self.scaler_m2.transform(
            self.imp_m2.transform(X[self.features_m2].values))

        p_active  = self.m1.predict_proba(X_m1)[:, 1]
        p_m2      = self.m2.predict_proba(X_m2)[:, 1]   # P(correct) o P(up)
        p_neu     = 1.0 - p_active

        if side is None:
            # Modo anterior: M2 predice direccion (up=+1, down=-1)
            p_pos = p_active * p_m2
            p_neg = p_active * (1.0 - p_m2)
        else:
            # Modo side-aware: M2 predice si la apuesta EMA sera correcta
            s = np.asarray(side, dtype=float)
            # Si side=+1 y M2 dice correct → P(+1) sube; si side=-1 → P(-1) sube
            p_correct = p_active * p_m2
            p_wrong   = p_active * (1.0 - p_m2)
            p_pos = np.where(s > 0, p_correct, p_wrong)
            p_neg = np.where(s < 0, p_correct, p_wrong)

        return np.column_stack([p_neg, p_neu, p_pos])
