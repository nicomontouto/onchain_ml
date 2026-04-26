"""
models_v4.py -- Tres modelos multiclase para el pipeline v4.

Clasificacion de labels Triple Barrier: -1, 0, +1
Internamente se mapean a 0, 1, 2 (orden: -1 -> 0, 0 -> 1, +1 -> 2)

Modelos:
  - XGBoost multi:softprob con early stopping (n_estimators=1000, lr=0.01)
  - LightGBM multiclass con early stopping
  - MLP PyTorch 16x3 con BatchNorm + Dropout

Todos comparten la interfaz:
    fit(X_train, y_train, X_val, y_val, feature_names, class_weights=None)
    predict_proba(X) -> (n, 3) con cols (P(-1), P(0), P(+1))
    feature_importance(X_ref=None, y_ref=None) -> np.array
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field

import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

# Evitar deadlock entre OpenMP de LightGBM y el de PyTorch en macOS.
# Se fija UN solo thread y se permite duplicidad de libomp.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# ---------------------------------------------------------------------------
# Mapeo de labels
# ---------------------------------------------------------------------------
LABEL_TO_IDX = {-1: 0, 0: 1, 1: 2}
IDX_TO_LABEL = {0: -1, 1: 0, 2: 1}


def remap_labels(y: np.ndarray) -> np.ndarray:
    return np.array([LABEL_TO_IDX[int(v)] for v in y], dtype=np.int64)


def unmap_labels(y_idx: np.ndarray) -> np.ndarray:
    return np.array([IDX_TO_LABEL[int(v)] for v in y_idx], dtype=np.int64)


# ---------------------------------------------------------------------------
# XGBoost
# ---------------------------------------------------------------------------

XGB_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="multi:softprob",
    num_class=3,
    eval_metric="mlogloss",
    random_state=42,
    tree_method="hist",
)


class XGBWrapper:
    def __init__(self, **overrides):
        import xgboost as xgb
        params = {**XGB_PARAMS, **overrides}
        self._xgb = xgb
        self.model = xgb.XGBClassifier(
            **params,
            early_stopping_rounds=50,
        )
        self.feature_names_: list[str] = []
        self.best_iteration_ = None

    def fit(self, X_tr, y_tr, X_val, y_val, feature_names=None, sample_weight=None):
        self.feature_names_ = list(feature_names) if feature_names else []
        y_tr_idx = remap_labels(y_tr)
        y_val_idx = remap_labels(y_val)
        self.model.fit(
            X_tr, y_tr_idx,
            eval_set=[(X_val, y_val_idx)],
            sample_weight=sample_weight,
            verbose=False,
        )
        self.best_iteration_ = getattr(self.model, "best_iteration", None)
        return self

    def predict_proba(self, X) -> np.ndarray:
        # Cols ya estan en orden (-1, 0, +1) porque el mapping es ordenado
        return self.model.predict_proba(X)

    def feature_importance(self, X=None, y=None) -> np.ndarray:
        return self.model.feature_importances_


# ---------------------------------------------------------------------------
# LightGBM
# ---------------------------------------------------------------------------

LGBM_PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    num_leaves=15,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="multiclass",
    num_class=3,
    random_state=42,
    verbose=-1,
)


class LGBMWrapper:
    def __init__(self, **overrides):
        import lightgbm as lgb
        params = {**LGBM_PARAMS, **overrides}
        self._lgb = lgb
        self.model = lgb.LGBMClassifier(**params)
        self.feature_names_: list[str] = []
        self.best_iteration_ = None

    def fit(self, X_tr, y_tr, X_val, y_val, feature_names=None, sample_weight=None):
        self.feature_names_ = list(feature_names) if feature_names else []
        y_tr_idx = remap_labels(y_tr)
        y_val_idx = remap_labels(y_val)
        import lightgbm as lgb
        self.model.fit(
            X_tr, y_tr_idx,
            eval_set=[(X_val, y_val_idx)],
            sample_weight=sample_weight,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )
        self.best_iteration_ = getattr(self.model, "best_iteration_", None)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return self.model.predict_proba(X)

    def feature_importance(self, X=None, y=None) -> np.ndarray:
        # Normalizar a fraccion (como XGB)
        imp = self.model.feature_importances_.astype(float)
        s = imp.sum()
        return imp / s if s > 0 else imp


# ---------------------------------------------------------------------------
# MLP PyTorch
# ---------------------------------------------------------------------------

@dataclass
class MLPHistory:
    train_loss: list = field(default_factory=list)
    val_loss:   list = field(default_factory=list)
    train_acc:  list = field(default_factory=list)
    val_acc:    list = field(default_factory=list)
    best_epoch: int = -1


def _build_mlp(input_dim: int):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(input_dim, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(16, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(16, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(16, 3),
    )


class MLPWrapper:
    """
    Arquitectura 16x3 (tres bloques lineales de ancho 16 + salida de 3).

    La capa Softmax del enunciado se omite porque CrossEntropyLoss de
    PyTorch ya combina LogSoftmax + NLL: agregarla generaria gradientes
    duplicados. En inferencia se aplica softmax explicitamente sobre
    los logits para obtener probabilidades.
    """

    def __init__(
        self,
        epochs: int = 200,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 20,
        grad_clip: float = 1.0,
        device: str | None = None,
    ):
        import torch
        self._torch = torch
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.grad_clip = grad_clip
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = None
        self.feature_names_: list[str] = []
        self.history = MLPHistory()

    def fit(self, X_tr, y_tr, X_val, y_val, feature_names=None, sample_weight=None):
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        torch.set_num_threads(1)

        self.feature_names_ = list(feature_names) if feature_names else []
        y_tr_idx = remap_labels(y_tr)
        y_val_idx = remap_labels(y_val)

        X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr_idx, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
        y_val_t = torch.tensor(y_val_idx, dtype=torch.long).to(self.device)

        ds = TensorDataset(X_tr_t, y_tr_t)
        loader = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        input_dim = X_tr.shape[1]
        self.net = _build_mlp(input_dim).to(self.device)

        if sample_weight is not None:
            # sample_weight se provee por clase para 'balanced' --
            # lo traducimos a pesos de clase para CrossEntropyLoss
            import numpy as _np
            classes, counts = _np.unique(y_tr_idx, return_counts=True)
            class_weights = _np.zeros(3, dtype=_np.float32)
            n = len(y_tr_idx)
            for c, cnt in zip(classes, counts):
                class_weights[c] = n / (3 * cnt)
            w = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
            criterion = torch.nn.CrossEntropyLoss(weight=w)
        else:
            criterion = torch.nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        best_state = None
        bad_epochs = 0

        for epoch in range(1, self.epochs + 1):
            self.net.train()
            running = 0.0
            correct = 0
            total = 0
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                optimizer.step()
                running += float(loss.item()) * xb.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += xb.size(0)
            tr_loss = running / max(total, 1)
            tr_acc = correct / max(total, 1)

            # Validacion
            self.net.eval()
            with torch.no_grad():
                logits_v = self.net(X_val_t)
                val_loss = float(criterion(logits_v, y_val_t).item())
                val_acc = float((logits_v.argmax(dim=1) == y_val_t).float().mean().item())

            self.history.train_loss.append(tr_loss)
            self.history.val_loss.append(val_loss)
            self.history.train_acc.append(tr_acc)
            self.history.val_acc.append(val_acc)

            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                self.history.best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    print(f"    [MLP] early stop epoch={epoch} "
                          f"(best={self.history.best_epoch}, val_loss={best_val_loss:.4f})")
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def predict_proba(self, X) -> np.ndarray:
        import torch
        self.net.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32, device=self.device)
            logits = self.net(Xt)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
        return proba

    def feature_importance(self, X=None, y=None, n_repeats: int = 5,
                           rng_seed: int = 42) -> np.ndarray:
        """
        Permutation importance sobre el set que se le pase (usualmente test).
        Se mide la caida en accuracy al permutar cada feature.
        """
        if X is None or y is None:
            return np.zeros(len(self.feature_names_) or 0)

        rng = np.random.default_rng(rng_seed)
        y_idx = remap_labels(y)

        # Baseline
        base_preds = np.argmax(self.predict_proba(X), axis=1)
        base_acc = float((base_preds == y_idx).mean())

        n_features = X.shape[1]
        importances = np.zeros(n_features, dtype=float)
        for j in range(n_features):
            drops = []
            for _ in range(n_repeats):
                Xp = X.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                p = np.argmax(self.predict_proba(Xp), axis=1)
                acc = float((p == y_idx).mean())
                drops.append(base_acc - acc)
            importances[j] = float(np.mean(drops))
        return importances


# ---------------------------------------------------------------------------
# Helper: class_weight = 'balanced' (sklearn style)
# ---------------------------------------------------------------------------

def compute_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    sklearn-style balanced class weights, proyectados a sample_weight.
    """
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y)
    cw = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    mapping = dict(zip(classes, cw))
    return np.array([mapping[int(v)] for v in y], dtype=np.float64)


def is_severely_imbalanced(y: np.ndarray, threshold: float = 0.60) -> bool:
    """True si alguna clase supera `threshold` del total."""
    _, counts = np.unique(y, return_counts=True)
    return (counts / len(y)).max() > threshold
