"""
model_baseline.py — XGBoost con walk-forward validation.

V2: fine_tune_xgboost ahora incluye min_child_weight en param_grid.

Puede correrse de forma independiente:
  python model_baseline.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb

from feature_engineering import get_feature_cols

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"

XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "random_state": 42,
}
EARLY_STOPPING_ROUNDS = 30


# --- Split temporal ---

def temporal_split(df: pd.DataFrame, train_ratio: float = 0.8):
    """
    Split temporal: train = primeras train_ratio filas, test = resto.
    Nunca aleatorio.
    """
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    date_col = "datetime" if "datetime" in df.columns else "date"
    if date_col in df.columns:
        print(f"\n[split] Train: {len(train)} filas | "
              f"{train[date_col].min()} -> {train[date_col].max()}")
        print(f"[split] Test:  {len(test)} filas | "
              f"{test[date_col].min()} -> {test[date_col].max()}")
    else:
        print(f"\n[split] Train: {len(train)} | Test: {len(test)}")
    return train, test


# --- Entrenamiento XGBoost ---

def train_xgboost(df: pd.DataFrame, target_col: str = "target") -> tuple:
    """
    Entrena XGBoost sobre el split temporal 80/20.

    Retorna:
      model, scaler, imputer, feature_cols, df_importance, (X_test_scaled, y_test, y_proba)
    """
    print("\n[XGBoost] Entrenando modelo baseline...")

    feature_cols = get_feature_cols(df)
    train, test = temporal_split(df)

    X_train = train[feature_cols].values
    y_train = train[target_col].values
    X_test = test[feature_cols].values
    y_test = test[target_col].values

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    eval_set = [(X_train_scaled, y_train), (X_test_scaled, y_test)]

    model = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
    model.fit(
        X_train_scaled, y_train,
        eval_set=eval_set,
        verbose=False,
    )

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = float("nan")
    print(f"  [XGBoost] Test Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")

    n_imp = len(model.feature_importances_)
    imp_names = feature_cols[:n_imp] if len(feature_cols) >= n_imp else (
        feature_cols + [f"feat_{i}" for i in range(len(feature_cols), n_imp)]
    )
    importance = pd.DataFrame({
        "feature": imp_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    print("\n  [XGBoost] Top 10 features:")
    print(importance.head(10).to_string(index=False))

    return model, scaler, imputer, feature_cols, importance, (X_test_scaled, y_test, y_proba)


# --- Walk-Forward Validation ---

def walk_forward_validation(df: pd.DataFrame, n_folds: int = 5, target_col: str = "target") -> pd.DataFrame:
    """
    Walk-forward validation con n_folds incrementales.
    """
    print(f"\n[WFV] Walk-forward validation con {n_folds} folds...")

    feature_cols = get_feature_cols(df)
    n = len(df)
    fold_size = n // (n_folds + 1)

    results = []
    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = min(test_start + fold_size, n)

        if test_end <= test_start:
            break

        train = df.iloc[:train_end]
        test = df.iloc[test_start:test_end]

        if len(train) < 30 or len(test) < 5:
            print(f"  [WFV] Fold {fold + 1}: insuficientes datos, skip")
            continue

        X_train = train[feature_cols].values
        y_train = train[target_col].values
        X_test = test[feature_cols].values
        y_test = test[target_col].values

        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy="median")
        X_train = imputer.fit_transform(X_train)
        X_test = imputer.transform(X_test)

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = xgb.XGBClassifier(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(
            X_train_s, y_train,
            eval_set=[(X_test_s, y_test)],
            verbose=False,
        )

        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except ValueError:
            auc = np.nan

        date_col = "datetime" if "datetime" in df.columns else "date"
        results.append({
            "fold": fold + 1,
            "train_start": df.iloc[0][date_col] if date_col in df.columns else 0,
            "train_end": df.iloc[train_end - 1][date_col] if date_col in df.columns else train_end,
            "test_start": df.iloc[test_start][date_col] if date_col in df.columns else test_start,
            "test_end": df.iloc[test_end - 1][date_col] if date_col in df.columns else test_end,
            "train_size": len(train),
            "test_size": len(test),
            "accuracy": acc,
            "roc_auc": auc,
        })
        print(f"  [WFV] Fold {fold + 1}: accuracy={acc:.4f} | AUC={auc:.4f} | "
              f"train={len(train)} | test={len(test)}")

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        print(f"\n  [WFV] Media accuracy: {df_results['accuracy'].mean():.4f} +/- "
              f"{df_results['accuracy'].std():.4f}")
        print(f"  [WFV] Media AUC:      {df_results['roc_auc'].mean():.4f} +/- "
              f"{df_results['roc_auc'].std():.4f}")
    return df_results


# --- Fine-Tuning XGBoost ---

def fine_tune_xgboost(df: pd.DataFrame, n_combos: int = 20, target_col: str = "target") -> tuple:
    """
    Grid search sobre combinaciones de hiperparametros XGBoost.
    Usa walk-forward con 3 folds para evaluar cada combo.
    Prueba al menos n_combos combinaciones.

    V2: Incluye min_child_weight en param_grid.

    Retorna:
      best_model, best_params, df_results, scaler, imputer, y_proba, test_final
    """
    import itertools
    print(f"\n[FineTune] Probando {n_combos} combinaciones de hiperparametros...")

    param_grid = {
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [200, 300, 500],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "min_child_weight": [1, 3, 5],
    }

    import random
    random.seed(42)
    keys = list(param_grid.keys())
    all_combos = list(itertools.product(*param_grid.values()))
    random.shuffle(all_combos)
    combos = all_combos[:n_combos]

    feature_cols = get_feature_cols(df)
    n = len(df)
    fold_size = n // 4
    from sklearn.impute import SimpleImputer

    results = []
    best_auc = -1
    best_params = None

    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        fold_aucs = []

        for fold in range(3):
            train_end = fold_size * (fold + 1)
            test_start = train_end
            test_end = min(test_start + fold_size, n)

            if test_end <= test_start or train_end < 20:
                continue

            train = df.iloc[:train_end]
            test = df.iloc[test_start:test_end]

            if len(train) < 20 or len(test) < 5:
                continue

            X_tr = train[feature_cols].values
            y_tr = train[target_col].values
            X_te = test[feature_cols].values
            y_te = test[target_col].values

            imp = SimpleImputer(strategy="median")
            X_tr = imp.fit_transform(X_tr)
            X_te = imp.transform(X_te)

            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_te = sc.transform(X_te)

            try:
                m = xgb.XGBClassifier(
                    **params,
                    eval_metric="logloss",
                    random_state=42,
                    early_stopping_rounds=20,
                )
                m.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)
                y_proba = m.predict_proba(X_te)[:, 1]
                auc = roc_auc_score(y_te, y_proba)
                fold_aucs.append(auc)
            except Exception:
                continue

        if not fold_aucs:
            continue

        mean_auc = float(np.mean(fold_aucs))
        results.append({**params, "mean_auc": mean_auc, "n_folds": len(fold_aucs)})

        print(f"  [{i+1:2d}/{n_combos}] depth={params['max_depth']} "
              f"lr={params['learning_rate']} n_est={params['n_estimators']} "
              f"mcw={params['min_child_weight']} | AUC={mean_auc:.4f}")

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_params = params

    df_results = pd.DataFrame(results).sort_values("mean_auc", ascending=False).reset_index(drop=True)

    print(f"\n[FineTune] Mejor combinacion (AUC={best_auc:.4f}):")
    print(f"  {best_params}")

    train_final, test_final = temporal_split(df)
    feature_cols = get_feature_cols(df)

    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy="median")
    X_tr = imputer.fit_transform(train_final[feature_cols].values)
    X_te = imputer.transform(test_final[feature_cols].values)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    y_tr = train_final[target_col].values
    y_te = test_final[target_col].values

    best_model = xgb.XGBClassifier(
        **best_params,
        eval_metric="logloss",
        random_state=42,
        early_stopping_rounds=30,
    )
    best_model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

    y_pred = best_model.predict(X_te)
    y_proba = best_model.predict_proba(X_te)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    try:
        auc = roc_auc_score(y_te, y_proba)
    except Exception:
        auc = float("nan")
    print(f"  [FineTune] Test final -> Accuracy: {acc:.4f} | ROC-AUC: {auc:.4f}")

    return best_model, best_params, df_results, scaler, imputer, y_proba, test_final


# --- Entry point ---

if __name__ == "__main__":
    features_path = PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        print("Primero corre feature_engineering.py para generar features.parquet")
        exit(1)

    print("=" * 60)
    print("TEST INDEPENDIENTE: model_baseline.py v2")
    print("=" * 60)

    df = pd.read_parquet(features_path)
    print(f"Dataset cargado: {df.shape}")

    model, scaler, imputer, feature_cols, importance, test_data = train_xgboost(df)
    df_wfv = walk_forward_validation(df)

    importance.to_parquet(PROCESSED_DIR / "xgb_importance.parquet", index=False)
    df_wfv.to_parquet(PROCESSED_DIR / "wfv_results.parquet", index=False)
    print("\n[OK] model_baseline.py v2 corrio sin errores.")
