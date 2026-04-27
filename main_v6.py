"""
main_v6.py — Pipeline v6 con meta-labeling de dos modelos.

Cambios vs v5:
  - Feature selection: dos SFS separados con LGBM (binario)
      * SFS-M1: activo vs neutral (todos los eventos)
      * SFS-M2: sube vs baja (solo eventos activos)
  - Meta-labeling:
      * M1 predice si hay senial (activo/neutral)
      * M2 predice la direccion (sube/baja) dado que hay senial
  - Tres pares de modelos: LGBM, XGB, MLP (mismo algoritmo para M1 y M2)
  - 33 features candidatas (10 originales + 12 de v6 + 11 clasicos)

Uso:
  python main_v6.py
"""

from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

BASE_DIR = Path(__file__).parent


def banner(t: str):
    print(f"\n{'=' * 70}\n  {t}\n{'=' * 70}")


def step(n: int, t: str):
    print(f"\n{'-' * 70}\n  PASO {n}: {t}\n{'-' * 70}")


# ---------------------------------------------------------------------------
# Paso 1: dataset
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    from process_data_v5 import build_dataset_4h
    return build_dataset_4h()


# ---------------------------------------------------------------------------
# Paso 2: labels Triple Barrier
# ---------------------------------------------------------------------------

def get_labels(df_raw: pd.DataFrame) -> pd.DataFrame:
    from triple_barrier_v4 import build_labels_triple_barrier
    from meta_labeling_v6 import SFS_CANDIDATE_FEATURES

    df_idx = df_raw.copy()
    df_idx["datetime"] = pd.to_datetime(df_idx["datetime"])
    df_idx = df_idx.set_index("datetime").sort_index()

    # Señal primaria EMA9/EMA21 para side-based TBM
    side_series = df_idx["ema_signal"] if "ema_signal" in df_idx.columns else None
    if side_series is not None:
        print(f"  [labels] EMA side: long={int((side_series > 0).sum())} "
              f"| short={int((side_series < 0).sum())}")

    events = build_labels_triple_barrier(df_idx, freq="4h", side_series=side_series)
    if events.empty:
        raise RuntimeError("Triple Barrier no produjo eventos.")

    labels  = events["bin"].astype(int).rename("label")
    cols    = [c for c in SFS_CANDIDATE_FEATURES if c in df_idx.columns]

    # Incluir ema_signal como columna extra (no es feature ML, es señal direccional)
    extra = ["close"]
    if "ema_signal" in df_idx.columns:
        extra.append("ema_signal")
    joined  = df_idx[cols + extra].join(labels, how="inner").dropna(subset=["label"])

    joined = joined.reset_index()
    if "index" in joined.columns:
        joined = joined.rename(columns={"index": "datetime"})
    joined["datetime"] = pd.to_datetime(joined["datetime"])

    dist = dict(joined["label"].value_counts().sort_index())
    print(f"  [labels] Eventos: {len(joined)} | dist: {dist}")
    return joined


# ---------------------------------------------------------------------------
# Paso 3: split temporal
# ---------------------------------------------------------------------------

def temporal_split(df: pd.DataFrame, ratio: float = 0.8):
    idx   = int(len(df) * ratio)
    train = df.iloc[:idx].copy().reset_index(drop=True)
    test  = df.iloc[idx:].copy().reset_index(drop=True)
    print(f"  [split] train={len(train)} | test={len(test)}")
    return train, test


# ---------------------------------------------------------------------------
# Paso 4: SFS doble
# ---------------------------------------------------------------------------

def run_sfs(df_joined: pd.DataFrame) -> tuple[dict, dict]:
    from meta_labeling_v6 import (
        SFS_CANDIDATE_FEATURES, run_sfs_m1, run_sfs_m2, plot_sfs_curves,
    )
    feat_df = df_joined[[c for c in SFS_CANDIDATE_FEATURES
                          if c in df_joined.columns]].reset_index(drop=True)
    labels  = df_joined["label"].reset_index(drop=True)

    sfs1 = run_sfs_m1(feat_df, labels)
    sfs2 = run_sfs_m2(feat_df, labels)

    plot_sfs_curves(sfs1, sfs2, save_path=str(BASE_DIR / "sfs_meta_curves.png"))
    return sfs1, sfs2


# ---------------------------------------------------------------------------
# Paso 5: entrenar y simular pares de modelos
# ---------------------------------------------------------------------------

def train_and_simulate(
    df_train: pd.DataFrame,
    df_test:  pd.DataFrame,
    features_m1: list[str],
    features_m2: list[str],
) -> dict:
    from meta_labeling_v6 import MetaPipeline, make_meta_labels
    from simulate_v4 import (simulate_strategy_v4, simulate_buy_hold,
                              compute_financial_stats, classification_metrics)

    y_train = df_train["label"].values.astype(int)
    y_test  = df_test["label"].values.astype(int)

    # Usar ultimos 20% de train como validacion interna para early stopping
    val_start = int(len(df_train) * 0.8)
    df_tr  = df_train.iloc[:val_start]
    df_val = df_train.iloc[val_start:]
    y_tr   = y_train[:val_start]
    y_val  = y_train[val_start:]

    results: dict = {}
    errors:  list = []

    # Señal EMA del test set para side-aware predict_proba
    side_test = df_test["ema_signal"].values if "ema_signal" in df_test.columns else None

    # Filtro de regimen de volatilidad — threshold calculado sobre train (sin lookahead)
    VOL_COL = "log_return_roll_std72"
    if VOL_COL in df_train.columns and VOL_COL in df_test.columns:
        vol_threshold = float(df_train[VOL_COL].median())
        vol_mask_test = (df_test[VOL_COL] >= vol_threshold).values
        n_flat = int((~vol_mask_test).sum())
        print(f"\n  [vol-filter] threshold={vol_threshold:.5f} | "
              f"FLAT forzado en {n_flat}/{len(df_test)} barras "
              f"({n_flat/len(df_test)*100:.1f}% baja vol)")
    else:
        vol_mask_test = None
        print("\n  [vol-filter] columna vol no disponible, filtro desactivado")

    for model_type in ["lgbm", "xgb", "mlp"]:
        name = model_type.upper()
        try:
            print(f"\n  [train] {name} M1+M2...")
            pipeline = MetaPipeline(
                model_type=model_type,
                features_m1=features_m1,
                features_m2=features_m2,
            )
            pipeline.fit(df_tr, y_tr, df_val, y_val)

            proba = pipeline.predict_proba(df_test, side=side_test)

            # Metricas adicionales de cada sub-modelo
            meta_test, dir_test = make_meta_labels(y_test)
            p_active = pipeline.m1.predict_proba(
                pipeline.scaler_m1.transform(
                    pipeline.imp_m1.transform(
                        df_test[features_m1].values)))[:, 1]
            pred_active = (p_active >= 0.5).astype(int)
            f1_m1 = f1_score(meta_test, pred_active, average="macro", zero_division=0)

            # --- Sin filtro ---
            cls = classification_metrics(y_test, proba)
            trades, equity = simulate_strategy_v4(df_test, proba,
                                                   model_kind=model_type)
            fin = compute_financial_stats(trades, equity, freq="4h")
            results[name] = {
                "classification_metrics": cls,
                "financial_stats":        fin,
                "f1_m1":                  f1_m1,
            }

            # --- Con filtro de vol ---
            if vol_mask_test is not None:
                proba_f = proba.copy()
                proba_f[~vol_mask_test] = [0.0, 1.0, 0.0]
                cls_f = classification_metrics(y_test, proba_f)
                trades_f, equity_f = simulate_strategy_v4(df_test, proba_f,
                                                           model_kind=model_type)
                fin_f = compute_financial_stats(trades_f, equity_f, freq="4h")
                results[name + "_F"] = {
                    "classification_metrics": cls_f,
                    "financial_stats":        fin_f,
                    "f1_m1":                  f1_m1,
                }
                print(f"    M1 f1_macro={f1_m1:.4f} | "
                      f"P&L=${fin.get('net_pnl_usd', 0):+,.2f} (sin filtro) | "
                      f"P&L=${fin_f.get('net_pnl_usd', 0):+,.2f} (con filtro)")
            else:
                print(f"    M1 f1_macro={f1_m1:.4f} | "
                      f"P&L=${fin.get('net_pnl_usd', 0):+,.2f}")

        except Exception as e:
            errors.append(f"{name}: {e}")
            traceback.print_exc()

    # Buy & Hold
    try:
        trades_bh, equity_bh = simulate_buy_hold(df_test)
        fin_bh = compute_financial_stats(trades_bh, equity_bh, freq="4h")
        results["BH"] = {"financial_stats": fin_bh, "classification_metrics": None}
    except Exception as e:
        errors.append(f"BH: {e}")

    if errors:
        print("\n  [WARN] Errores no fatales:")
        for e in errors:
            print(f"    * {e}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    banner("PIPELINE ML ON-CHAIN v6 — META-LABELING (2 modelos)")

    step(1, "CONSTRUYENDO DATASET 4h")
    df_raw = build_dataset()

    step(2, "TRIPLE BARRIER — LABELS")
    df_joined = get_labels(df_raw)

    step(3, "SPLIT TEMPORAL 80/20")
    train, test = temporal_split(df_joined)

    step(4, "SFS DOBLE — M1 (activo/neutral) + M2 (sube/baja)")
    sfs_m1, sfs_m2 = run_sfs(df_joined)
    features_m1 = sfs_m1["best_features"]
    features_m2 = sfs_m2["best_features"]

    print(f"\n  M1 features ({len(features_m1)}): {features_m1}")
    print(f"  M2 features ({len(features_m2)}): {features_m2}")

    step(5, "ENTRENAMIENTO + SIMULACION — LGBM / XGB / MLP (pares M1+M2)")
    results = train_and_simulate(train, test, features_m1, features_m2)

    # ---------------------------------------------------------------------------
    # Resultados
    # ---------------------------------------------------------------------------
    banner("RESULTADOS — Meta-Labeling v6")

    print(f"\n  M1 k={sfs_m1['best_k']} features: {features_m1}")
    print(f"  M2 k={sfs_m2['best_k']} features: {features_m2}\n")

    print(f"  {'Modelo':<8} {'Net P&L':>12}  {'Sharpe':>7}  {'MaxDD%':>7}  "
          f"{'Acc':>6}  {'F1-mac':>7}  {'AUC-OvR':>8}  {'F1-M1':>7}  {'Trades':>7}  {'%Flat':>6}")
    print("  " + "-" * 96)

    for name in ["LGBM", "LGBM_F", "XGB", "XGB_F", "MLP", "MLP_F", "BH"]:
        if name not in results:
            continue
        d    = results[name]
        fin  = d.get("financial_stats") or {}
        cls  = d.get("classification_metrics") or {}
        acc  = cls.get("accuracy",    float("nan"))
        f1m  = cls.get("f1_macro",    float("nan"))
        auc  = cls.get("roc_auc_ovr", float("nan"))
        f1m1 = d.get("f1_m1",         float("nan"))
        pnl  = fin.get("net_pnl_usd",   0)
        shr  = fin.get("sharpe_ratio",  0)
        dd   = fin.get("max_dd_pct",    0)
        tr   = fin.get("total_trades",  0)
        fl   = fin.get("pct_flat",      0)
        print(f"  {name:<8} ${pnl:>+11,.2f}  {shr:>7.3f}  {dd:>6.2f}%  "
              f"{acc:>6.3f}  {f1m:>7.3f}  {auc:>8.3f}  {f1m1:>7.3f}  {tr:>7}  {fl:>5.1f}%")

    print("\n  Curvas SFS guardadas en: sfs_meta_curves.png")


if __name__ == "__main__":
    main()
