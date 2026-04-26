"""
main_v5.py -- Orquestador del pipeline v5.

Cambios vs v4:
  1. Solo frecuencia 4h (daily eliminado)
  2. Datos 100% horarios — sin leakage
  3. Fuentes: Binance 1h, The Graph 1h, Dune transfers 1h, Dollar bar duration
  4. Features nuevas: transfer_count, whale_volume_ratio, unique_senders,
     transfer_count_pct_change, dollar_bar_duration
  5. Features eliminadas: whale_balance_pct, whale_delta_pct, herfindahl_index,
     holders_growth, fear_greed_value (todas diarias -> leakage en 4h)

Uso:
  python3 main_v5.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

BASE_DIR      = Path(__file__).parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

errors_log: list[str] = []


def banner(t: str):
    print(f"\n{'=' * 70}\n  {t}\n{'=' * 70}")


def step(n: int, t: str):
    print(f"\n{'-' * 70}\n  PASO {n}: {t}\n{'-' * 70}")


# ---------------------------------------------------------------------------
# Paso 1: construir dataset 4h
# ---------------------------------------------------------------------------

def load_or_build_dataset() -> pd.DataFrame:
    from process_data_v5 import build_dataset_4h
    return build_dataset_4h()


# ---------------------------------------------------------------------------
# Paso 2-3: seleccion de features + Triple Barrier
# ---------------------------------------------------------------------------

def prepare_dataset(df_raw: pd.DataFrame) -> dict:
    from feature_selection_v5 import select_features
    from triple_barrier_v4 import build_labels_triple_barrier, get_daily_vol

    selection = select_features(df_raw)

    df_idx = df_raw.copy()
    df_idx["datetime"] = pd.to_datetime(df_idx["datetime"])
    df_idx = df_idx.set_index("datetime").sort_index()

    events = build_labels_triple_barrier(df_idx, freq="4h")
    if events.empty:
        raise RuntimeError("Triple Barrier no produjo eventos para freq=4h")

    final_feats = selection["final_features"]
    df_cand = selection["df_candidates"].copy()
    df_cand.index = df_idx.index

    labels = events["bin"].astype(int).rename("label")
    joined = df_cand[final_feats].join(labels, how="inner")
    joined["close"] = df_idx["close"].reindex(joined.index)

    joined = joined.dropna(subset=final_feats + ["close", "label"]).copy()
    joined = joined.reset_index()
    if "index" in joined.columns:
        joined = joined.rename(columns={"index": "datetime"})
    joined["datetime"] = pd.to_datetime(joined["datetime"])

    y_labels    = joined["label"].values.astype(int)
    close_s     = df_idx["close"]
    vol_s       = get_daily_vol(close_s, span=20)

    return {
        "dataset":        joined,
        "final_features": final_feats,
        "selection":      selection,
        "events_merged":  events,
        "labels_y":       y_labels,
        "close_series":   close_s,
        "vol_series":     vol_s,
    }


# ---------------------------------------------------------------------------
# Paso 4: split temporal
# ---------------------------------------------------------------------------

def temporal_split(df: pd.DataFrame, ratio: float = 0.8):
    idx   = int(len(df) * ratio)
    train = df.iloc[:idx].reset_index(drop=True)
    test  = df.iloc[idx:].reset_index(drop=True)
    print(f"  [split] train={len(train)} | test={len(test)}")
    return train, test


# ---------------------------------------------------------------------------
# Paso 5: walk-forward validation
# ---------------------------------------------------------------------------

def walk_forward_xgb(df_train: pd.DataFrame, feature_cols: list,
                     n_folds: int = 5) -> pd.DataFrame:
    from models_v4 import XGBWrapper, compute_sample_weights, is_severely_imbalanced

    n         = len(df_train)
    fold_size = n // (n_folds + 1)
    rows      = []

    for fold in range(n_folds):
        tr_end   = fold_size * (fold + 1)
        te_start = tr_end
        te_end   = min(te_start + fold_size, n)
        if te_end - te_start < 10 or tr_end < 30:
            continue
        tr = df_train.iloc[:tr_end]
        te = df_train.iloc[te_start:te_end]

        X_tr = tr[feature_cols].values
        y_tr = tr["label"].values.astype(int)
        X_te = te[feature_cols].values
        y_te = te["label"].values.astype(int)

        imp  = SimpleImputer(strategy="median")
        X_tr = imp.fit_transform(X_tr); X_te = imp.transform(X_te)
        sc   = StandardScaler()
        X_tr = sc.fit_transform(X_tr);  X_te = sc.transform(X_te)

        sw = compute_sample_weights(y_tr) if is_severely_imbalanced(y_tr) else None
        try:
            m      = XGBWrapper().fit(X_tr, y_tr, X_te, y_te,
                                      feature_names=feature_cols, sample_weight=sw)
            proba  = m.predict_proba(X_te)
            y_pred = np.array([{0: -1, 1: 0, 2: 1}[v] for v in np.argmax(proba, axis=1)])
            acc    = accuracy_score(y_te, y_pred)
            f1m    = f1_score(y_te, y_pred, labels=[-1, 0, 1],
                              average="macro", zero_division=0)
        except Exception as e:
            errors_log.append(f"WFV fold {fold+1}: {e}")
            continue

        rows.append({"fold": fold + 1, "model": "XGB",
                     "train_size": len(tr), "test_size": len(te),
                     "accuracy": acc, "f1_macro": f1m})
        print(f"    [WFV] fold {fold+1}: acc={acc:.4f} | f1m={f1m:.4f} | "
              f"tr={len(tr)} te={len(te)}")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paso 6-7: entrenar modelos y simular
# ---------------------------------------------------------------------------

def train_and_simulate(df_train: pd.DataFrame, df_test: pd.DataFrame,
                       feature_cols: list) -> dict:
    from models_v4 import (XGBWrapper, LGBMWrapper, MLPWrapper,
                            compute_sample_weights, is_severely_imbalanced)
    from simulate_v4 import (simulate_strategy_v4, simulate_buy_hold,
                              compute_financial_stats, classification_metrics)

    X_tr_raw = df_train[feature_cols].values
    y_tr     = df_train["label"].values.astype(int)
    X_te_raw = df_test[feature_cols].values
    y_te     = df_test["label"].values.astype(int)

    imp    = SimpleImputer(strategy="median")
    X_tr   = imp.fit_transform(X_tr_raw)
    X_te   = imp.transform(X_te_raw)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    imbalanced = is_severely_imbalanced(y_tr)
    sw         = compute_sample_weights(y_tr) if imbalanced else None
    if imbalanced:
        print("  [4h] desbalance severo. Aplicando class_weight='balanced'.")
        errors_log.append("4h: clase dominante >60%, usando class_weight balanced")

    out: dict = {}

    for ModelClass, name in [(XGBWrapper, "XGB"), (LGBMWrapper, "LGBM"), (MLPWrapper, "MLP")]:
        try:
            print(f"\n  [train] {name}-4h...")
            model  = ModelClass().fit(X_tr_s, y_tr, X_te_s, y_te,
                                      feature_names=feature_cols, sample_weight=sw)
            proba  = model.predict_proba(X_te_s)
            cls    = classification_metrics(y_te, proba)
            trades, equity = simulate_strategy_v4(
                df_test, proba, model_kind=name.lower())
            fin    = compute_financial_stats(trades, equity, freq="4h")

            entry = {
                "model":                  model,
                "trades_df":              trades,
                "equity_df":              equity,
                "classification_metrics": cls,
                "financial_stats":        fin,
                "feature_importance":     model.feature_importance() if name != "MLP"
                                          else model.feature_importance(X_te_s, y_te),
                "feature_names":          feature_cols,
            }
            if name == "MLP":
                entry["mlp_history"] = model.history
            else:
                entry["best_iteration"] = model.best_iteration_
                if model.best_iteration_ is not None and model.best_iteration_ < 50:
                    errors_log.append(
                        f"{name}-4h: early stopping en iter {model.best_iteration_} (<50)")

            out[f"{name}-4h"] = entry
        except Exception as e:
            errors_log.append(f"{name}-4h: {e}")
            traceback.print_exc()

    # Buy & Hold
    try:
        trades_bh, equity_bh = simulate_buy_hold(df_test)
        fin_bh = compute_financial_stats(trades_bh, equity_bh, freq="4h")
        out["BH-4h"] = {
            "trades_df": trades_bh, "equity_df": equity_bh,
            "financial_stats": fin_bh, "classification_metrics": None,
        }
    except Exception as e:
        errors_log.append(f"BH-4h: {e}")

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    banner("PIPELINE ML ON-CHAIN v5 - UNI (Uniswap)")
    print("  Frecuencia:  4h (unica)")
    print("  Modelos:     XGBoost, LightGBM, MLP (16x3)")
    print("  Labeling:    Triple Barrier Method (Lopez de Prado)")
    print("  Features:    15 candidatas | Sin leakage | Fuentes 100% horarias")
    print("  Nuevas:      transfer_count, whale_volume_ratio,")
    print("               transfer_count_pct_change, dollar_bar_duration")
    print("  Capital:     $10,000 | Trade size: $500 | Fee 0.1% por lado")

    step(1, "CONSTRUYENDO DATASET 4h (merge horario + resample)")
    df_raw = load_or_build_dataset()

    step(2, "SELECCION DE FEATURES + TRIPLE BARRIER (4h)")
    prep       = prepare_dataset(df_raw)
    selection  = prep["selection"]
    df         = prep["dataset"]
    feats      = prep["final_features"]
    print(f"  [4h] dataset final: shape={df.shape} | features={len(feats)}: {feats}")

    tb_data = {
        "4h": {
            "labels_y":      prep["labels_y"],
            "events":        prep["events_merged"],
            "events_merged": prep["events_merged"],
            "close_series":  prep["close_series"],
            "vol_series":    prep["vol_series"],
        }
    }

    step(3, "SPLIT TEMPORAL 80/20 (4h)")
    train, test = temporal_split(df, ratio=0.8)

    step(4, "WALK-FORWARD VALIDATION — XGB (4h)")
    wfv_results = {"4h": walk_forward_xgb(train, feats, n_folds=5)}

    step(5, "ENTRENAMIENTO + SIMULACION (4h)")
    all_results = train_and_simulate(train, test, feats)

    step(6, "GENERANDO REPORTE PDF")
    try:
        from generate_report_v5 import generate_report_v5
        generate_report_v5(
            feature_selection=selection,
            tb_results=tb_data,
            all_results=all_results,
            wfv_results=wfv_results,
            errors_log=errors_log,
        )
    except Exception as e:
        print(f"  [WARN] Falla generando PDF: {e}")
        errors_log.append(f"PDF: {e}")
        traceback.print_exc()

    banner("PIPELINE V5 COMPLETO")

    print("\nTABLA RESUMEN:")
    print("-" * 90)
    print(f"  {'Model':<12} {'Net P&L':<14} {'Sharpe':<8} {'MaxDD%':<9} "
          f"{'Acc':<7} {'F1m':<7} {'Trades':<8} {'%Flat'}")
    print("-" * 90)

    ranked = sorted(
        [(k, d) for k, d in all_results.items() if not k.startswith("BH")],
        key=lambda x: x[1].get("financial_stats", {}).get("net_pnl_usd", 0),
        reverse=True,
    )
    for k, d in ranked:
        st  = d.get("financial_stats", {}) or {}
        cls = d.get("classification_metrics", {}) or {}
        acc = cls.get("accuracy", float("nan"))
        f1m = cls.get("f1_macro", float("nan"))
        print(f"  {k:<12} ${st.get('net_pnl_usd', 0):<+13,.2f} "
              f"{st.get('sharpe_ratio', 0):<8.3f} {st.get('max_dd_pct', 0):<8.2f}% "
              f"{acc:<7.3f} {f1m:<7.3f} {st.get('total_trades', 0):<8} "
              f"{st.get('pct_flat', 0):.1f}%")

    bh = all_results.get("BH-4h")
    if bh:
        st = bh["financial_stats"]
        print(f"  {'BH-4h':<12} ${st['net_pnl_usd']:<+13,.2f} "
              f"{st['sharpe_ratio']:<8.3f} {st['max_dd_pct']:<8.2f}% "
              f"{'-':<7} {'-':<7} {st['total_trades']:<8} {st['pct_flat']:.1f}%")
    print("-" * 90)

    if errors_log:
        print(f"\nErrores no fatales: {len(errors_log)}")
        for e in errors_log[:5]:
            print(f"  * {str(e)[:120]}")

    print("\nPIPELINE V5 COMPLETO. Reporte en reporte_uni_ml_v5.pdf")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[Pipeline] Interrumpido por el usuario.")
        sys.exit(0)
