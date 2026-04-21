"""
main_v3.py -- Orquestador del pipeline v3.

V3: Decision layer LONG/SHORT/FLAT con capital real USD.
    Cambios exclusivamente en simulacion -- reutiliza modelos v2.

Frecuencias: 4h, daily (se descarta 1h)
Variantes:
  A -- XGBoost umbral fijo
  B -- XGBoost umbral dinamico Fear & Greed
  C -- LSTM umbral fijo asimetrico
  BH -- Buy & Hold benchmark

Total: 3 variantes x 2 frecuencias + 1 BH = 7 runs

Uso:
  python3 main_v3.py
  python3 main_v3.py --skip-xgb-retrain   # usa XGB probas cacheadas si existen
  python3 main_v3.py --no-cache            # fuerza reentrenamiento de XGB
"""

import argparse
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR       = Path(__file__).parent
PROCESSED_DIR  = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_MAP = {
    "4h":    "target_4h",
    "daily": "target_24h",
}
LOOKBACK_MAP = {
    "4h":    6,
    "daily": 7,
}
FREQS = ["4h", "daily"]

errors_log = []


def banner(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def step(n, name: str):
    print(f"\n{'-' * 60}")
    print(f"  PASO {n}: {name}")
    print(f"{'-' * 60}")


# ---------------------------------------------------------------------------
# PASO 1: Cargar features (ya generadas por v2)
# ---------------------------------------------------------------------------

def load_features(freq: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"features_{freq}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"features_{freq}.parquet no encontrado. "
            f"Ejecuta main.py (v2) primero para generar los datos."
        )
    df = pd.read_parquet(path)
    print(f"  [Features] {freq}: {df.shape} cargado desde {path.name}")
    return df


# ---------------------------------------------------------------------------
# PASO 2: Obtener probabilidades XGBoost
# ---------------------------------------------------------------------------

def get_xgb_probas(df_feat: pd.DataFrame, freq: str, force_retrain: bool = False) -> tuple:
    """
    Intenta cargar probas cacheadas. Si no existen o force_retrain=True,
    reentrena con los mejores parametros de la tuning v2 (si disponible),
    o con parametros base.

    Retorna: (y_proba, df_test)
    """
    proba_path  = PROCESSED_DIR / f"xgb_proba_v3_{freq}.npy"
    dftest_path = PROCESSED_DIR / f"xgb_dftest_v3_{freq}.parquet"

    if not force_retrain and proba_path.exists() and dftest_path.exists():
        y_proba = np.load(str(proba_path))
        df_test = pd.read_parquet(str(dftest_path))
        print(f"  [XGB] {freq}: probas cargadas desde cache ({len(y_proba)} muestras)")
        return y_proba, df_test

    print(f"  [XGB] {freq}: entrenando modelo...")

    sys.path.insert(0, str(BASE_DIR))
    from model_baseline import train_xgboost, temporal_split
    from feature_engineering import get_feature_cols
    import xgboost as xgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import accuracy_score, roc_auc_score

    target_col = TARGET_MAP[freq]

    # Intentar usar params del tuning v2
    tuning_path = PROCESSED_DIR / f"xgb_tuning_{freq}.parquet"
    if tuning_path.exists():
        try:
            df_tuning = pd.read_parquet(str(tuning_path))
            best_row = df_tuning.iloc[0]
            int_keys = {"max_depth", "n_estimators", "min_child_weight"}
            best_params = {}
            for k in ["max_depth", "learning_rate", "n_estimators",
                      "subsample", "colsample_bytree", "min_child_weight"]:
                if k in best_row:
                    best_params[k] = int(best_row[k]) if k in int_keys else float(best_row[k])

            print(f"  [XGB] {freq}: usando best params de tuning v2: {best_params}")

            feature_cols = get_feature_cols(df_feat)
            train, test = temporal_split(df_feat)

            imputer = SimpleImputer(strategy="median")
            X_tr = imputer.fit_transform(train[feature_cols].values)
            X_te = imputer.transform(test[feature_cols].values)
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            y_tr = train[target_col].values
            y_te = test[target_col].values

            model = xgb.XGBClassifier(
                **best_params,
                eval_metric="logloss",
                random_state=42,
                early_stopping_rounds=30,
            )
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], verbose=False)

            y_proba = model.predict_proba(X_te)[:, 1]
            y_pred  = (y_proba >= 0.5).astype(int)

            try:
                auc = roc_auc_score(y_te, y_proba)
            except Exception:
                auc = float("nan")
            acc = accuracy_score(y_te, y_pred)
            print(f"  [XGB] {freq}: Accuracy={acc:.4f} | AUC={auc:.4f}")

            df_test = test.copy().reset_index(drop=True)
            df_test = df_test.iloc[:len(y_proba)].copy()

        except Exception as e:
            print(f"  [WARN] {freq}: fallo usar tuning params: {e}. Usando baseline.")
            errors_log.append(f"XGB tuned params {freq} fallo: {e}")
            _, _, _, _, _, test_data = train_xgboost(df_feat, target_col=target_col)
            _, y_te, y_proba = test_data
            split_idx = int(len(df_feat) * 0.8)
            df_test = df_feat.iloc[split_idx:].copy().reset_index(drop=True)
            df_test = df_test.iloc[:len(y_proba)].copy()
    else:
        # Fallback: baseline
        print(f"  [XGB] {freq}: no hay tuning v2, usando baseline")
        _, _, _, _, _, test_data = train_xgboost(df_feat, target_col=target_col)
        _, y_te, y_proba = test_data
        split_idx = int(len(df_feat) * 0.8)
        df_test = df_feat.iloc[split_idx:].copy().reset_index(drop=True)
        df_test = df_test.iloc[:len(y_proba)].copy()

    # Completar columnas necesarias
    if "pct_return" not in df_test.columns:
        df_test["pct_return"] = df_test["close"].pct_change().fillna(0)

    # Guardar cache
    np.save(str(proba_path), y_proba)
    df_test.to_parquet(str(dftest_path), index=False)
    print(f"  [XGB] {freq}: probas guardadas ({len(y_proba)} muestras)")

    return y_proba, df_test


# ---------------------------------------------------------------------------
# PASO 3: Cargar probabilidades LSTM
# ---------------------------------------------------------------------------

def load_lstm_probas(df_feat: pd.DataFrame, freq: str) -> tuple:
    """
    Carga las probas LSTM guardadas por v2.
    Reconstruye df_test_lstm con el mismo slice usado en el entrenamiento.

    Retorna: (y_proba_lstm, df_test_lstm) o (None, None) si no disponible.
    """
    proba_path = PROCESSED_DIR / f"lstm_proba_{freq}.npy"
    if not proba_path.exists():
        print(f"  [LSTM] {freq}: lstm_proba_{freq}.npy no encontrado, omitiendo LSTM")
        errors_log.append(f"LSTM proba {freq} no encontrado")
        return None, None

    y_proba_lstm = np.load(str(proba_path))
    lookback = LOOKBACK_MAP[freq]
    split_idx = int(len(df_feat) * 0.8)

    df_test_lstm = df_feat.iloc[split_idx + lookback:].copy().reset_index(drop=True)
    df_test_lstm = df_test_lstm.iloc[:len(y_proba_lstm)].copy()

    if "pct_return" not in df_test_lstm.columns:
        df_test_lstm["pct_return"] = df_test_lstm["close"].pct_change().fillna(0)

    # Reemplazar NaN en probas con 0.5 (FLAT por defecto)
    nan_mask = np.isnan(y_proba_lstm)
    if nan_mask.sum() > 0:
        print(f"  [LSTM] {freq}: {nan_mask.sum()} NaN en probas -> reemplazados con 0.5 (FLAT)")
        y_proba_lstm = np.where(nan_mask, 0.5, y_proba_lstm)

    print(f"  [LSTM] {freq}: {len(y_proba_lstm)} muestras cargadas")
    return y_proba_lstm, df_test_lstm


# ---------------------------------------------------------------------------
# PASO 4: Correr simulaciones
# ---------------------------------------------------------------------------

def run_all_simulations(
    xgb_data: dict,   # {freq: (y_proba, df_test)}
    lstm_data: dict,  # {freq: (y_proba, df_test)}
) -> dict:
    """
    Corre las 7 simulaciones: A/B/C para 4h y daily + Buy & Hold.
    Retorna all_results con estructura compatible con generate_report_v3.
    """
    from simulate_v3 import simulate_strategy_v3, simulate_buy_hold, compute_stats

    all_results = {}
    INITIAL_CAPITAL = 10_000.0
    TRADE_SIZE = 500.0
    FEE_RATE = 0.001

    # Variantes A y B: XGBoost
    for freq in FREQS:
        if freq not in xgb_data or xgb_data[freq][0] is None:
            print(f"  [SIM] Saltando XGB {freq}: sin probas")
            continue

        y_proba_xgb, df_test_xgb = xgb_data[freq]

        for variant in ["A", "B"]:
            key = f"{variant}-{freq}"
            print(f"\n  [SIM] Corriendo variante {key}...")
            try:
                trades_df, equity_df = simulate_strategy_v3(
                    df_test=df_test_xgb,
                    y_proba=y_proba_xgb,
                    variant=variant,
                    initial_capital=INITIAL_CAPITAL,
                    trade_size=TRADE_SIZE,
                    fee_rate=FEE_RATE,
                )
                stats = compute_stats(trades_df, equity_df,
                                      initial_capital=INITIAL_CAPITAL, freq=freq)
                print(f"  [SIM] {key}: trades={stats['total_trades']} | "
                      f"net_pnl=${stats['net_pnl_usd']:+,.2f} | "
                      f"sharpe={stats['sharpe_ratio']:.3f}")
                all_results[key] = {
                    "trades_df": trades_df,
                    "equity_df": equity_df,
                    "stats":     stats,
                }
                # Documentar si 0 trades (modelo no cruza umbrales)
                if stats["total_trades"] == 0:
                    p = y_proba_xgb
                    errors_log.append(
                        f"{key}: 0 trades. El modelo XGB no cruza los umbrales. "
                        f"Probas: mean={p.mean():.4f} std={p.std():.4f} "
                        f"max={p.max():.4f} min={p.min():.4f}. "
                        f"Todos los valores caen en la zona FLAT."
                    )

                # Guardar resultados
                trades_df.to_parquet(PROCESSED_DIR / f"trades_v3_{key}.parquet", index=False)
                equity_df.to_parquet(PROCESSED_DIR / f"equity_v3_{key}.parquet", index=False)

            except Exception as e:
                msg = f"Simulacion {key} fallo: {e}"
                print(f"  [WARN] {msg}")
                errors_log.append(msg)
                traceback.print_exc()

    # Variante C: LSTM
    for freq in FREQS:
        if freq not in lstm_data or lstm_data[freq][0] is None:
            print(f"  [SIM] Saltando LSTM {freq}: sin probas")
            continue

        y_proba_lstm, df_test_lstm = lstm_data[freq]
        key = f"C-{freq}"
        print(f"\n  [SIM] Corriendo variante {key}...")
        try:
            trades_df, equity_df = simulate_strategy_v3(
                df_test=df_test_lstm,
                y_proba=y_proba_lstm,
                variant="C",
                initial_capital=INITIAL_CAPITAL,
                trade_size=TRADE_SIZE,
                fee_rate=FEE_RATE,
            )
            stats = compute_stats(trades_df, equity_df,
                                  initial_capital=INITIAL_CAPITAL, freq=freq)
            print(f"  [SIM] {key}: trades={stats['total_trades']} | "
                  f"net_pnl=${stats['net_pnl_usd']:+,.2f} | "
                  f"sharpe={stats['sharpe_ratio']:.3f}")
            if stats["total_trades"] == 0:
                p = y_proba_lstm
                errors_log.append(
                    f"{key}: 0 trades. El LSTM produce probas muy cercanas a 0.5. "
                    f"Probas: mean={p.mean():.4f} std={p.std():.4f} "
                    f"max={p.max():.4f} min={p.min():.4f}. "
                    f"Umbrales C: 0.58/0.42. El LSTM no entrego conviccion suficiente."
                )
            all_results[key] = {
                "trades_df": trades_df,
                "equity_df": equity_df,
                "stats":     stats,
            }
            trades_df.to_parquet(PROCESSED_DIR / f"trades_v3_{key}.parquet", index=False)
            equity_df.to_parquet(PROCESSED_DIR / f"equity_v3_{key}.parquet", index=False)

        except Exception as e:
            msg = f"Simulacion {key} fallo: {e}"
            print(f"  [WARN] {msg}")
            errors_log.append(msg)
            traceback.print_exc()

    # Buy & Hold: uno por frecuencia + uno "canonico" (daily) para la tabla maestra
    print("\n  [SIM] Corriendo Buy & Hold benchmarks...")
    bh_canonical = None

    for freq in FREQS:
        if freq not in xgb_data or xgb_data[freq][0] is None:
            continue
        bh_df_test = xgb_data[freq][1]
        try:
            trades_bh, equity_bh = simulate_buy_hold(
                df_test=bh_df_test,
                initial_capital=INITIAL_CAPITAL,
                trade_size=TRADE_SIZE,
                fee_rate=FEE_RATE,
            )
            stats_bh = compute_stats(trades_bh, equity_bh,
                                     initial_capital=INITIAL_CAPITAL, freq=freq)
            print(f"  [SIM] BH-{freq}: net_pnl=${stats_bh['net_pnl_usd']:+,.2f}")

            bh_entry = {"trades_df": trades_bh, "equity_df": equity_bh, "stats": stats_bh}
            all_results[f"BH-{freq}"] = bh_entry

            # Inyectar el B&H correspondiente en cada variante de esta frecuencia
            for variant in ["A", "B", "C"]:
                key = f"{variant}-{freq}"
                if key in all_results:
                    all_results[key]["bh_equity_df"] = equity_bh

            # Guardar el BH daily como canonico para la tabla maestra
            if freq == "daily":
                bh_canonical = bh_entry

        except Exception as e:
            msg = f"Buy & Hold {freq} fallo: {e}"
            print(f"  [WARN] {msg}")
            errors_log.append(msg)
            traceback.print_exc()

    # Entrada "BH" para la tabla maestra (usar daily si disponible, sino 4h)
    if bh_canonical:
        all_results["BH"] = bh_canonical
    elif "BH-4h" in all_results:
        all_results["BH"] = all_results["BH-4h"]

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pipeline ML On-Chain UNI v3")
    parser.add_argument("--no-cache",     action="store_true",
                        help="Forzar reentrenamiento de XGBoost (ignora cache)")
    parser.add_argument("--freq",         default=None,
                        help="Solo esta frecuencia (4h, daily)")
    args = parser.parse_args()

    banner("PIPELINE ML ON-CHAIN v3 - UNI (Uniswap)")
    print(f"  Token:        UNI")
    print(f"  Frecuencias:  4h, daily (se descarta 1h)")
    print(f"  Variantes:    A (XGB fijo), B (XGB F&G), C (LSTM), BH")
    print(f"  Capital:      $10,000 | Trade size: $500 | Fee: 0.1%/turn")
    print(f"  Cache XGB:    {'desactivado (--no-cache)' if args.no_cache else 'activado'}")

    target_freqs = [args.freq] if args.freq and args.freq in FREQS else FREQS

    xgb_data  = {}
    lstm_data = {}

    try:
        # ---- PASO 1: Cargar features ----
        step(1, "CARGANDO FEATURES DE V2")
        all_features = {}
        for freq in target_freqs:
            try:
                all_features[freq] = load_features(freq)
            except FileNotFoundError as e:
                print(f"\n  [ERROR] {e}")
                print("  Ejecuta 'python3 main.py' (pipeline v2) primero.")
                sys.exit(1)

        # ---- PASO 2: Probas XGBoost ----
        step(2, "OBTENIENDO PROBABILIDADES XGBOOST")
        for freq in target_freqs:
            df_feat = all_features[freq]
            try:
                y_proba_xgb, df_test_xgb = get_xgb_probas(
                    df_feat, freq, force_retrain=args.no_cache
                )
                xgb_data[freq] = (y_proba_xgb, df_test_xgb)
            except Exception as e:
                msg = f"XGBoost {freq}: {e}"
                print(f"  [WARN] {msg}")
                errors_log.append(msg)
                traceback.print_exc()
                xgb_data[freq] = (None, None)

        # ---- PASO 2b: Diagnostico de distribuciones de probas XGB ----
        for freq, (y_proba, _) in xgb_data.items():
            if y_proba is None:
                continue
            msg = (f"XGB {freq}: n={len(y_proba)} | mean={y_proba.mean():.4f} | "
                   f"std={y_proba.std():.4f} | "
                   f"range=[{y_proba.min():.4f}, {y_proba.max():.4f}]")
            print(f"  [Diag] {msg}")

        # ---- PASO 3: Probas LSTM ----
        step(3, "CARGANDO PROBABILIDADES LSTM (de V2)")
        for freq in target_freqs:
            df_feat = all_features[freq]
            try:
                y_proba_lstm, df_test_lstm = load_lstm_probas(df_feat, freq)
                lstm_data[freq] = (y_proba_lstm, df_test_lstm)
            except Exception as e:
                msg = f"LSTM {freq}: {e}"
                print(f"  [WARN] {msg}")
                errors_log.append(msg)
                lstm_data[freq] = (None, None)

        # ---- PASO 4: Simulaciones ----
        step(4, "CORRIENDO SIMULACIONES LONG/SHORT/FLAT")
        all_results = run_all_simulations(xgb_data, lstm_data)

        # ---- PASO 5: Reporte PDF ----
        step(5, "GENERANDO REPORTE PDF V3")
        try:
            from generate_report_v3 import generate_report_v3
            report_path = generate_report_v3(
                all_results=all_results,
                errors_log=errors_log,
            )
            print(f"\n  [Report] Guardado: {report_path}")
        except Exception as e:
            msg = f"Generacion PDF v3 fallo: {e}"
            print(f"  [WARN] {msg}")
            errors_log.append(msg)
            traceback.print_exc()

        # ---- Resumen final ----
        banner("PIPELINE V3 COMPLETO. Reporte en reporte_uni_ml_v3.pdf")

        print("\nRESUMEN POR COMBINACION:")
        print("-" * 75)
        print(f"  {'Variante':<12} {'Freq':<8} {'Trades':<8} {'%Flat':<8} "
              f"{'Fees':<10} {'Net P&L':<14} {'Sharpe':<8} {'MaxDD'}")
        print("-" * 75)

        best_key = None
        best_net = float("-inf")

        COMBINATIONS = ["A-4h", "B-4h", "C-4h", "A-daily", "B-daily", "C-daily", "BH"]
        for key in COMBINATIONS:
            data = all_results.get(key, {})
            stats = data.get("stats", {})
            if not stats:
                continue
            variant = key.split("-")[0] if "-" in key else key
            freq = key.split("-")[1] if "-" in key else "-"
            net = stats.get("net_pnl_usd", 0)
            if key != "BH" and net > best_net:
                best_net = net
                best_key = key
            print(
                f"  {variant:<12} {freq:<8} "
                f"{stats.get('total_trades', 0):<8} "
                f"{stats.get('pct_flat', 0):<7.1f}% "
                f"${stats.get('total_fees_usd', 0):<9.2f} "
                f"${net:<+13,.2f} "
                f"{stats.get('sharpe_ratio', 0):<8.3f} "
                f"{stats.get('max_dd_pct', 0):.2f}%"
            )

        print("-" * 75)

        if best_key:
            print(f"\n  MEJOR COMBINACION: {best_key}")
            print(f"  Net Profit:        ${best_net:+,.2f}")

        if errors_log:
            print(f"\n  Errores no fatales: {len(errors_log)}")
            for err in errors_log[:5]:
                print(f"    * {str(err)[:100]}")

        print(f"\nPIPELINE V3 COMPLETO. Reporte en reporte_uni_ml_v3.pdf")

    except KeyboardInterrupt:
        print("\n\n[Pipeline] Interrumpido por el usuario.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Pipeline] ERROR INESPERADO: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
