"""
main.py — Orquestador del pipeline completo v2.

V2: Pipeline multi-frecuencia (1h, 4h, daily) con datos reales.

Orden:
  1. Fetch: Binance hourly + Fear & Greed + Dune holders + TheGraph protocol
  2. Build multi-freq datasets (1h, 4h, daily)
  3. Per freq: feature engineering + 3 models + evaluation
  4. Generate reporte_uni_ml_v2.pdf
  5. Print summary

Uso:
  python3 main.py
  python3 main.py --skip-lstm      # omitir LSTM (mas rapido)
  python3 main.py --freq daily     # solo frecuencia daily
"""

import argparse
import sys
import traceback
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path(__file__).parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FIGS_DIR = Path(__file__).parent / "data" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

FREQS = ["1h", "4h", "daily"]
TARGET_MAP = {
    "1h": "target_1h",
    "4h": "target_4h",
    "daily": "target_24h",
}
LOOKBACK_MAP = {
    "1h": [24, 72, 168],
    "4h": [6, 18, 42],
    "daily": [7, 14, 30],
}
ANNUALIZE_MAP = {
    "1h": 252 * 24,
    "4h": 252 * 6,
    "daily": 252,
}

errors_log = []


def banner(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def step(n, name: str):
    print(f"\n{'-' * 60}")
    print(f"  PASO {n}: {name}")
    print(f"{'-' * 60}")


# ===== PASO 1: Fetch =====

def run_fetch():
    step(1, "INGESTA DE DATOS v2")

    from fetch_data import (
        fetch_price_binance_hourly,
        fetch_fear_greed,
        load_dune_holders,
        load_thegraph_protocol,
    )

    # Binance hourly (paginado, 5 anios)
    print("\n[Fetch] Descargando datos horarios de Binance...")
    df_hourly = fetch_price_binance_hourly("UNIUSDT", days=1825)
    print(f"  [Fetch] Horario: {df_hourly.shape} | "
          f"{df_hourly['datetime'].min()} -> {df_hourly['datetime'].max()}")

    # Fear & Greed
    df_fg = pd.DataFrame()
    try:
        print("\n[Fetch] Descargando Fear & Greed Index...")
        df_fg = fetch_fear_greed(limit=2000)
        print(f"  [Fetch] Fear&Greed: {df_fg.shape}")
    except Exception as e:
        msg = f"Fear & Greed fallo: {e}"
        print(f"  [WARN] {msg}")
        errors_log.append(msg)

    # Dune holders (CSV real)
    df_holders = pd.DataFrame()
    try:
        print("\n[Fetch] Cargando Dune holders CSV...")
        df_holders = load_dune_holders()
        if df_holders.empty:
            raise ValueError("Dune holders CSV vacio o no encontrado")
        print(f"  [Fetch] Dune holders: {df_holders.shape}")
    except Exception as e:
        msg = f"Dune holders fallo: {e}"
        print(f"  [WARN] {msg}")
        errors_log.append(msg)

    # The Graph protocol (CSV real)
    df_protocol = pd.DataFrame()
    try:
        print("\n[Fetch] Cargando The Graph protocol CSV...")
        df_protocol = load_thegraph_protocol()
        if df_protocol.empty:
            raise ValueError("TheGraph protocol CSV vacio o no encontrado")
        print(f"  [Fetch] TheGraph protocol: {df_protocol.shape}")
    except Exception as e:
        msg = f"TheGraph protocol fallo: {e}"
        print(f"  [WARN] {msg}")
        errors_log.append(msg)

    return df_hourly, df_holders, df_protocol, df_fg


# ===== PASO 2: Build multi-freq datasets =====

def run_build_datasets(df_hourly, df_holders, df_protocol, df_fg):
    step(2, "CONSTRUYENDO DATASETS MULTI-FRECUENCIA")

    from process_data import build_multi_freq_datasets

    datasets = build_multi_freq_datasets(df_hourly, df_holders, df_protocol, df_fg)

    for freq, df in datasets.items():
        out = PROCESSED_DIR / f"dataset_{freq}.parquet"
        df.to_parquet(out, index=False)
        date_col = "datetime" if "datetime" in df.columns else "date"
        print(f"  [Build] {freq}: {df.shape} | guardado en {out.name}")

    return datasets


# ===== PASO 3: Feature engineering =====

def run_features_freq(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    from feature_engineering import feature_pipeline_freq

    target_col = TARGET_MAP[freq]
    df_feat = feature_pipeline_freq(df, freq=freq, target_col=target_col)

    out = PROCESSED_DIR / f"features_{freq}.parquet"
    df_feat.to_parquet(out, index=False)
    print(f"  [Features] {freq}: {df_feat.shape} | guardado en {out.name}")
    return df_feat


# ===== PASO 4a: XGBoost Baseline =====

def run_xgb_baseline(df_feat: pd.DataFrame, freq: str) -> tuple:
    from model_baseline import train_xgboost, walk_forward_validation
    from feature_engineering import get_feature_cols

    target_col = TARGET_MAP[freq]
    print(f"\n[XGB Baseline] {freq} | target={target_col}")

    model, scaler, imputer, feature_cols, importance, test_data = \
        train_xgboost(df_feat, target_col=target_col)
    X_test_scaled, y_test, y_proba = test_data

    df_wfv = walk_forward_validation(df_feat, n_folds=5, target_col=target_col)

    importance.to_parquet(PROCESSED_DIR / f"xgb_importance_{freq}.parquet", index=False)
    df_wfv.to_parquet(PROCESSED_DIR / f"wfv_results_{freq}.parquet", index=False)

    split_idx = int(len(df_feat) * 0.8)
    df_test = df_feat.iloc[split_idx:].copy().reset_index(drop=True)
    df_test["target"] = df_test.get("target", df_test.get(target_col, y_test))
    if "pct_return" not in df_test.columns:
        df_test["pct_return"] = df_test["close"].pct_change().fillna(0)

    # Asegurar longitud consistente
    df_test = df_test.iloc[:len(y_proba)].copy()

    return model, scaler, imputer, importance, y_proba, df_test, df_wfv


# ===== PASO 4b: XGBoost Fine-Tuned =====

def run_xgb_tuned(df_feat: pd.DataFrame, freq: str) -> tuple:
    from model_baseline import fine_tune_xgboost

    target_col = TARGET_MAP[freq]
    print(f"\n[XGB Tuned] {freq} | target={target_col}")

    try:
        best_model, best_params, df_tuning, scaler_ft, imputer_ft, y_proba_ft, df_test_ft = \
            fine_tune_xgboost(df_feat, n_combos=20, target_col=target_col)

        df_tuning.to_parquet(PROCESSED_DIR / f"xgb_tuning_{freq}.parquet", index=False)

        df_test_ft["target"] = df_test_ft.get("target", df_test_ft.get(target_col))
        if "pct_return" not in df_test_ft.columns:
            df_test_ft["pct_return"] = df_test_ft["close"].pct_change().fillna(0)

        df_test_ft = df_test_ft.iloc[:len(y_proba_ft)].copy()
        return best_model, best_params, df_tuning, y_proba_ft, df_test_ft

    except Exception as e:
        msg = f"XGB Fine-tuning {freq} fallo: {e}"
        print(f"  [WARN] {msg}")
        errors_log.append(msg)
        traceback.print_exc()
        return None, None, None, None, None


# ===== PASO 4c: LSTM (en subprocess) =====

def run_lstm_freq(df_feat: pd.DataFrame, freq: str, lookback: int) -> tuple:
    """Ejecuta LSTM en subprocess separado para evitar conflicto XGBoost+PyTorch."""
    target_col = TARGET_MAP[freq]
    print(f"\n[LSTM] {freq} | lookback={lookback} | target={target_col}")

    features_path = PROCESSED_DIR / f"features_{freq}.parquet"
    if not features_path.exists():
        df_feat.to_parquet(features_path, index=False)

    proba_path = PROCESSED_DIR / f"lstm_proba_{freq}.npy"
    true_path = PROCESSED_DIR / f"lstm_ytrue_{freq}.npy"
    hist_path = PROCESSED_DIR / f"lstm_history_{freq}.parquet"

    lstm_script = f"""
import sys
sys.path.insert(0, '{Path(__file__).parent}')
import numpy as np
import pandas as pd
from pathlib import Path
from model_lstm import train_lstm

df = pd.read_parquet('{features_path}')
model, scaler, imputer, history, test_data = train_lstm(
    df, lookback={lookback}, target_col='{target_col}'
)
X_te_seq, y_te_seq, y_proba = test_data

np.save('{proba_path}', y_proba)
np.save('{true_path}', y_te_seq)
if history:
    pd.DataFrame(history).to_parquet('{hist_path}', index=False)
print('[LSTM worker] Completado OK para freq={freq}')
"""
    script_path = PROCESSED_DIR / f"_lstm_runner_{freq}.py"
    script_path.write_text(lstm_script)

    print(f"  [LSTM] Lanzando subprocess para {freq}...")
    result = subprocess.run(
        [sys.executable, "-u", str(script_path)],
        capture_output=False,
        timeout=1800,  # 30 min timeout para 1h con 43k filas
        cwd=str(Path(__file__).parent),
    )

    if result.returncode != 0:
        msg = f"LSTM subprocess {freq} fallo con codigo {result.returncode}"
        print(f"  [WARN] {msg}")
        errors_log.append(msg)
        return None, None, None

    if not Path(proba_path).exists():
        msg = f"LSTM {freq}: archivo de probas no encontrado"
        errors_log.append(msg)
        return None, None, None

    y_proba_lstm = np.load(str(proba_path))
    y_te_seq = np.load(str(true_path))

    # Reconstruir df_test para LSTM
    split_idx = int(len(df_feat) * 0.8)
    df_test_lstm = df_feat.iloc[split_idx + lookback:].copy().reset_index(drop=True)
    df_test_lstm = df_test_lstm.iloc[:len(y_proba_lstm)].copy()

    if "target" not in df_test_lstm.columns:
        if target_col in df_test_lstm.columns:
            df_test_lstm["target"] = df_test_lstm[target_col]
        else:
            df_test_lstm["target"] = y_te_seq[:len(df_test_lstm)]

    if "pct_return" not in df_test_lstm.columns:
        df_test_lstm["pct_return"] = df_test_lstm["close"].pct_change().fillna(0)

    lstm_history = pd.read_parquet(hist_path) if Path(hist_path).exists() else pd.DataFrame()

    return y_proba_lstm, df_test_lstm, lstm_history


# ===== PASO 5: Evaluate =====

def run_evaluate_freq(
    freq: str,
    y_proba_xgb: np.ndarray,
    df_test_xgb: pd.DataFrame,
    y_proba_ft=None,
    df_test_ft=None,
    y_proba_lstm=None,
    df_test_lstm=None,
) -> tuple:
    from evaluate import evaluate_model

    results = []
    sim_results = {}

    def safe_eval(y_proba, df_test, model_name):
        if y_proba is None or df_test is None or len(y_proba) == 0:
            return
        try:
            df_t = df_test.copy()
            if "target" not in df_t.columns:
                target_col = TARGET_MAP[freq]
                if target_col in df_t.columns:
                    df_t["target"] = df_t[target_col]
                else:
                    print(f"  [WARN] No target column for {model_name} {freq}")
                    return
            if "pct_return" not in df_t.columns:
                df_t["pct_return"] = df_t["close"].pct_change().fillna(0)

            # Alinear longitudes
            min_len = min(len(y_proba), len(df_t))
            y_proba_al = y_proba[:min_len]
            df_t = df_t.iloc[:min_len].copy()

            r, df_sim = evaluate_model(
                df_t, y_proba_al,
                model_name=model_name,
                freq=freq,
                transaction_cost=0.001,
            )
            results.append(r)
            sim_results[model_name] = df_sim
            df_sim.to_parquet(PROCESSED_DIR / f"sim_{model_name}_{freq}.parquet", index=False)
        except Exception as e:
            msg = f"Evaluacion {model_name} {freq} fallo: {e}"
            print(f"  [WARN] {msg}")
            errors_log.append(msg)
            traceback.print_exc()

    safe_eval(y_proba_xgb, df_test_xgb, "XGB_baseline")
    safe_eval(y_proba_ft, df_test_ft, "XGB_tuned")
    safe_eval(y_proba_lstm, df_test_lstm, "LSTM")

    return results, sim_results


# ===== Main =====

def main():
    parser = argparse.ArgumentParser(description="Pipeline ML On-Chain UNI v2")
    parser.add_argument("--skip-lstm", action="store_true", help="Omitir LSTM")
    parser.add_argument("--freq", default=None, help="Solo esta frecuencia (1h, 4h, daily)")
    args = parser.parse_args()

    banner("PIPELINE ML ON-CHAIN v2 - UNI (Uniswap) - REAL DATA")
    print(f"  Token:     UNI")
    print(f"  Fuentes:   Binance hourly + Dune holders + TheGraph + Fear&Greed")
    print(f"  Freqs:     {FREQS}")
    print(f"  Modelos:   XGBoost baseline, XGBoost tuned, LSTM")
    print(f"  LSTM:      {'desactivado (--skip-lstm)' if args.skip_lstm else 'activado'}")
    if args.freq:
        print(f"  Modo:      Solo freq={args.freq}")

    # Coleccion de resultados
    all_datasets = {}
    all_features = {}
    all_results = {}
    all_sim_results = {}
    all_importance = {}
    all_tuning = {}
    all_best_params = {}
    all_lstm_history = {}
    all_wfv = {}

    try:
        # === PASO 1: Fetch ===
        df_hourly, df_holders, df_protocol, df_fg = run_fetch()

        # === PASO 2: Build multi-freq ===
        datasets = run_build_datasets(df_hourly, df_holders, df_protocol, df_fg)
        all_datasets = datasets

        # Determinar frecuencias a procesar
        target_freqs = [args.freq] if args.freq and args.freq in FREQS else FREQS

        for freq in target_freqs:
            if freq not in datasets:
                print(f"  [WARN] Frecuencia {freq} no disponible en datasets")
                continue

            df_freq = datasets[freq]

            if len(df_freq) < 100:
                msg = f"Frecuencia {freq}: solo {len(df_freq)} filas, saltando"
                print(f"  [WARN] {msg}")
                errors_log.append(msg)
                continue

            banner(f"PROCESANDO FRECUENCIA: {freq}")

            # === PASO 3: Feature engineering ===
            step(3, f"FEATURE ENGINEERING - {freq}")
            try:
                df_feat = run_features_freq(df_freq, freq)
                all_features[freq] = df_feat
            except Exception as e:
                msg = f"Feature engineering {freq} fallo: {e}"
                print(f"  [ERROR] {msg}")
                errors_log.append(msg)
                traceback.print_exc()
                continue

            if len(df_feat) < 100:
                msg = f"{freq}: solo {len(df_feat)} filas despues de feature eng, saltando"
                print(f"  [WARN] {msg}")
                errors_log.append(msg)
                continue

            # === PASO 4a: XGBoost Baseline ===
            step(4, f"XGBoost BASELINE - {freq}")
            y_proba_xgb = None
            df_test_xgb = None
            importance = None
            df_wfv_freq = None

            try:
                model_xgb, scaler_xgb, imputer_xgb, importance, y_proba_xgb, \
                    df_test_xgb, df_wfv_freq = run_xgb_baseline(df_feat, freq)
                all_importance[freq] = importance
                all_wfv[freq] = df_wfv_freq
            except Exception as e:
                msg = f"XGB Baseline {freq} fallo: {e}"
                print(f"  [WARN] {msg}")
                errors_log.append(msg)
                traceback.print_exc()

            # === PASO 4b: XGBoost Fine-Tuned ===
            step(5, f"XGBoost FINE-TUNED - {freq}")
            y_proba_ft = None
            df_test_ft = None
            best_params = None

            try:
                best_model_ft, best_params, df_tuning, y_proba_ft, df_test_ft = \
                    run_xgb_tuned(df_feat, freq)
                if best_params:
                    all_tuning[freq] = df_tuning
                    all_best_params[freq] = best_params
            except Exception as e:
                msg = f"XGB Tuned {freq} fallo: {e}"
                print(f"  [WARN] {msg}")
                errors_log.append(msg)
                traceback.print_exc()

            # === PASO 4c: LSTM ===
            y_proba_lstm = None
            df_test_lstm = None
            lstm_history = pd.DataFrame()

            if not args.skip_lstm:
                step(6, f"LSTM - {freq}")
                try:
                    # Para 1h con 43k filas: usar solo el primer lookback para velocidad
                    lookback = LOOKBACK_MAP[freq][0]
                    y_proba_lstm, df_test_lstm, lstm_history = \
                        run_lstm_freq(df_feat, freq, lookback)
                    if lstm_history is not None and not lstm_history.empty:
                        all_lstm_history[freq] = lstm_history
                except Exception as e:
                    msg = f"LSTM {freq} fallo: {e}"
                    print(f"  [WARN] {msg}")
                    errors_log.append(msg)
                    traceback.print_exc()

            # === PASO 7: Evaluate ===
            step(7, f"EVALUACION - {freq}")
            try:
                results_freq, sim_results_freq = run_evaluate_freq(
                    freq=freq,
                    y_proba_xgb=y_proba_xgb,
                    df_test_xgb=df_test_xgb,
                    y_proba_ft=y_proba_ft,
                    df_test_ft=df_test_ft,
                    y_proba_lstm=y_proba_lstm,
                    df_test_lstm=df_test_lstm,
                )
                all_results[freq] = results_freq
                all_sim_results[freq] = sim_results_freq
            except Exception as e:
                msg = f"Evaluacion {freq} fallo: {e}"
                print(f"  [ERROR] {msg}")
                errors_log.append(msg)
                traceback.print_exc()

        # === PASO 8: Reporte PDF ===
        step(8, "GENERANDO REPORTE PDF v2")
        try:
            from generate_report import generate_report

            report_path = generate_report(
                all_datasets=all_datasets,
                all_features=all_features,
                all_results=all_results,
                all_sim_results=all_sim_results,
                all_importance=all_importance,
                all_tuning=all_tuning,
                all_best_params=all_best_params,
                all_lstm_history=all_lstm_history,
                all_wfv=all_wfv,
                errors_log=errors_log,
            )
            print(f"\n  [Report] Guardado: {report_path}")
        except Exception as e:
            msg = f"Generacion de PDF fallo: {e}"
            print(f"  [WARN] {msg}")
            errors_log.append(msg)
            traceback.print_exc()

        # === Resumen final ===
        banner("PIPELINE V2 COMPLETO. Reporte en reporte_uni_ml_v2.pdf")

        print("\nRESUMEN POR FRECUENCIA:")
        print("-" * 80)
        best_per_freq = {}
        for freq in target_freqs:
            res_list = all_results.get(freq, [])
            if not res_list:
                print(f"  {freq}: sin resultados")
                continue
            best = max(res_list, key=lambda r: r.get("directional_accuracy", 0))
            best_per_freq[freq] = best
            print(f"\n  Frecuencia: {freq}")
            print(f"    Mejor modelo:        {best['model']}")
            print(f"    Directional Accuracy: {best['directional_accuracy']:.2%}")
            print(f"    Sharpe Ratio:         {best['sharpe_strategy']:.3f}")
            print(f"    Return vs B&H:        "
                  f"{best['total_return_strategy']:.2%} vs {best['total_return_bh']:.2%}")
            print(f"    Max Drawdown:         {best['max_drawdown']:.2%}")

        print("\n" + "-" * 80)
        if errors_log:
            print(f"\nErrores no fatales durante la ejecucion: {len(errors_log)}")
            for err in errors_log[:5]:
                print(f"  * {str(err)[:100]}")
            if len(errors_log) > 5:
                print(f"  ... y {len(errors_log)-5} mas (ver PDF seccion 11)")

        print("\nPIPELINE V2 COMPLETO. Reporte en reporte_uni_ml_v2.pdf")

    except KeyboardInterrupt:
        print("\n\n[Pipeline] Interrumpido por el usuario.")
        sys.exit(0)
    except Exception as e:
        print(f"\n[Pipeline] ERROR INESPERADO: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
