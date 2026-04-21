"""
evaluate.py — Metricas financieras y comparacion estrategia vs buy & hold.

V2: simulate_strategy con transaction_cost y freq.
    max_drawdown, sharpe_ratio con analizacion por frecuencia.
    evaluate_model ampliado con max_drawdown.

Puede correrse de forma independiente:
  python evaluate.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"

ANNUALIZE_MAP = {
    "1h": 252 * 24,
    "4h": 252 * 6,
    "daily": 252,
}


# --- Metricas ---

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Porcentaje de periodos con direccion correcta."""
    return float(np.mean(y_true == y_pred))


def information_coefficient(y_proba: np.ndarray, y_true: np.ndarray) -> float:
    """
    Correlacion de Spearman entre y_proba e y_true.
    IC > 0.05 se considera senal real en quant finance.
    """
    try:
        corr, _ = spearmanr(y_proba, y_true)
        return float(corr)
    except Exception:
        return float("nan")


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum drawdown de la curva de equity."""
    if len(equity_curve) == 0:
        return 0.0
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max.replace(0, np.nan)
    return float(drawdown.min())


def sharpe_ratio(returns, freq: str = "daily") -> float:
    """
    Sharpe ratio anualizado.
    freq='daily': multiply by sqrt(252)
    freq='4h':    multiply by sqrt(252*6)
    freq='1h':    multiply by sqrt(252*24)
    """
    if isinstance(returns, pd.Series):
        r = returns.dropna()
    else:
        r = pd.Series(returns).dropna()

    if len(r) == 0 or r.std() == 0:
        return 0.0

    ann_factor = ANNUALIZE_MAP.get(freq, 252)
    sr = r.mean() / r.std() * np.sqrt(ann_factor)
    return float(sr)


def simulate_strategy(
    df_test: pd.DataFrame,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    transaction_cost: float = 0.001,
    freq: str = "daily",
) -> pd.DataFrame:
    """
    Simula estrategia long/flat:
    - Long cuando model predice 1 (proba >= threshold)
    - Flat cuando predice 0
    - transaction_cost: 0.001 = 0.1% por media vuelta (entrada o salida)
    - Un round-trip completo cuesta 2 * transaction_cost = 0.2%
    - Se aplica cuando la senal cambia (0->1 = entrada, 1->0 = salida)

    strategy_return = pct_return * signal - cost_applied

    Requiere que df_test tenga columna 'pct_return'.
    Retorna DataFrame con columnas:
      date/datetime, pct_return, signal, strategy_return, cum_strategy, cum_bh
    """
    df = df_test.copy().reset_index(drop=True)
    df["signal"] = (y_proba >= threshold).astype(int)

    # Calcular costos de transaccion por cambio de senal
    # Un cambio de senal (0->1 o 1->0) genera un costo de transaction_cost
    # El costo se aplica en el periodo del cambio, sobre el retorno bruto
    signal_change = df["signal"].diff().abs().fillna(0)

    # strategy_return bruto: pct_return cuando estamos largos
    gross_return = df["pct_return"] * df["signal"]

    # Costo: se descuenta cuando hay un cambio de senal (entry o exit)
    # Si entramos (0->1): pagamos transaction_cost
    # Si salimos (1->0): pagamos transaction_cost
    # El costo se descuenta del retorno del periodo en que ocurre el cambio
    cost = signal_change * transaction_cost

    df["strategy_return"] = gross_return - cost

    df["cum_strategy"] = (1 + df["strategy_return"]).cumprod()
    df["cum_bh"] = (1 + df["pct_return"]).cumprod()

    date_col = "datetime" if "datetime" in df.columns else "date"
    cols = [date_col, "pct_return", "signal", "strategy_return", "cum_strategy", "cum_bh"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


def print_comparison_table(
    df_sim: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    freq: str = "daily",
):
    """Imprime tabla comparativa: estrategia vs buy & hold."""
    da = directional_accuracy(y_true, y_pred)
    ic = information_coefficient(y_proba, y_true)
    sr_strategy = sharpe_ratio(df_sim["strategy_return"], freq=freq)
    sr_bh = sharpe_ratio(df_sim["pct_return"], freq=freq)
    md = max_drawdown(df_sim["cum_strategy"])

    final_strategy = df_sim["cum_strategy"].iloc[-1] - 1
    final_bh = df_sim["cum_bh"].iloc[-1] - 1

    print("\n" + "=" * 55)
    print(f"{'METRICA':<30} {'ESTRATEGIA':>12} {'BUY & HOLD':>12}")
    print("=" * 55)
    print(f"{'Directional Accuracy':<30} {da:>11.2%} {'---':>12}")
    print(f"{'Information Coefficient (IC)':<30} {ic:>11.4f} {'---':>12}")
    print(f"{'Sharpe Ratio (anualizado)':<30} {sr_strategy:>11.3f} {sr_bh:>11.3f}")
    print(f"{'Retorno total periodo':<30} {final_strategy:>11.2%} {final_bh:>11.2%}")
    print(f"{'Max Drawdown':<30} {md:>11.2%} {'---':>12}")
    print(f"{'Periodos en mercado':<30} {int(df_sim['signal'].sum()):>11} {len(df_sim):>11}")
    print("=" * 55)

    if ic > 0.05:
        print("IC > 0.05: senal estadisticamente relevante")
    else:
        print("IC <= 0.05: senal debil o ruidosa")

    if sr_strategy > sr_bh:
        print("Estrategia supera Buy & Hold en Sharpe")
    else:
        print("Buy & Hold supera la estrategia en Sharpe")


# --- Evaluacion completa ---

def evaluate_model(
    df_test: pd.DataFrame,
    y_proba: np.ndarray,
    model_name: str = "Modelo",
    freq: str = "daily",
    transaction_cost: float = 0.001,
) -> tuple:
    """
    Calcula todas las metricas para un modelo.

    df_test debe tener columnas: date/datetime, pct_return, target
    y_proba: probabilidades del modelo para la clase positiva

    V2: Incluye max_drawdown, annualization por frecuencia, transaction_cost.

    Retorna (dict con todas las metricas, df_sim).
    """
    print(f"\n[evaluate] Evaluando: {model_name} (freq={freq})")
    date_col = "datetime" if "datetime" in df_test.columns else "date"
    if date_col in df_test.columns:
        print(f"  Test samples: {len(df_test)} | Periodo: "
              f"{df_test[date_col].min()} -> {df_test[date_col].max()}")

    y_true = df_test["target"].values.astype(int)
    y_pred = (y_proba >= 0.5).astype(int)

    da = directional_accuracy(y_true, y_pred)
    ic = information_coefficient(y_proba, y_true)

    df_sim = simulate_strategy(df_test, y_proba, transaction_cost=transaction_cost, freq=freq)
    sr = sharpe_ratio(df_sim["strategy_return"], freq=freq)
    sr_bh = sharpe_ratio(df_sim["pct_return"], freq=freq)
    md = max_drawdown(df_sim["cum_strategy"])

    try:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = float("nan")

    print_comparison_table(df_sim, y_true, y_pred, y_proba, freq=freq)

    results = {
        "model": model_name,
        "freq": freq,
        "directional_accuracy": da,
        "information_coefficient": ic,
        "roc_auc": auc,
        "sharpe_strategy": sr,
        "sharpe_bh": sr_bh,
        "total_return_strategy": float(df_sim["cum_strategy"].iloc[-1] - 1),
        "total_return_bh": float(df_sim["cum_bh"].iloc[-1] - 1),
        "max_drawdown": md,
        "periods_in_market": int(df_sim["signal"].sum()),
        "transaction_cost": transaction_cost,
    }
    return results, df_sim


# --- Entry point ---

if __name__ == "__main__":
    features_path = PROCESSED_DIR / "features.parquet"
    if not features_path.exists():
        print("Primero corre feature_engineering.py para generar features.parquet")
        exit(1)

    print("=" * 60)
    print("TEST INDEPENDIENTE: evaluate.py v2")
    print("=" * 60)

    df = pd.read_parquet(features_path)
    split_idx = int(len(df) * 0.8)
    df_test = df.iloc[split_idx:].copy()

    np.random.seed(42)
    y_proba_mock = np.random.uniform(0.3, 0.7, size=len(df_test))

    results, df_sim = evaluate_model(
        df_test, y_proba_mock,
        model_name="Mock (random)",
        freq="daily",
        transaction_cost=0.001,
    )
    df_sim.to_parquet(PROCESSED_DIR / "simulation_results.parquet", index=False)
    print("\n[OK] evaluate.py v2 corrio sin errores.")
