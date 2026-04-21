"""
feature_engineering.py — Lags, rolling stats, seniales whale y target.

V2: Agrega feature_pipeline_freq para multi-frecuencia.
    LAG_COLS ampliados: log_return, whale_delta_pct, whale_balance_pct,
                        volumeUSD, fear_greed_value
    Targets multiples: target_1h, target_4h, target_24h

Puede correrse de forma independiente:
  python feature_engineering.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Columnas sobre las que se calculan lags y rolling
LAG_COLS = [
    "log_return",
    "whale_delta_pct",
    "whale_pct_delta",
    "whale_balance_pct",
    "holders_growth",
    "number_of_holders",
    "whale_count",
    "herfindahl_index",
    "volumeUSD",
    "volume_usd",
    "tvl_usd",
    "fees_usd",
    "feesUSD",
    "volume_tvl_ratio",
    "fee_revenue",
    "fear_greed_value",
    "fear_greed_class",
    "activity_count",
    "activity_delta_pct",
    "whale_activity_ratio",
    "volume_delta_pct",
    "unique_addresses",
    "price_divergence",
    "high_low_range",
    "volatility_24h",
]
LAG_PERIODS = [1, 2, 3, 7]  # default daily

ROLLING_COLS = [
    "log_return",
    "whale_delta_pct",
    "whale_balance_pct",
    "number_of_holders",
    "volume_usd",
    "volumeUSD",
    "tvl_usd",
    "volume_tvl_ratio",
    "activity_count",
    "whale_activity_ratio",
    "uni_volume",
    "fear_greed_value",
    "high_low_range",
    "volatility_24h",
]
ROLLING_WINDOWS = [7, 14, 30]  # default daily

# Configuracion por frecuencia
FREQ_CONFIG = {
    "1h": {
        "lag_periods": [1, 4, 24, 168],
        "rolling_windows": [24, 72, 168],
    },
    "4h": {
        "lag_periods": [1, 6, 18, 42],
        "rolling_windows": [24, 72, 168],
    },
    "daily": {
        "lag_periods": [1, 2, 3, 7],
        "rolling_windows": [7, 14, 30],
    },
}


# --- Lags ---

def add_lags(df: pd.DataFrame, lag_periods: list = None) -> pd.DataFrame:
    """
    Agrega columnas lagged. shift hacia atras garantiza no data leakage.
    """
    df = df.copy()
    periods = lag_periods if lag_periods is not None else LAG_PERIODS
    created = []
    for col in LAG_COLS:
        if col not in df.columns:
            continue
        for lag in periods:
            new_col = f"{col}_lag{lag}"
            df[new_col] = df[col].shift(lag)
            created.append(new_col)
    print(f"  [lags] {len(created)} columnas creadas")
    return df


# --- Rolling stats ---

def add_rolling_stats(df: pd.DataFrame, rolling_windows: list = None) -> pd.DataFrame:
    """
    Agrega media y std rolling. shift(1) antes del rolling excluye el dia actual.
    """
    df = df.copy()
    windows = rolling_windows if rolling_windows is not None else ROLLING_WINDOWS
    created = []
    for col in ROLLING_COLS:
        if col not in df.columns:
            continue
        shifted = df[col].shift(1)
        for window in windows:
            mean_col = f"{col}_roll_mean{window}"
            std_col = f"{col}_roll_std{window}"
            df[mean_col] = shifted.rolling(window).mean()
            df[std_col] = shifted.rolling(window).std()
            created.extend([mean_col, std_col])
    print(f"  [rolling] {len(created)} columnas creadas")
    return df


# --- Seniales whale ---

def add_whale_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega seniales binarias basadas en divergencias/confluencias whale-precio.
    """
    df = df.copy()

    if "log_return" not in df.columns:
        df["whale_accumulation"] = 0
        df["whale_distribution"] = 0
        df["whale_confluence"] = 0
        df["sell_pressure"] = 0
        return df

    price_up = df["log_return"] > 0
    price_down = df["log_return"] < 0

    has_whale_delta = "whale_delta_pct" in df.columns and df["whale_delta_pct"].abs().sum() > 0
    has_holders = "holders_growth" in df.columns and df["holders_growth"].abs().sum() > 0
    has_balance_pct = "whale_balance_pct" in df.columns and df["whale_balance_pct"].abs().sum() > 0
    has_whale_ratio = "whale_activity_ratio" in df.columns and df["whale_activity_ratio"].abs().sum() > 0
    has_unique_addr = "unique_addresses" in df.columns and df["unique_addresses"].abs().sum() > 0

    if has_whale_delta:
        whale_increasing = df["whale_delta_pct"] > 0
        whale_decreasing = df["whale_delta_pct"] < 0

        df["whale_accumulation"] = (price_down & whale_increasing).astype(int)
        df["whale_distribution"] = (price_up & whale_decreasing).astype(int)
        df["whale_confluence"] = (price_up & whale_increasing).astype(int)
    else:
        df["whale_accumulation"] = 0
        df["whale_distribution"] = 0
        df["whale_confluence"] = 0

    if has_whale_delta and has_holders:
        holders_leaving = df["holders_growth"] < 0
        df["sell_pressure"] = (holders_leaving & (df["whale_delta_pct"] < 0)).astype(int)
    else:
        df["sell_pressure"] = 0

    if has_balance_pct and has_unique_addr:
        denom = df["unique_addresses"].replace(0, np.nan)
        df["concentration_ratio"] = df["whale_balance_pct"] / denom
    elif has_balance_pct:
        df["concentration_ratio"] = df["whale_balance_pct"]
    else:
        df["concentration_ratio"] = np.nan

    if has_whale_ratio:
        df["whale_dominance"] = (df["whale_activity_ratio"] > df["whale_activity_ratio"].median()).astype(int)
    else:
        df["whale_dominance"] = 0

    return df


# --- Targets ---

def build_target(df: pd.DataFrame, target_type: str = "direction") -> pd.DataFrame:
    """
    Construye el target sin data leakage (shift(-1)).
    """
    df = df.copy()

    if target_type == "direction":
        df["target"] = (df["log_return"].shift(-1) > 0).astype(float)
        df.loc[df.index[-1], "target"] = np.nan
    elif target_type == "return":
        df["target"] = df["log_return"].shift(-1)
        df.loc[df.index[-1], "target"] = np.nan
    else:
        raise ValueError(f"target_type '{target_type}' no valido")

    valid = df["target"].notna().sum()
    print(f"  [target] {valid} filas con target valido")
    return df


def build_multi_targets(df: pd.DataFrame, freq: str = "daily") -> pd.DataFrame:
    """
    Construye multiples targets segun frecuencia:
    - target_1h:  log_return.shift(-1) > 0
    - target_4h:  para 4h: shift(-1) > 0. Para 1h: cum return next 4 hours > 0
    - target_24h: para daily: shift(-1) > 0. Para 1h: cum return next 24 hours > 0

    No data leakage garantizado.
    """
    df = df.copy()

    lr = df["log_return"]

    # target_1h: siempre shift(-1) del log_return
    df["target_1h"] = (lr.shift(-1) > 0).astype(float)
    df.loc[df.index[-1], "target_1h"] = np.nan

    if freq == "1h":
        # target_4h en 1h: retorno acumulado de las proximas 4 horas > 0
        cum_4 = sum([lr.shift(-i) for i in range(1, 5)])
        df["target_4h"] = (cum_4 > 0).astype(float)
        df.iloc[-4:, df.columns.get_loc("target_4h")] = np.nan

        # target_24h en 1h: retorno acumulado de las proximas 24 horas > 0
        cum_24 = sum([lr.shift(-i) for i in range(1, 25)])
        df["target_24h"] = (cum_24 > 0).astype(float)
        df.iloc[-24:, df.columns.get_loc("target_24h")] = np.nan

    elif freq == "4h":
        # target_4h en 4h: shift(-1)
        df["target_4h"] = (lr.shift(-1) > 0).astype(float)
        df.loc[df.index[-1], "target_4h"] = np.nan

        # target_24h en 4h: retorno acumulado de los proximos 6 periodos (24h) > 0
        cum_6 = sum([lr.shift(-i) for i in range(1, 7)])
        df["target_24h"] = (cum_6 > 0).astype(float)
        df.iloc[-6:, df.columns.get_loc("target_24h")] = np.nan

    else:  # daily
        df["target_4h"] = (lr.shift(-1) > 0).astype(float)
        df.loc[df.index[-1], "target_4h"] = np.nan
        df["target_24h"] = (lr.shift(-1) > 0).astype(float)
        df.loc[df.index[-1], "target_24h"] = np.nan

    return df


# --- Pipeline v1 (compat) ---

def feature_pipeline(df: pd.DataFrame, target_type: str = "direction") -> pd.DataFrame:
    """
    Pipeline completo de feature engineering v1 (compatible).
    """
    print("\n[features] Iniciando feature engineering...")

    df = add_whale_signals(df)
    df = add_lags(df)
    df = add_rolling_stats(df)
    df = build_target(df, target_type=target_type)

    df = df.dropna(subset=["target"]).reset_index(drop=True)

    n_features = len(get_feature_cols(df))
    print(f"  [features] Total features: {n_features} | Filas finales: {len(df)}")
    return df


# --- Pipeline v2 por frecuencia ---

def feature_pipeline_freq(df: pd.DataFrame, freq: str = "1h", target_col: str = "target_1h") -> pd.DataFrame:
    """
    Pipeline completo de feature engineering para una frecuencia dada.

    freq: '1h', '4h', 'daily'
    target_col: columna target a usar como 'target' principal

    Retorna DataFrame con todas las features y columna 'target'.
    """
    print(f"\n[features v2] Pipeline feature engineering para freq={freq}...")

    cfg = FREQ_CONFIG.get(freq, FREQ_CONFIG["daily"])
    lag_periods = cfg["lag_periods"]
    rolling_windows = cfg["rolling_windows"]

    df = df.copy()

    # Asegurar que hay columna date o datetime
    if "datetime" in df.columns and "date" not in df.columns:
        df["date"] = df["datetime"].dt.normalize()
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    # Seniales whale
    df = add_whale_signals(df)

    # Lags con periodos apropiados para la frecuencia
    df = add_lags(df, lag_periods=lag_periods)

    # Rolling con ventanas apropiadas para la frecuencia
    df = add_rolling_stats(df, rolling_windows=rolling_windows)

    # Targets multi-frecuencia
    if "log_return" in df.columns:
        df = build_multi_targets(df, freq=freq)

    # Asignar target principal
    if target_col in df.columns:
        df["target"] = df[target_col]
    elif "target" not in df.columns and "log_return" in df.columns:
        df["target"] = (df["log_return"].shift(-1) > 0).astype(float)
        df.loc[df.index[-1], "target"] = np.nan

    # Eliminar filas sin target principal
    if "target" in df.columns:
        df = df.dropna(subset=["target"]).reset_index(drop=True)

    n_features = len(get_feature_cols(df))
    print(f"  [features v2] freq={freq} | features={n_features} | filas={len(df)}")
    return df


def get_feature_cols(df: pd.DataFrame) -> list:
    """
    Retorna la lista de columnas de features (excluye metadatos, OHLCV y targets).
    """
    exclude = {
        "date", "datetime", "target", "target_1h", "target_4h", "target_24h",
        "open", "high", "low", "close", "volume",
        "open_time", "close_time",
    }
    return [c for c in df.columns if c not in exclude]


# --- Entry point ---

if __name__ == "__main__":
    from fetch_data import fetch_price_binance, fetch_contract_activity, align_datasets
    from process_data import process_pipeline

    UNI_ADDRESS = "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984"

    print("=" * 60)
    print("TEST INDEPENDIENTE: feature_engineering.py v2")
    print("=" * 60)

    df_price = fetch_price_binance("UNIUSDT", days=365)
    df_activity = fetch_contract_activity(UNI_ADDRESS, max_pages=20)
    df_aligned = align_datasets(df_price, df_activity)
    df_processed = process_pipeline(df_aligned)
    df_features = feature_pipeline(df_processed, target_type="direction")

    out_path = PROCESSED_DIR / "features.parquet"
    df_features.to_parquet(out_path, index=False)
    print(f"\n[OK] Guardado en {out_path}")
    print(f"Shape: {df_features.shape}")
    print(f"Features: {get_feature_cols(df_features)[:10]} ...")
