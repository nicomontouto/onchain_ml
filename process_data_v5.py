"""
process_data_v5.py — Pipeline de datos v5.

Cambios vs v4:
  - Fuentes 100% horarias (sin datos diarios):
      * Binance 1h OHLCV
      * The Graph tokenHourDatas
      * Dune UNI transfers 1h
      * Dollar bar duration
  - Sin leakage: todos los datos tienen granularidad <= 4h
  - Solo frecuencia 4h (daily eliminado)
  - Features diarias eliminadas: whale_balance_pct, whale_delta_pct,
    herfindahl_index, holders_growth, fear_greed_value
  - Reemplazadas por: transfer_count, whale_volume_ratio,
    unique_senders, transfer_count_pct_change, dollar_bar_duration

Uso:
  python3 process_data_v5.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR      = Path(__file__).parent
DATA_DIR      = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Carga de fuentes
# ---------------------------------------------------------------------------

def load_binance() -> pd.DataFrame:
    path = DATA_DIR / "cache" / "binance_uniusdt_1h_1825d.parquet"
    df = pd.read_parquet(path)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"  [load] Binance 1h: {df.shape} | {df['datetime'].min()} -> {df['datetime'].max()}")
    return df


def load_thegraph() -> pd.DataFrame:
    path = DATA_DIR / "thegraph_uni_hourly.csv"
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df.rename(columns={"timestamp": "datetime"})
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"  [load] TheGraph 1h: {df.shape} | {df['datetime'].min()} -> {df['datetime'].max()}")
    return df


def load_dune() -> pd.DataFrame:
    path = DATA_DIR / "dune_uni_activity_1h.csv"
    df = pd.read_csv(path, parse_dates=["period_1h"])
    df = df.rename(columns={"period_1h": "datetime"})
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"  [load] Dune 1h: {df.shape} | {df['datetime'].min()} -> {df['datetime'].max()}")
    return df


def load_dollar_bar_duration() -> pd.DataFrame:
    path = DATA_DIR / "dollar_bar_duration.csv"
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"  [load] Dollar bar duration: {df.shape} | {df['datetime'].min()} -> {df['datetime'].max()}")
    return df


# ---------------------------------------------------------------------------
# Merge horario
# ---------------------------------------------------------------------------

def merge_hourly(
    df_binance: pd.DataFrame,
    df_tg: pd.DataFrame,
    df_dune: pd.DataFrame,
    df_dbar: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge de las 4 fuentes sobre el datetime de Binance (base).
    - The Graph: left join, forward-fill gaps (<= 6h)
    - Dune: left join, fill 0 donde no hay transferencias (valido)
    - Dollar bar: left join, forward-fill (la duracion se mantiene hasta la siguiente barra)
    """
    print("\n  [merge] Mergeando fuentes horarias...")

    df = df_binance.copy()

    # The Graph
    df = df.merge(
        df_tg[["datetime", "volumeUSD", "totalValueLockedUSD", "priceUSD", "feesUSD"]],
        on="datetime", how="left",
    )
    # Forward-fill gaps del The Graph (max 6h para no propagar demasiado)
    for col in ["volumeUSD", "totalValueLockedUSD", "priceUSD", "feesUSD"]:
        df[col] = df[col].ffill(limit=6).fillna(0)

    # Dune (horas sin actividad = 0, no ffill)
    df = df.merge(
        df_dune[["datetime", "transfer_count", "volume_uni",
                 "unique_receivers", "unique_senders", "whale_volume_uni"]],
        on="datetime", how="left",
    )
    for col in ["transfer_count", "volume_uni", "unique_receivers",
                "unique_senders", "whale_volume_uni"]:
        df[col] = df[col].fillna(0)

    # Dollar bar duration (forward-fill: el valor de la ultima barra completada)
    df = df.merge(df_dbar[["datetime", "dollar_bar_duration"]], on="datetime", how="left")
    df["dollar_bar_duration"] = df["dollar_bar_duration"].ffill().bfill()

    df = df.dropna(subset=["close"]).sort_values("datetime").reset_index(drop=True)
    print(f"  [merge] Dataset 1h: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# Features a nivel horario (antes de resamplear)
# ---------------------------------------------------------------------------

def add_hourly_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computa features que deben calcularse sobre datos horarios antes del resample.
    """
    df = df.copy().sort_values("datetime").reset_index(drop=True)

    # Divergencia precio CEX vs DEX (horaria, sin leakage)
    price_ref = df["priceUSD"].replace(0, np.nan)
    df["price_divergence_1h"] = ((df["close"] - price_ref).abs() / price_ref).fillna(0)

    # Ratio whale vs total (horario)
    vol_uni = df["volume_uni"].replace(0, np.nan)
    df["whale_volume_ratio_1h"] = (df["whale_volume_uni"] / vol_uni).fillna(0)

    # Volume/TVL ratio horario
    tvl = df["totalValueLockedUSD"].replace(0, np.nan)
    df["volume_tvl_ratio_1h"] = (df["volumeUSD"] / tvl).fillna(0)

    return df


# ---------------------------------------------------------------------------
# Resample a 4h
# ---------------------------------------------------------------------------

def resample_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega dataset horario a 4h.
    OHLCV: agregacion estandar.
    The Graph: volumeUSD y feesUSD = sum (flujos), TVL y priceUSD = last (snapshot).
    Dune: transfer_count, volume_uni, whale_volume_uni = sum; unique_* = mean.
    Features horarias derivadas: mean dentro del periodo.
    dollar_bar_duration: last (duracion de la ultima barra completada en el periodo).
    """
    df = df_1h.copy()
    df = df.set_index("datetime")

    agg = {
        # OHLCV
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
        # The Graph
        "volumeUSD":            "sum",
        "totalValueLockedUSD":  "last",
        "priceUSD":             "last",
        "feesUSD":              "sum",
        # Dune
        "transfer_count":    "sum",
        "volume_uni":        "sum",
        "unique_receivers":  "mean",
        "unique_senders":    "mean",
        "whale_volume_uni":  "sum",
        # Features horarias
        "price_divergence_1h":   "mean",
        "whale_volume_ratio_1h": "mean",
        "volume_tvl_ratio_1h":   "mean",
        # Dollar bar
        "dollar_bar_duration": "last",
    }

    # Solo agregar columnas que existen
    agg = {k: v for k, v in agg.items() if k in df.columns}

    df_4h = df.resample("4h").agg(agg)
    df_4h = df_4h.dropna(subset=["close"]).reset_index()
    df_4h = df_4h.rename(columns={
        "price_divergence_1h":   "price_divergence",
        "whale_volume_ratio_1h": "whale_volume_ratio",
        "volume_tvl_ratio_1h":   "volume_tvl_ratio",
    })
    print(f"  [resample] Dataset 4h: {df_4h.shape}")
    return df_4h


# ---------------------------------------------------------------------------
# Features a nivel 4h
# ---------------------------------------------------------------------------

def add_4h_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computa features sobre el dataset 4h.
    """
    df = df.copy().sort_values("datetime").reset_index(drop=True)

    # Retornos
    df["log_return"]  = np.log(df["close"] / df["close"].shift(1))
    df["pct_return"]  = df["close"].pct_change()

    # Rango high-low normalizado
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

    # Volatilidad: rolling 6 barras de 4h = 24h
    df["volatility_24h"] = df["log_return"].rolling(6).std()

    # Volatilidad largo plazo: rolling 18 barras = 72h (~3 dias)
    df["log_return_roll_std72"] = df["log_return"].shift(1).rolling(18).std()

    # Vol rolling del volumen DEX
    df["volumeUSD_roll_std72"] = df["volumeUSD"].shift(1).rolling(18).std()

    # Cambio porcentual de actividad on-chain
    tc = df["transfer_count"].replace(0, np.nan)
    df["transfer_count_pct_change"] = tc.pct_change() * 100

    # Lags de log_return (1 barra = 4h, 4 barras = 16h)
    df["log_return_lag1"] = df["log_return"].shift(1)
    df["log_return_lag4"] = df["log_return"].shift(4)

    # Lag de price_divergence
    df["price_divergence_lag1"] = df["price_divergence"].shift(1)

    # Winsorizar outliers (IQR 3x)
    for col in ["log_return", "transfer_count_pct_change", "volume_tvl_ratio",
                "high_low_range", "whale_volume_ratio"]:
        if col not in df.columns:
            continue
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        df[col] = df[col].clip(q1 - 3 * iqr, q3 + 3 * iqr)

    print(f"  [features 4h] {df.shape[1]} columnas | {df.shape[0]} filas")
    return df


# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

def build_dataset_4h() -> pd.DataFrame:
    """
    Pipeline completo v5: carga, merge, resample, features.
    Guarda en data/processed/features_4h.parquet y retorna el DataFrame.
    """
    print("\n" + "=" * 60)
    print("PROCESS DATA v5 — Dataset 4h sin leakage")
    print("=" * 60)

    df_binance = load_binance()
    df_tg      = load_thegraph()
    df_dune    = load_dune()
    df_dbar    = load_dollar_bar_duration()

    df_1h  = merge_hourly(df_binance, df_tg, df_dune, df_dbar)
    df_1h  = add_hourly_features(df_1h)
    df_4h  = resample_to_4h(df_1h)
    df_4h  = add_4h_features(df_4h)

    out = PROCESSED_DIR / "features_4h.parquet"
    df_4h.to_parquet(out, index=False)
    print(f"\n  [save] Guardado: {out}")
    print(f"  Shape final: {df_4h.shape}")
    print(f"  Rango: {df_4h['datetime'].min()} -> {df_4h['datetime'].max()}")
    print(f"  Nulos:\n{df_4h.isnull().sum()[df_4h.isnull().sum() > 0].to_string()}")

    return df_4h


if __name__ == "__main__":
    df = build_dataset_4h()
    print("\nColumnas disponibles:")
    print(list(df.columns))
