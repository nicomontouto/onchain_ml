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

    # --- Nuevas features tecnicas (Binance) ---

    # RSI 14 barras
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD histogram (EMA12 - EMA26 - signal EMA9)
    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    df["macd_histogram"] = macd_line - macd_line.ewm(span=9, adjust=False).mean()

    # Bollinger Band width: 2*std / media (volatilidad relativa)
    roll20_mean = df["close"].rolling(20).mean()
    roll20_std  = df["close"].rolling(20).std()
    df["bb_width"] = (2 * roll20_std) / roll20_mean.replace(0, np.nan)

    # ATR 14 normalizado por close
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean() / df["close"].replace(0, np.nan)

    # Ratio volumen Binance vs media 24h (6 barras de 4h)
    vol_mean_24h = df["volume"].rolling(6).mean()
    df["volume_ratio"] = df["volume"] / vol_mean_24h.replace(0, np.nan)

    # Skewness de retornos en ventana 72h (18 barras)
    df["return_skew_72h"] = df["log_return"].rolling(18).skew()

    # Lags adicionales
    df["log_return_lag8"]  = df["log_return"].shift(8)
    df["log_return_lag12"] = df["log_return"].shift(12)

    # --- Nuevas features The Graph ---

    # Proxy de APR de fees: feesUSD / TVL por barra
    tvl = df["totalValueLockedUSD"].replace(0, np.nan)
    df["fee_apr_proxy"] = df["feesUSD"] / tvl

    # Cambio porcentual de TVL (liquidez entrando/saliendo)
    df["tvl_change_pct"] = df["totalValueLockedUSD"].pct_change().replace(
        [np.inf, -np.inf], np.nan)

    # --- Nuevas features Dune ---

    # Ratio de volumen retail (no-whale) sobre total
    vol_uni = df["volume_uni"].replace(0, np.nan)
    df["net_flow_proxy"] = (df["volume_uni"] - df["whale_volume_uni"]) / vol_uni

    # --- Indicadores clasicos adicionales ---

    # Stochastic %K y %D (N=14)
    low14  = df["low"].rolling(14).min()
    high14 = df["high"].rolling(14).max()
    hl_range = (high14 - low14).replace(0, np.nan)
    df["stoch_k"] = 100 * (df["close"] - low14) / hl_range
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # CCI — Commodity Channel Index (N=20)
    tp      = (df["high"] + df["low"] + df["close"]) / 3
    tp_mean = tp.rolling(20).mean()
    tp_mad  = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df["cci_20"] = (tp - tp_mean) / (0.015 * tp_mad.replace(0, np.nan))

    # Williams %R (N=14)
    df["williams_r"] = -100 * (high14 - df["close"]) / hl_range

    # ROC — Rate of Change (N=10 barras = 40h)
    df["roc_10"] = df["close"].pct_change(10) * 100

    # OBV momentum: cambio del OBV acumulado en 10 barras (40h)
    obv_sign      = np.sign(df["close"].diff()).fillna(0)
    obv           = (obv_sign * df["volume"]).cumsum()
    df["obv_momentum"] = obv - obv.shift(10)

    # MFI — Money Flow Index (N=14)
    mf         = tp * df["volume"]
    pos_mf     = mf.where(tp.diff() > 0, 0).rolling(14).sum()
    neg_mf     = mf.where(tp.diff() < 0, 0).rolling(14).sum()
    df["mfi_14"] = 100 - (100 / (1 + pos_mf / neg_mf.replace(0, np.nan)))

    # EMA ratio (EMA9 / EMA21) — tendencia continua
    ema9  = df["close"].ewm(span=9,  adjust=False).mean()
    ema21 = df["close"].ewm(span=21, adjust=False).mean()
    df["ema_ratio"] = ema9 / ema21.replace(0, np.nan)

    # Senal de cruce EMA9/EMA21 — señal primaria para meta-labeling con side
    # +1 si EMA9 > EMA21 (tendencia alcista), -1 si EMA9 < EMA21 (bajista)
    _ema_diff = ema9 - ema21
    df["ema_signal"] = np.sign(_ema_diff).astype(float).replace(0.0, 1.0)

    # ADX — fuerza de tendencia (N=14, Wilder smoothing via EWM)
    up_move  = df["high"] - df["high"].shift(1)
    dn_move  = df["low"].shift(1) - df["low"]
    plus_dm  = pd.Series(
        np.where((up_move > dn_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(
        np.where((dn_move > up_move) & (dn_move > 0), dn_move, 0.0), index=df.index)
    tr_adx   = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr_adx  = tr_adx.ewm(span=14, adjust=False).mean().replace(0, np.nan)
    plus_di  = 100 * plus_dm.ewm(span=14, adjust=False).mean() / atr_adx
    minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / atr_adx
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df["adx_14"] = dx.ewm(span=14, adjust=False).mean()

    # Donchian Channel position (N=20): donde esta el precio en su rango
    don_high = df["high"].rolling(20).max()
    don_low  = df["low"].rolling(20).min()
    df["donchian_pos"] = (df["close"] - don_low) / (don_high - don_low).replace(0, np.nan)

    # VWAP deviation rolling (N=24 barras ~ 4 dias)
    vwap_num = (df["close"] * df["volume"]).rolling(24).sum()
    vwap_den = df["volume"].rolling(24).sum().replace(0, np.nan)
    vwap     = vwap_num / vwap_den
    df["vwap_dev"] = (df["close"] - vwap) / vwap.replace(0, np.nan)

    # --- Features de direccion (para M2 del meta-labeling) ---

    # Candle body ratio: (close-open)/(high-low) — fuerza direccional intrabar [-1, 1]
    bar_range = (df["high"] - df["low"]).replace(0, np.nan)
    df["body_ratio"] = (df["close"] - df["open"]) / bar_range

    # Wicks: presion compradora y vendedora intrabar [0, 1]
    df["upper_wick_ratio"] = (df["high"] - df["close"]) / bar_range
    df["lower_wick_ratio"] = (df["close"] - df["low"]) / bar_range

    # Bollinger %B: posicion del precio dentro de las bandas
    upper_bb = roll20_mean + 2 * roll20_std
    lower_bb = roll20_mean - 2 * roll20_std
    df["bb_pct_b"] = (df["close"] - lower_bb) / (upper_bb - lower_bb).replace(0, np.nan)

    # Precio relativo a su EMA21 — por encima/debajo de la tendencia
    df["close_ema20_ratio"] = df["close"] / ema21.replace(0, np.nan)

    # Lags finos de momentum
    df["log_return_lag2"] = df["log_return"].shift(2)
    df["log_return_lag3"] = df["log_return"].shift(3)
    df["log_return_lag6"] = df["log_return"].shift(6)

    # RSI divergence: cambio del RSI en 3 barras (proxy de divergencia)
    df["rsi_divergence"] = df["rsi_14"] - df["rsi_14"].shift(3)

    # --- Features de microestructura (Lopez de Prado cap. 19) ---

    # BVC — Bulk Volume Classification (Easley et al. 2012)
    # Estima fraccion de volumen buyer-initiated usando direccion del precio
    # Z = Φ(log_return / σ) → buy_vol = Z*V, sell_vol = (1-Z)*V
    from scipy.special import ndtr as _norm_cdf
    _sigma_bvc = df["log_return"].rolling(20).std().replace(0, np.nan)
    _z         = pd.Series(_norm_cdf(df["log_return"] / _sigma_bvc), index=df.index)
    _buy_vol   = _z * df["volume"]
    _sell_vol  = (1 - _z) * df["volume"]
    _roll_v    = df["volume"].rolling(12).sum().replace(0, np.nan)
    df["bvc_imbalance"] = (_buy_vol - _sell_vol).rolling(12).sum() / _roll_v

    # Amihud Illiquidity — |retorno| / volumen en USD (rolling 12 barras = 48h)
    # Alta iliquidez = gran movimiento por poco volumen = mercado thin
    _dollar_vol = df["volumeUSD"].replace(0, np.nan)
    df["amihud_illiquidity"] = (df["log_return"].abs() / _dollar_vol).rolling(12).mean()

    # Roll's Spread — estimador de bid-ask spread implicito (rolling 20 barras)
    # spread = 2 * sqrt(max(-cov(ΔP_t, ΔP_{t-1}), 0)), normalizado por close
    _delta_c = df["close"].diff()
    _cov     = _delta_c.rolling(20).cov(_delta_c.shift(1))
    df["roll_spread"] = 2 * np.sqrt(np.maximum(-_cov, 0)) / df["close"].replace(0, np.nan)

    # Winsorizar outliers (IQR 3x)
    _winsor_cols = [
        "log_return", "transfer_count_pct_change", "volume_tvl_ratio",
        "high_low_range", "whale_volume_ratio",
        "volume_ratio", "return_skew_72h", "tvl_change_pct", "net_flow_proxy",
        "macd_histogram", "bb_width", "atr_14",
        "cci_20", "roc_10", "obv_momentum", "vwap_dev",
        "bb_pct_b", "close_ema20_ratio", "rsi_divergence",
        "bvc_imbalance", "amihud_illiquidity", "roll_spread",
    ]
    for col in _winsor_cols:
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
