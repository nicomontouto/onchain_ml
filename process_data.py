"""
process_data.py — Limpieza, metricas derivadas y seniales whale.

V2: Agrega build_multi_freq_datasets para 1h, 4h y daily.

Fuentes de columnas esperadas:
  - Binance hourly: datetime, open, high, low, close, volume
  - Dune holders: date, number_of_holders, whale_count, whale_balance_pct, ...
  - TheGraph protocol: date, volumeUSD, totalValueLockedUSD, priceUSD, feesUSD
  - Fear & Greed: date, fear_greed_value, fear_greed_class

Puede correrse de forma independiente:
  python3 process_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# --- Metricas de precio ---

def add_price_metrics(df: pd.DataFrame, periods: int = 24) -> pd.DataFrame:
    """
    log_return, pct_return, price_ma24 (o price_ma7 para daily),
    price_ma168, volatility_24h, high_low_range, volume_ma24
    """
    df = df.copy()

    date_col = "datetime" if "datetime" in df.columns else "date"
    df = df.sort_values(date_col).reset_index(drop=True)

    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["pct_return"] = df["close"].pct_change()

    # Moving averages (period-agnostic: use 24 and 168 for sub-daily)
    p1 = periods       # 24 for 1h, 6 for 4h, 7 for daily
    p2 = periods * 7   # 168 for 1h, 42 for 4h, 30 for daily

    df[f"price_ma{p1}"] = df["close"].rolling(p1).mean()
    df[f"price_ma{p2}"] = df["close"].rolling(p2).mean()

    # Alias for downstream code that expects price_ma24 / price_ma168
    if f"price_ma{p1}" != "price_ma24":
        df["price_ma24"] = df[f"price_ma{p1}"]
    if f"price_ma{p2}" != "price_ma168":
        df["price_ma168"] = df[f"price_ma{p2}"]

    df["volatility_24h"] = df["log_return"].rolling(p1).std()
    df["high_low_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["volume_ma24"] = df["volume"].rolling(p1).mean()

    # Legacy aliases
    df["price_ma7"] = df["close"].rolling(7).mean()
    df["price_ma30"] = df["close"].rolling(30).mean()

    return df


# --- Metricas de The Graph ---

def add_graph_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deriva metricas de protocolo desde columnas de The Graph:
    volume_tvl_ratio, fee_revenue, price_divergence
    """
    df = df.copy()

    # Soporte para columnas con nombre mayuscula (TheGraph CSV) o minuscula (subgraph)
    vol_col = "volumeUSD" if "volumeUSD" in df.columns else "volume_usd"
    tvl_col = "totalValueLockedUSD" if "totalValueLockedUSD" in df.columns else "tvl_usd"
    fee_col = "feesUSD" if "feesUSD" in df.columns else "fees_usd"
    price_col = "priceUSD" if "priceUSD" in df.columns else "price_usd"

    if vol_col in df.columns and tvl_col in df.columns:
        tvl_pos = df[tvl_col].replace(0, np.nan)
        df["volume_tvl_ratio"] = df[vol_col] / tvl_pos
        df["volume_tvl_ratio"] = df["volume_tvl_ratio"].fillna(0)

    if fee_col in df.columns and vol_col in df.columns:
        vol_pos = df[vol_col].replace(0, np.nan)
        df["fee_revenue"] = df[fee_col] / vol_pos
        df["fee_revenue"] = df["fee_revenue"].fillna(0)

    if vol_col in df.columns:
        df["volume_usd"] = df[vol_col]
        df["volume_usd_delta"] = df[vol_col].pct_change() * 100

    if tvl_col in df.columns:
        df["tvl_usd"] = df[tvl_col]
        df["tvl_usd_delta"] = df[tvl_col].pct_change() * 100

    if fee_col in df.columns:
        df["fees_usd"] = df[fee_col]

    # price_divergence: abs(close - priceUSD) / priceUSD
    if price_col in df.columns and "close" in df.columns:
        price_ref = df[price_col].replace(0, np.nan)
        df["price_divergence"] = (df["close"] - price_ref).abs() / price_ref
        df["price_divergence"] = df["price_divergence"].fillna(0)

    return df


# --- Metricas on-chain derivadas ---

def add_onchain_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega metricas de variacion diaria y seniales whale.
    """
    df = df.copy()

    date_col = "datetime" if "datetime" in df.columns else "date"
    df = df.sort_values(date_col).reset_index(drop=True)

    if "number_of_holders" in df.columns:
        if "holders_growth" not in df.columns or df["holders_growth"].isna().all():
            df["holders_growth"] = df["number_of_holders"].pct_change() * 100

    if "whale_balance_pct" in df.columns:
        if "whale_delta_pct" not in df.columns or df["whale_delta_pct"].isna().all():
            df["whale_delta_pct"] = df["whale_balance_pct"].pct_change() * 100
        if "whale_pct_delta" not in df.columns:
            df["whale_pct_delta"] = df["whale_delta_pct"]

    if "herfindahl_index" in df.columns:
        max_h = df["herfindahl_index"].max()
        if max_h > 0:
            df["herfindahl_index"] = df["herfindahl_index"] / max_h

    return df


# --- Winzorizacion por IQR ---

def remove_outliers_iqr(df: pd.DataFrame, cols: list, k: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        before_min, before_max = df[col].min(), df[col].max()
        df[col] = df[col].clip(lower=lower, upper=upper)
        clipped = ((df[col] == lower) | (df[col] == upper)).sum()
        if clipped > 0:
            print(f"  [iqr] {col}: {clipped} valores clipeados")
    return df


# --- Pipeline principal v1 ---

def process_pipeline(df_aligned: pd.DataFrame) -> pd.DataFrame:
    """
    Pipeline completo de procesamiento v1 (compatible).
    """
    print("\n[process] Iniciando pipeline de procesamiento...")

    df = df_aligned.copy()
    df = add_price_metrics(df, periods=7)  # daily: 7 periodos = semana
    df = add_graph_metrics(df)
    df = add_onchain_metrics(df)

    winsor_cols = [
        c for c in [
            "log_return", "whale_delta_pct", "holders_growth",
            "volume_usd_delta", "tvl_usd_delta", "volume_tvl_ratio",
            "fee_revenue", "activity_delta_pct", "volume_delta_pct",
        ]
        if c in df.columns
    ]
    if winsor_cols:
        df = remove_outliers_iqr(df, winsor_cols, k=3.0)

    print(f"  [process] Dataset procesado: {df.shape}")
    return df


# --- Pipeline v2: multi-frecuencia ---

def build_multi_freq_datasets(
    df_hourly: pd.DataFrame,
    df_holders: pd.DataFrame,
    df_protocol: pd.DataFrame,
    df_fg: pd.DataFrame,
) -> dict:
    """
    Construye datasets para 3 frecuencias: '1h', '4h', 'daily'.

    Para '1h': df_hourly + datos diarios repetidos por hora
    Para '4h': resample horario a 4h + datos diarios
    Para 'daily': resample horario a diario + todos los datos diarios

    Merge: para cada timestamp horario/4h, busca la fecha calendario
    y hace left join con datos diarios (holders, protocol, fear&greed).

    Despues del merge:
    - Forward-fill columnas on-chain
    - Dropna donde close es NaN
    - Agrega columnas derivadas de precio, holders, protocol
    """
    print("\n[process v2] Construyendo datasets multi-frecuencia...")

    # --- Preparar datos diarios ---
    daily_dfs = []

    if df_holders is not None and not df_holders.empty:
        dh = df_holders.copy()
        dh["date"] = pd.to_datetime(dh["date"]).dt.normalize()
        daily_dfs.append(("holders", dh))
        print(f"  [process v2] Holders: {len(dh)} filas")

    if df_protocol is not None and not df_protocol.empty:
        dp = df_protocol.copy()
        dp["date"] = pd.to_datetime(dp["date"]).dt.normalize()
        daily_dfs.append(("protocol", dp))
        print(f"  [process v2] Protocol: {len(dp)} filas")

    if df_fg is not None and not df_fg.empty:
        dfg = df_fg.copy()
        dfg["date"] = pd.to_datetime(dfg["date"]).dt.normalize()
        daily_dfs.append(("fear_greed", dfg))
        print(f"  [process v2] Fear&Greed: {len(dfg)} filas")

    def merge_daily_data(df_base: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
        """Merge datos diarios sobre un dataframe con columna de fecha."""
        df_base = df_base.copy()
        df_base["_date"] = pd.to_datetime(df_base[datetime_col]).dt.normalize()

        for name, ddf in daily_dfs:
            # Evitar columnas duplicadas excepto la de join
            overlap = [c for c in ddf.columns if c in df_base.columns and c != "date"]
            if overlap:
                ddf = ddf.drop(columns=overlap)
            df_base = pd.merge_asof(
                df_base.sort_values("_date"),
                ddf.sort_values("date").rename(columns={"date": "_date_right"}),
                left_on="_date",
                right_on="_date_right",
                direction="backward",
            )
            if "_date_right" in df_base.columns:
                df_base = df_base.drop(columns=["_date_right"])

        df_base = df_base.drop(columns=["_date"])
        return df_base

    def add_derived_cols(df: pd.DataFrame, periods: int = 24) -> pd.DataFrame:
        """Agrega columnas derivadas de precio, holders, protocol y seniales whale."""
        df = df.copy()

        date_col = "datetime" if "datetime" in df.columns else "date"
        df = df.sort_values(date_col).reset_index(drop=True)

        # Precio
        if "close" in df.columns:
            df["log_return"] = np.log(df["close"] / df["close"].shift(1))
            df["pct_return"] = df["close"].pct_change()
            df["price_ma24"] = df["close"].rolling(periods).mean()
            df["price_ma168"] = df["close"].rolling(periods * 7).mean()
            df["volatility_24h"] = df["log_return"].rolling(periods).std()
            df["volume_ma24"] = df["volume"].rolling(periods).mean() if "volume" in df.columns else np.nan
            if "high" in df.columns and "low" in df.columns:
                df["high_low_range"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)

        # Holders
        if "number_of_holders" in df.columns:
            nh = df["number_of_holders"].replace(0, np.nan)
            if "holders_growth" not in df.columns or df["holders_growth"].isna().all():
                df["holders_growth"] = nh.pct_change() * 100
        if "whale_balance_pct" in df.columns:
            if "whale_delta_pct" not in df.columns or df["whale_delta_pct"].isna().all():
                df["whale_delta_pct"] = df["whale_balance_pct"].pct_change() * 100
            if "whale_pct_delta" not in df.columns:
                df["whale_pct_delta"] = df["whale_delta_pct"]

        # Protocol
        vol_col = "volumeUSD" if "volumeUSD" in df.columns else ("volume_usd" if "volume_usd" in df.columns else None)
        tvl_col = "totalValueLockedUSD" if "totalValueLockedUSD" in df.columns else ("tvl_usd" if "tvl_usd" in df.columns else None)
        fee_col = "feesUSD" if "feesUSD" in df.columns else ("fees_usd" if "fees_usd" in df.columns else None)
        price_ref_col = "priceUSD" if "priceUSD" in df.columns else ("price_usd" if "price_usd" in df.columns else None)

        if vol_col and tvl_col:
            tvl_pos = df[tvl_col].replace(0, np.nan)
            df["volume_tvl_ratio"] = (df[vol_col] / tvl_pos).fillna(0)
            df["volume_usd"] = df[vol_col]
            df["tvl_usd"] = df[tvl_col]

        if fee_col and vol_col:
            vol_pos = df[vol_col].replace(0, np.nan)
            df["fee_revenue"] = (df[fee_col] / vol_pos).fillna(0)
            df["fees_usd"] = df[fee_col]

        if price_ref_col and "close" in df.columns:
            pr = df[price_ref_col].replace(0, np.nan)
            df["price_divergence"] = ((df["close"] - pr).abs() / pr).fillna(0)

        # Seniales whale
        if "log_return" in df.columns and "whale_delta_pct" in df.columns:
            has_wd = df["whale_delta_pct"].abs().sum() > 0
            price_up = df["log_return"] > 0
            price_down = df["log_return"] < 0
            if has_wd:
                whale_inc = df["whale_delta_pct"] > 0
                whale_dec = df["whale_delta_pct"] < 0
                df["whale_accumulation"] = (price_down & whale_inc).astype(int)
                df["whale_distribution"] = (price_up & whale_dec).astype(int)
                df["whale_confluence"] = (price_up & whale_inc).astype(int)
                if "holders_growth" in df.columns and df["holders_growth"].abs().sum() > 0:
                    df["sell_pressure"] = ((df["holders_growth"] < 0) & whale_dec).astype(int)
                else:
                    df["sell_pressure"] = 0
            else:
                df["whale_accumulation"] = 0
                df["whale_distribution"] = 0
                df["whale_confluence"] = 0
                df["sell_pressure"] = 0

        return df

    results = {}

    # ===== 1h =====
    print("  [process v2] Construyendo dataset 1h...")
    df_1h = df_hourly.copy()
    if "datetime" not in df_1h.columns and "date" in df_1h.columns:
        df_1h = df_1h.rename(columns={"date": "datetime"})
    df_1h["datetime"] = pd.to_datetime(df_1h["datetime"])
    df_1h = df_1h.sort_values("datetime").reset_index(drop=True)
    df_1h = merge_daily_data(df_1h, "datetime")

    # Forward-fill on-chain columns
    onchain_1h = [c for c in df_1h.columns if c not in {"datetime", "open", "high", "low", "close", "volume"}]
    df_1h[onchain_1h] = df_1h[onchain_1h].ffill().fillna(0)
    df_1h = df_1h.dropna(subset=["close"]).reset_index(drop=True)
    df_1h = add_derived_cols(df_1h, periods=24)
    results["1h"] = df_1h
    print(f"  [process v2] 1h: {df_1h.shape}")

    # ===== 4h =====
    print("  [process v2] Construyendo dataset 4h...")
    df_4h_base = df_hourly.copy()
    if "datetime" not in df_4h_base.columns and "date" in df_4h_base.columns:
        df_4h_base = df_4h_base.rename(columns={"date": "datetime"})
    df_4h_base["datetime"] = pd.to_datetime(df_4h_base["datetime"])
    df_4h_base = df_4h_base.set_index("datetime")

    df_4h = df_4h_base[["open", "high", "low", "close", "volume"]].resample("4h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"]).reset_index()

    df_4h = merge_daily_data(df_4h, "datetime")
    onchain_4h = [c for c in df_4h.columns if c not in {"datetime", "open", "high", "low", "close", "volume"}]
    df_4h[onchain_4h] = df_4h[onchain_4h].ffill().fillna(0)
    df_4h = df_4h.dropna(subset=["close"]).reset_index(drop=True)
    df_4h = add_derived_cols(df_4h, periods=6)
    results["4h"] = df_4h
    print(f"  [process v2] 4h: {df_4h.shape}")

    # ===== daily =====
    print("  [process v2] Construyendo dataset daily...")
    df_daily_base = df_hourly.copy()
    if "datetime" not in df_daily_base.columns and "date" in df_daily_base.columns:
        df_daily_base = df_daily_base.rename(columns={"date": "datetime"})
    df_daily_base["datetime"] = pd.to_datetime(df_daily_base["datetime"])
    df_daily_base = df_daily_base.set_index("datetime")

    df_daily = df_daily_base[["open", "high", "low", "close", "volume"]].resample("1D").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["close"]).reset_index()
    df_daily = df_daily.rename(columns={"datetime": "date"})

    # Merge daily data directamente en date
    df_daily["date"] = pd.to_datetime(df_daily["date"]).dt.normalize()

    for name, ddf in daily_dfs:
        ddf2 = ddf.copy()
        overlap = [c for c in ddf2.columns if c in df_daily.columns and c != "date"]
        if overlap:
            ddf2 = ddf2.drop(columns=overlap)
        df_daily = pd.merge_asof(
            df_daily.sort_values("date"),
            ddf2.sort_values("date"),
            on="date",
            direction="backward",
        )

    onchain_daily = [c for c in df_daily.columns if c not in {"date", "open", "high", "low", "close", "volume"}]
    df_daily[onchain_daily] = df_daily[onchain_daily].ffill().fillna(0)
    df_daily = df_daily.dropna(subset=["close"]).reset_index(drop=True)
    df_daily = add_derived_cols(df_daily, periods=7)
    results["daily"] = df_daily
    print(f"  [process v2] daily: {df_daily.shape}")

    return results


# --- Entry point ---

if __name__ == "__main__":
    from fetch_data import (fetch_price_binance, fetch_graph_uniswap,
                            fetch_etherscan_holders, fetch_holders_fallback,
                            align_datasets, DAYS, UNI_TOKEN)

    UNI_ADDRESS = "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984"

    print("=" * 60)
    print("TEST INDEPENDIENTE: process_data.py")
    print("=" * 60)

    df_price = fetch_price_binance("UNIUSDT", days=DAYS)
    df_graph = fetch_graph_uniswap(UNI_TOKEN)
    try:
        df_holders = fetch_etherscan_holders(UNI_ADDRESS)
    except Exception as e:
        print(f"Etherscan fallo: {e}, usando fallback")
        df_holders = fetch_holders_fallback()

    df_aligned = align_datasets(df_price, df_graph, df_holders)
    df_processed = process_pipeline(df_aligned)

    out_path = PROCESSED_DIR / "processed_data.parquet"
    df_processed.to_parquet(out_path, index=False)
    print(f"\n[OK] Guardado en {out_path}")
    print(df_processed.tail(5).to_string())
