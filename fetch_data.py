"""
fetch_data.py — Ingesta de datos con cache agresivo en disco.

V2: Agrega fetch_price_binance_hourly, fetch_fear_greed,
    load_dune_holders, load_thegraph_protocol.

Fuentes:
  - Binance API (precios OHLCV, sin key) — 5 anios paginado (diario + horario)
  - Fear & Greed Index (alternative.me)
  - Dune Analytics CSV (holders reales)
  - The Graph CSV (protocolo Uniswap)
  - The Graph subgraph (backup)
  - Etherscan — reconstruccion de holders desde transfers historicas

Puede correrse de forma independiente:
  python3 fetch_data.py
"""

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np
import requests

# --- Configuracion ---

ETHERSCAN_API_KEY = "JV3XC71684223B2YS8AXSN6GKJ3AA86Z2H"
ETHERSCAN_BASE_URL = "https://api.etherscan.io/api"

GRAPH_URL = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"
UNI_TOKEN = "0x1f9840a85d5af5bf1d1762f925bdaddc4201f984"

CACHE_DIR = Path(__file__).parent / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = Path(__file__).parent / "data"

UNI_DECIMALS = 18
WHALE_THRESHOLD_UNI = 50_000
DAYS = 5 * 365


# --- Utilidades de cache ---

def _meta_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(".meta.json")


def _cache_is_valid(parquet_path: Path, ttl_days: int = 1) -> bool:
    meta_path = _meta_path(parquet_path)
    if not parquet_path.exists() or not meta_path.exists():
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        downloaded_at = datetime.fromisoformat(meta["downloaded_at"])
        age = datetime.now(timezone.utc) - downloaded_at
        return age < timedelta(days=ttl_days)
    except Exception:
        return False


def _save_cache(df: pd.DataFrame, parquet_path: Path, extra_meta: dict = None):
    df.to_parquet(parquet_path, index=False)
    meta = {
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(df),
        **(extra_meta or {}),
    }
    with open(_meta_path(parquet_path), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [cache] Guardado: {parquet_path.name} ({len(df)} filas)")


def _load_cache(parquet_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    with open(_meta_path(parquet_path)) as f:
        meta = json.load(f)
    print(f"  [cache] Cargado desde disco: {parquet_path.name} "
          f"({meta['rows']} filas, descargado {meta['downloaded_at'][:10]})")
    return df


# --- Binance: precios OHLCV diario (5 anios, paginado) ---

def fetch_price_binance(symbol: str = "UNIUSDT", days: int = DAYS) -> pd.DataFrame:
    """
    Trae OHLCV diario de los ultimos days dias desde Binance.
    Pagina si necesario (limite 1000 velas por request).
    Retorna DataFrame con columnas: date, open, high, low, close, volume
    """
    cache_path = CACHE_DIR / f"binance_{symbol.lower()}_{days}d.parquet"
    if _cache_is_valid(cache_path, ttl_days=1):
        return _load_cache(cache_path)

    print(f"\n[Binance] Descargando {days} dias de {symbol} diario (paginado)...")

    url = "https://api.binance.com/api/v3/klines"
    end_time = int(time.time() * 1000)
    start_time = end_time - days * 24 * 60 * 60 * 1000

    all_rows = []
    current_start = start_time

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": "1d",
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            raw = resp.json()
        except Exception as e:
            print(f"  [Binance] Error en request: {e}")
            break

        if not raw:
            break

        all_rows.extend(raw)

        last_open_time = raw[-1][0]
        if len(raw) < 1000:
            break
        current_start = last_open_time + 24 * 60 * 60 * 1000
        time.sleep(0.2)

    if not all_rows:
        raise ValueError(f"Binance devolvio respuesta vacia para {symbol}")

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])

    df["date"] = pd.to_datetime(df["open_time"], unit="ms").dt.normalize()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[["date", "open", "high", "low", "close", "volume"]].copy()
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    print(f"  [Binance] {len(df)} filas | "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")

    _save_cache(df, cache_path)
    return df


# --- Binance: precios OHLCV horario (paginado) ---

def fetch_price_binance_hourly(symbol: str = "UNIUSDT", days: int = 1825) -> pd.DataFrame:
    """
    Trae OHLCV horario de los ultimos days dias desde Binance.
    Pagina con limite 1000 velas/request.
    Cache a data/cache/binance_uniusdt_1h_{days}d.parquet con 1-day TTL.
    Retorna DataFrame con columnas: datetime, open, high, low, close, volume
    """
    cache_path = CACHE_DIR / f"binance_{symbol.lower()}_1h_{days}d.parquet"
    if _cache_is_valid(cache_path, ttl_days=1):
        return _load_cache(cache_path)

    print(f"\n[Binance 1h] Descargando {days} dias de {symbol} horario (paginado)...")

    url = "https://api.binance.com/api/v3/klines"
    end_time = int(time.time() * 1000)
    start_time = end_time - days * 24 * 60 * 60 * 1000

    all_rows = []
    current_start = start_time
    hour_ms = 60 * 60 * 1000

    while current_start < end_time:
        params = {
            "symbol": symbol,
            "interval": "1h",
            "startTime": current_start,
            "endTime": end_time,
            "limit": 1000,
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            raw = resp.json()
        except Exception as e:
            print(f"  [Binance 1h] Error en request: {e}")
            time.sleep(2)
            break

        if not raw:
            break

        all_rows.extend(raw)

        last_open_time = raw[-1][0]
        if len(raw) < 1000:
            break
        current_start = last_open_time + hour_ms
        time.sleep(0.15)

        if len(all_rows) % 10000 == 0:
            print(f"  [Binance 1h] {len(all_rows)} velas descargadas...", end="\r")

    if not all_rows:
        raise ValueError(f"Binance 1h devolvio respuesta vacia para {symbol}")

    df = pd.DataFrame(all_rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])

    df["datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df[["datetime", "open", "high", "low", "close", "volume"]].copy()
    df = df.drop_duplicates("datetime").sort_values("datetime").reset_index(drop=True)

    print(f"\n  [Binance 1h] {len(df)} velas | "
          f"{df['datetime'].min()} -> {df['datetime'].max()}")

    _save_cache(df, cache_path)
    return df


# --- Fear & Greed Index ---

def fetch_fear_greed(limit: int = 2000) -> pd.DataFrame:
    """
    Trae el Fear & Greed Index desde alternative.me.
    Retorna DataFrame con: date, fear_greed_value, fear_greed_class
    Encoding: Extreme Fear=0, Fear=1, Neutral=2, Greed=3, Extreme Greed=4
    Cache a data/cache/fear_greed.parquet con 1-day TTL.
    """
    cache_path = CACHE_DIR / "fear_greed.parquet"
    if _cache_is_valid(cache_path, ttl_days=1):
        return _load_cache(cache_path)

    print(f"\n[FearGreed] Descargando Fear & Greed Index (limit={limit})...")

    url = f"https://api.alternative.me/fng/?limit={limit}&format=json"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"  [FearGreed] Error: {e}. Retornando DataFrame vacio.")
        return pd.DataFrame(columns=["date", "fear_greed_value", "fear_greed_class"])

    records = data.get("data", [])
    if not records:
        print("  [FearGreed] Sin datos.")
        return pd.DataFrame(columns=["date", "fear_greed_value", "fear_greed_class"])

    class_map = {
        "Extreme Fear": 0,
        "Fear": 1,
        "Neutral": 2,
        "Greed": 3,
        "Extreme Greed": 4,
    }

    rows = []
    for r in records:
        try:
            ts = int(r["timestamp"])
            date = pd.to_datetime(ts, unit="s").normalize()
            value = int(r["value"])
            cls_str = r.get("value_classification", "Neutral")
            cls_enc = class_map.get(cls_str, 2)
            rows.append({
                "date": date,
                "fear_greed_value": value,
                "fear_greed_class": cls_enc,
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    print(f"  [FearGreed] {len(df)} dias | "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")

    _save_cache(df, cache_path)
    return df


# --- Cargar CSVs reales de Dune y The Graph ---

def load_dune_holders() -> pd.DataFrame:
    """
    Carga data/dune_uni_holders.csv.
    Parse columna date. Retorna DataFrame.
    """
    csv_path = DATA_DIR / "dune_uni_holders.csv"
    if not csv_path.exists():
        print(f"  [Dune] Archivo no encontrado: {csv_path}")
        return pd.DataFrame()

    print(f"\n[Dune] Cargando {csv_path.name}...")
    df = pd.read_csv(csv_path)

    # Detectar columna de fecha
    date_col = None
    for candidate in ["date", "Date", "day", "Day", "timestamp", "time"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        # Tomar primera columna como fecha
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col], utc=True, errors="coerce")
    df[date_col] = df[date_col].dt.tz_localize(None).dt.normalize()
    df = df.rename(columns={date_col: "date"})
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    # Normalizar nombres de columnas clave
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if "holder" in cl and "number" not in cl and "count" not in cl:
            pass
        if cl in ["number_of_holders", "holders", "num_holders", "total_holders"]:
            col_map[col] = "number_of_holders"
        elif cl in ["whale_count", "whales"]:
            col_map[col] = "whale_count"
        elif cl in ["whale_balance_pct", "whale_pct", "whale_balance_percent"]:
            col_map[col] = "whale_balance_pct"
        elif cl in ["whale_balance_total", "whale_balance"]:
            col_map[col] = "whale_balance_total"
        elif cl in ["herfindahl_index", "herfindahl", "hhi"]:
            col_map[col] = "herfindahl_index"
        elif cl in ["total_supply_held", "total_supply"]:
            col_map[col] = "total_supply_held"

    if col_map:
        df = df.rename(columns=col_map)

    # Asegurar holders_growth y whale_delta_pct
    if "number_of_holders" in df.columns:
        df["number_of_holders"] = pd.to_numeric(df["number_of_holders"], errors="coerce").fillna(0)
        df["holders_growth"] = df["number_of_holders"].pct_change() * 100
    if "whale_balance_pct" in df.columns:
        df["whale_balance_pct"] = pd.to_numeric(df["whale_balance_pct"], errors="coerce").fillna(0)
        df["whale_delta_pct"] = df["whale_balance_pct"].pct_change() * 100
        df["whale_pct_delta"] = df["whale_delta_pct"]

    print(f"  [Dune] {len(df)} filas | columnas: {list(df.columns)}")
    return df


def load_thegraph_protocol() -> pd.DataFrame:
    """
    Carga data/thegraph_uni_protocol.csv.
    Parse columna date. Retorna DataFrame.
    """
    csv_path = DATA_DIR / "thegraph_uni_protocol.csv"
    if not csv_path.exists():
        print(f"  [TheGraph CSV] Archivo no encontrado: {csv_path}")
        return pd.DataFrame()

    print(f"\n[TheGraph CSV] Cargando {csv_path.name}...")
    df = pd.read_csv(csv_path)

    # Detectar columna de fecha
    date_col = None
    for candidate in ["date", "Date", "day", "Day", "timestamp", "time"]:
        if candidate in df.columns:
            date_col = candidate
            break

    if date_col is None:
        date_col = df.columns[0]

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[date_col] = df[date_col].dt.tz_localize(None).dt.normalize()
    df = df.rename(columns={date_col: "date"})
    df = df.dropna(subset=["date"])
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    # Normalizar nombres de columnas clave
    col_map = {}
    for col in df.columns:
        cl = col.lower()
        if cl in ["volumeusd", "volume_usd", "volumeusd"]:
            col_map[col] = "volumeUSD"
        elif cl in ["totalvaluelockedusd", "tvl_usd", "tvlusd", "totalvaluelocked"]:
            col_map[col] = "totalValueLockedUSD"
        elif cl in ["priceusd", "price_usd", "price"]:
            col_map[col] = "priceUSD"
        elif cl in ["feesusd", "fees_usd", "fees"]:
            col_map[col] = "feesUSD"

    if col_map:
        df = df.rename(columns=col_map)

    # Convertir numericas
    for col in ["volumeUSD", "totalValueLockedUSD", "priceUSD", "feesUSD"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    print(f"  [TheGraph CSV] {len(df)} filas | columnas: {list(df.columns)}")
    return df


# --- The Graph: tokenDayDatas para UNI (backup subgraph) ---

def fetch_graph_uniswap(token_address: str = UNI_TOKEN) -> pd.DataFrame:
    """
    Obtiene tokenDayDatas desde el subgraph oficial de Uniswap V3.
    Pagina con skip para cubrir 5 anios completos.
    Retorna DataFrame con: date, volume_usd, tvl_usd, price_usd, fees_usd, tx_count
    """
    cache_path = CACHE_DIR / "graph_uni_v3_tokenday.parquet"
    if _cache_is_valid(cache_path, ttl_days=1):
        return _load_cache(cache_path)

    print(f"\n[Graph] Descargando tokenDayDatas desde Uniswap V3 subgraph...")

    all_records = []
    skip = 0
    batch_size = 1000
    cutoff_ts = int((datetime.now() - timedelta(days=DAYS)).timestamp())
    errors = 0

    while True:
        query = """
        {
          tokenDayDatas(
            first: %d,
            skip: %d,
            orderBy: date,
            orderDirection: desc,
            where: {
              token: "%s",
              date_gt: %d
            }
          ) {
            date
            volumeUSD
            totalValueLockedUSD
            priceUSD
            feesUSD
            untrackedVolumeUSD
          }
        }
        """ % (batch_size, skip, token_address.lower(), cutoff_ts)

        try:
            resp = requests.post(
                GRAPH_URL,
                json={"query": query},
                timeout=60,
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  [Graph] Error en request (skip={skip}): {e}")
            errors += 1
            if errors >= 3:
                print("  [Graph] Demasiados errores, usando datos parciales")
                break
            time.sleep(5)
            continue

        if "errors" in data:
            print(f"  [Graph] GraphQL error: {data['errors']}")
            errors += 1
            if errors >= 3:
                break
            time.sleep(5)
            continue

        records = data.get("data", {}).get("tokenDayDatas", [])
        if not records:
            break

        all_records.extend(records)
        print(f"  [Graph] Descargados {len(all_records)} registros...", end="\r")

        if len(records) < batch_size:
            break

        skip += batch_size
        time.sleep(0.5)

    if not all_records:
        print("  [Graph] Sin datos del subgraph, usando DataFrame vacio")
        return pd.DataFrame(columns=["date", "volume_usd", "tvl_usd", "price_usd", "fees_usd", "tx_count"])

    rows = []
    for r in all_records:
        try:
            date = pd.to_datetime(int(r["date"]), unit="s").normalize()
            rows.append({
                "date": date,
                "volume_usd": float(r.get("volumeUSD") or 0),
                "tvl_usd": float(r.get("totalValueLockedUSD") or 0),
                "price_usd": float(r.get("priceUSD") or 0),
                "fees_usd": float(r.get("feesUSD") or 0),
                "tx_count": 0,
            })
        except Exception:
            continue

    df = pd.DataFrame(rows)
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)

    print(f"\n  [Graph] {len(df)} dias | "
          f"{df['date'].min().date()} -> {df['date'].max().date()}")

    _save_cache(df, cache_path)
    return df


# --- Etherscan ---

def _etherscan_get_transfers(token_address: str, page: int, offset: int = 10000) -> list:
    params = {
        "module": "account",
        "action": "tokentx",
        "contractaddress": token_address,
        "page": page,
        "offset": offset,
        "sort": "asc",
        "apikey": ETHERSCAN_API_KEY,
    }
    resp = requests.get(ETHERSCAN_BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    if data.get("status") == "1" and data.get("result"):
        return data["result"]
    return []


def fetch_etherscan_holders(token_address: str = UNI_TOKEN) -> pd.DataFrame:
    """
    Reconstruye distribucion de holders diaria desde transfers historicas.
    """
    cache_path = CACHE_DIR / "etherscan_holders_uni.parquet"
    if _cache_is_valid(cache_path, ttl_days=7):
        return _load_cache(cache_path)

    transfers_cache = CACHE_DIR / "etherscan_transfers_raw.parquet"
    cutoff_dt = datetime.now() - timedelta(days=DAYS)
    cutoff_ts = int(cutoff_dt.timestamp())

    if transfers_cache.exists():
        print("  [Etherscan] Cargando transfers desde cache...")
        df_transfers = pd.read_parquet(transfers_cache)
    else:
        print(f"\n[Etherscan] Descargando transfers historicas del token UNI...")

        all_transfers = []
        page = 1
        consecutive_empty = 0
        max_pages = 50

        while page <= max_pages:
            try:
                records = _etherscan_get_transfers(token_address, page=page, offset=10000)
            except Exception as e:
                print(f"  [Etherscan] Error pagina {page}: {e}")
                time.sleep(5)
                consecutive_empty += 1
                if consecutive_empty >= 3:
                    break
                continue

            if not records:
                consecutive_empty += 1
                if consecutive_empty >= 2:
                    break
                time.sleep(2)
                page += 1
                continue

            consecutive_empty = 0
            all_transfers.extend(records)

            newest_ts = int(records[-1].get("timeStamp", 0))
            print(f"  [Etherscan] Pagina {page}: {len(records)} transfers | "
                  f"hasta {datetime.fromtimestamp(newest_ts).date()}", end="\r")

            if len(records) < 10000:
                break

            page += 1
            time.sleep(0.25)

        print(f"\n  [Etherscan] Total transfers descargadas: {len(all_transfers)}")

        if not all_transfers:
            print("  [Etherscan] Sin transfers, usando DataFrame vacio")
            empty = pd.DataFrame(columns=["date", "number_of_holders", "whale_count",
                                           "whale_balance_total", "whale_balance_pct",
                                           "herfindahl_index", "holders_growth"])
            _save_cache(empty, cache_path)
            return empty

        df_transfers = pd.DataFrame(all_transfers)
        df_transfers["timeStamp"] = pd.to_numeric(df_transfers["timeStamp"], errors="coerce")
        df_transfers["value"] = pd.to_numeric(df_transfers["value"], errors="coerce").fillna(0)
        df_transfers["amount"] = df_transfers["value"] / (10 ** UNI_DECIMALS)
        df_transfers["date"] = pd.to_datetime(df_transfers["timeStamp"], unit="s").dt.normalize()
        df_transfers = df_transfers[df_transfers["timeStamp"] >= cutoff_ts].copy()
        df_transfers["from"] = df_transfers["from"].str.lower()
        df_transfers["to"] = df_transfers["to"].str.lower()
        df_transfers.to_parquet(transfers_cache, index=False)

    print("  [Etherscan] Reconstruyendo balances diarios...")

    NULL_ADDRESSES = {
        "0x0000000000000000000000000000000000000000",
        "0x000000000000000000000000000000000000dead",
    }

    df_transfers = df_transfers.sort_values("timeStamp").reset_index(drop=True)
    dates = pd.date_range(
        start=df_transfers["date"].min(),
        end=df_transfers["date"].max(),
        freq="D",
    )

    balances = defaultdict(float)
    daily_snapshots = []
    df_by_day = df_transfers.groupby("date")

    for current_date in dates:
        if current_date in df_by_day.groups:
            day_txs = df_by_day.get_group(current_date)
            for _, tx in day_txs.iterrows():
                frm = tx["from"]
                to = tx["to"]
                amt = float(tx["amount"])
                if frm not in NULL_ADDRESSES:
                    balances[frm] -= amt
                if to not in NULL_ADDRESSES:
                    balances[to] += amt

        pos_balances = {w: b for w, b in balances.items() if b > 0.01}
        n_holders = len(pos_balances)

        if n_holders == 0:
            daily_snapshots.append({
                "date": current_date,
                "number_of_holders": 0,
                "whale_count": 0,
                "whale_balance_total": 0.0,
                "whale_balance_pct": 0.0,
                "herfindahl_index": 0.0,
            })
            continue

        total_supply = sum(pos_balances.values())
        sorted_balances = sorted(pos_balances.values(), reverse=True)

        top_1pct_count = max(1, int(n_holders * 0.01))
        top_1pct_balances = sorted_balances[:top_1pct_count]
        whale_balances = [b for b in top_1pct_balances if b >= WHALE_THRESHOLD_UNI]
        whale_count = len(whale_balances)
        whale_total = sum(whale_balances)
        whale_pct = whale_total / total_supply if total_supply > 0 else 0.0

        ref_balances = whale_balances if whale_balances else top_1pct_balances[:10]
        ref_total = sum(ref_balances)
        if ref_total > 0 and len(ref_balances) > 0:
            shares = [b / ref_total for b in ref_balances]
            hhi = sum(s**2 for s in shares)
        else:
            hhi = 0.0

        daily_snapshots.append({
            "date": current_date,
            "number_of_holders": n_holders,
            "whale_count": whale_count,
            "whale_balance_total": whale_total,
            "whale_balance_pct": whale_pct * 100,
            "herfindahl_index": hhi,
        })

    df_holders = pd.DataFrame(daily_snapshots)
    df_holders = df_holders.sort_values("date").reset_index(drop=True)
    df_holders["holders_growth"] = df_holders["number_of_holders"].pct_change() * 100
    df_holders["whale_delta_pct"] = df_holders["whale_balance_pct"].pct_change() * 100
    df_holders["whale_pct_delta"] = df_holders["whale_delta_pct"]

    print(f"  [Etherscan] Holders reconstruidos: {len(df_holders)} dias | "
          f"Holders promedio: {df_holders['number_of_holders'].mean():.0f}")

    _save_cache(df_holders, cache_path)
    return df_holders


def fetch_holders_fallback(days: int = DAYS) -> pd.DataFrame:
    """Fallback sintetico si Etherscan falla completamente."""
    print("  [Fallback] Generando holders sinteticos basados en datos conocidos...")
    dates = pd.date_range(
        end=datetime.now().date(),
        periods=days,
        freq="D",
    )

    np.random.seed(42)
    n = len(dates)
    base = np.linspace(35000, 300000, n)
    noise = np.random.normal(0, 2000, n)
    holders = np.clip(base + noise, 1000, None).astype(int)

    whale_count = np.clip((holders * 0.005).astype(int), 50, 2000)
    whale_pct = np.clip(np.random.normal(45, 5, n), 25, 70)
    hhi = np.clip(np.random.normal(0.15, 0.05, n), 0.05, 0.5)

    df = pd.DataFrame({
        "date": dates,
        "number_of_holders": holders,
        "whale_count": whale_count,
        "whale_balance_total": whale_pct * 1_000_000_000 / 100,
        "whale_balance_pct": whale_pct,
        "herfindahl_index": hhi,
    })
    df["holders_growth"] = df["number_of_holders"].pct_change() * 100
    df["whale_delta_pct"] = df["whale_balance_pct"].pct_change() * 100
    df["whale_pct_delta"] = df["whale_delta_pct"]
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df.reset_index(drop=True)


# --- Alineacion de datasets (v1 compat) ---

def align_datasets(
    df_price: pd.DataFrame,
    df_graph: pd.DataFrame = None,
    df_holders: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Alinea precios, datos de The Graph y holders por fecha.
    Left join sobre df_price (el precio es obligatorio).
    Forward-fill de columnas on-chain.
    Dropna en columna 'close'.
    """
    print("\n[align] Alineando datasets por fecha...")

    df_price = df_price.copy()
    df_price["date"] = pd.to_datetime(df_price["date"]).dt.normalize()
    merged = df_price.sort_values("date")

    if df_graph is not None and not df_graph.empty:
        df_graph = df_graph.copy()
        df_graph["date"] = pd.to_datetime(df_graph["date"]).dt.normalize()
        merged = merged.merge(df_graph, on="date", how="left")
        print(f"  [align] The Graph: {df_graph['date'].min().date()} -> {df_graph['date'].max().date()}")

    if df_holders is not None and not df_holders.empty:
        df_holders = df_holders.copy()
        df_holders["date"] = pd.to_datetime(df_holders["date"]).dt.normalize()
        merged = merged.merge(df_holders, on="date", how="left")
        print(f"  [align] Holders: {df_holders['date'].min().date()} -> {df_holders['date'].max().date()}")

    onchain_cols = [
        c for c in merged.columns
        if c not in {"date", "open", "high", "low", "close", "volume"}
    ]
    merged[onchain_cols] = merged[onchain_cols].ffill().fillna(0)

    before = len(merged)
    merged = merged.dropna(subset=["close"])
    after = len(merged)
    if before != after:
        print(f"  [align] Eliminadas {before - after} filas sin precio")

    merged = merged.sort_values("date").reset_index(drop=True)
    print(f"  [align] Dataset final: {len(merged)} filas | "
          f"{merged['date'].min().date()} -> {merged['date'].max().date()}")
    return merged


# --- Compatibilidad ---

def fetch_contract_activity(token_address: str, force_refresh: bool = False,
                             max_pages: int = 40) -> pd.DataFrame:
    return fetch_etherscan_holders(token_address)


# --- Entry point ---

if __name__ == "__main__":
    print("=" * 60)
    print("TEST INDEPENDIENTE: fetch_data.py v2")
    print("=" * 60)

    df_hourly = fetch_price_binance_hourly("UNIUSDT", days=365)
    print(f"Horario shape: {df_hourly.shape}")
    print(df_hourly.tail(3).to_string())

    df_fg = fetch_fear_greed(limit=500)
    print(f"\nFear & Greed shape: {df_fg.shape}")

    df_dune = load_dune_holders()
    print(f"\nDune holders shape: {df_dune.shape}")

    df_tg = load_thegraph_protocol()
    print(f"\nTheGraph protocol shape: {df_tg.shape}")

    print("\n[OK] fetch_data.py v2 corrio sin errores.")
