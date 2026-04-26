"""
triple_barrier_v4.py -- Triple Barrier Method (Lopez de Prado, AFML).

Implementacion fiel al pseudocodigo del libro, adaptada a pandas moderno
(iteritems -> items). Genera labels {-1, 0, +1}:
  +1  profit-taking tocado primero
  -1  stop-loss tocado primero
   0  expiro la barrera temporal sin tocar ninguna

Parametros por defecto del pipeline v4:
  ptSl = [1, 1] (barreras simetricas)
  side = 1 (neutral, sin direccion previa)
  trgt = rolling(20).std() del log_return del close como retorno %
  t1   = 12 velas (4h -> 48h) | 5 velas (daily -> 5 dias)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

T1_HORIZON = {
    "4h":    12,   # 12 velas de 4h = 48h (2 dias)
    "daily":  5,   # 5 velas diarias
}


# ---------------------------------------------------------------------------
# Volatilidad dinamica (target de las barreras, expresada como retorno %)
# ---------------------------------------------------------------------------

def get_daily_vol(close: pd.Series, span: int = 20) -> pd.Series:
    """
    Volatilidad rolling std del retorno simple close/close.shift - 1.
    shift(1) antes del rolling para evitar lookahead (vol[t] no usa ret[t]).
    """
    ret = close.pct_change()
    vol = ret.shift(1).rolling(span).std()
    return vol


# ---------------------------------------------------------------------------
# Filtro CUSUM simetrico (Lopez de Prado, AFML cap. 2)
# ---------------------------------------------------------------------------

def get_cusum_events(close: pd.Series, h: float) -> pd.DatetimeIndex:
    """
    Filtra eventos significativos usando el CUSUM simetrico.
    Solo genera un evento cuando la suma acumulada de retornos supera
    el umbral h (positivo o negativo).
    h = trgt.mean() tipicamente (volatilidad promedio).
    """
    t_events, s_pos, s_neg = [], 0.0, 0.0
    diff = close.pct_change().dropna()
    for i in diff.index:
        s_pos = max(0.0, s_pos + diff[i])
        s_neg = min(0.0, s_neg + diff[i])
        if s_neg < -h:
            s_neg = 0.0
            t_events.append(i)
        elif s_pos > h:
            s_pos = 0.0
            t_events.append(i)
    return pd.DatetimeIndex(t_events)


# ---------------------------------------------------------------------------
# Paso 1 del Triple Barrier: tocar pt / sl antes de t1
# ---------------------------------------------------------------------------

def applyPtSlOnT1(
    close: pd.Series,
    events: pd.DataFrame,
    ptSl: list,
    molecule: list,
) -> pd.DataFrame:
    """
    Replica del snippet de Lopez de Prado. Para cada evento en `molecule`:
      - Calcula barrera superior pt = ptSl[0] * trgt (si ptSl[0] > 0)
      - Calcula barrera inferior sl = -ptSl[1] * trgt (si ptSl[1] > 0)
      - Busca el primer timestamp en [loc, t1] donde el retorno
        ajustado por side cruza cada barrera.

    Adaptaciones a pandas moderno:
      - events_['t1'].items() en lugar de iteritems()
    """
    events_ = events.loc[molecule]
    out = events_[["t1"]].copy(deep=True)

    if ptSl[0] > 0:
        pt = ptSl[0] * events_["trgt"]
    else:
        pt = pd.Series(index=events.index, dtype=float)

    if ptSl[1] > 0:
        sl = -ptSl[1] * events_["trgt"]
    else:
        sl = pd.Series(index=events.index, dtype=float)

    # iteritems -> items (compat pandas >= 1.5)
    for loc, t1 in events_["t1"].fillna(close.index[-1]).items():
        df0 = close[loc:t1]
        df0 = (df0 / close[loc] - 1.0) * events_.at[loc, "side"]
        sl_hit = df0[df0 < sl[loc]].index.min() if pd.notna(sl[loc]) else pd.NaT
        pt_hit = df0[df0 > pt[loc]].index.min() if pd.notna(pt[loc]) else pd.NaT
        out.loc[loc, "sl"] = sl_hit
        out.loc[loc, "pt"] = pt_hit

    return out


# ---------------------------------------------------------------------------
# Armar events (tIn, t1, trgt, side) y obtener labels
# ---------------------------------------------------------------------------

def add_vertical_barrier(
    t_events: pd.DatetimeIndex,
    close: pd.Series,
    num_bars: int,
) -> pd.Series:
    """
    t1 = timestamp ubicado `num_bars` filas despues de cada evento.
    Si la proyeccion excede el final de la serie, queda NaT.
    """
    idx = close.index
    t1_list = []
    positions = idx.get_indexer(t_events)
    for pos in positions:
        target = pos + num_bars
        if 0 <= pos and target < len(idx):
            t1_list.append(idx[target])
        else:
            t1_list.append(pd.NaT)
    return pd.Series(t1_list, index=t_events)


def get_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    ptSl: list,
    trgt: pd.Series,
    min_ret: float = 0.0,
    num_bars: int | None = None,
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Construye el DataFrame de eventos con columnas: t1, trgt, side.
    Aplica applyPtSlOnT1 para encontrar la primera barrera tocada.
    """
    # Alinear trgt y filtrar por min_ret
    trgt = trgt.reindex(t_events).dropna()
    trgt = trgt[trgt > min_ret]
    if trgt.empty:
        return pd.DataFrame(columns=["t1", "trgt", "side", "sl", "pt"])

    # Barrera temporal
    if num_bars is None:
        t1 = pd.Series(pd.NaT, index=trgt.index)
    else:
        t1 = add_vertical_barrier(trgt.index, close, num_bars)

    # Side: por defecto 1 (meta-labels neutros)
    if side is None:
        side_ = pd.Series(1.0, index=trgt.index)
    else:
        side_ = side.reindex(trgt.index).fillna(1.0)

    events = pd.concat({
        "t1":   t1,
        "trgt": trgt,
        "side": side_,
    }, axis=1).dropna(subset=["trgt"])

    # Para eventos sin t1 (proyeccion fuera del indice), usamos el ultimo ts
    # durante la busqueda dentro de applyPtSlOnT1 (ya lo hace con fillna).
    out = applyPtSlOnT1(close, events, ptSl, events.index)
    events["sl"] = out["sl"]
    events["pt"] = out["pt"]
    return events


# ---------------------------------------------------------------------------
# Labels finales
# ---------------------------------------------------------------------------

def get_bins(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """
    A partir del events de Triple Barrier, produce:
      - t_end: timestamp en que se cerro el evento (min entre sl, pt, t1)
      - ret: retorno realizado (close_end / close_start - 1) * side
      - bin: label {-1, 0, +1}
          +1 si tocó pt primero
          -1 si tocó sl primero
           0 si expiró t1 sin tocar barreras
    """
    rows = []
    for loc, ev in events.iterrows():
        t1 = ev["t1"] if pd.notna(ev["t1"]) else close.index[-1]
        sl = ev.get("sl", pd.NaT)
        pt = ev.get("pt", pd.NaT)

        candidates = {"t1": t1}
        if pd.notna(sl):
            candidates["sl"] = sl
        if pd.notna(pt):
            candidates["pt"] = pt
        first = min(candidates, key=lambda k: candidates[k])
        t_end = candidates[first]

        ret = (close.loc[t_end] / close.loc[loc] - 1.0) * ev["side"]

        if first == "pt":
            label = 1
        elif first == "sl":
            label = -1
        else:
            label = 0

        rows.append({
            "t_start": loc,
            "t_end":   t_end,
            "first":   first,
            "ret":     ret,
            "bin":     label,
        })

    return pd.DataFrame(rows).set_index("t_start")


# ---------------------------------------------------------------------------
# Pipeline de alto nivel para v4
# ---------------------------------------------------------------------------

def build_labels_triple_barrier(
    df: pd.DataFrame,
    freq: str,
    ptSl: list | None = None,
    vol_span: int = 20,
    min_ret: float = 0.0,
) -> pd.DataFrame:
    """
    Entry point del pipeline v4.

    df debe tener un indice temporal y columna 'close'.
    Si df tiene columna datetime/date y no indice temporal, se usa esa col.

    Retorna df con columnas:
      t_end, first, ret, bin, trgt, t1, sl, pt
    alineadas al indice original (t_start).
    """
    if ptSl is None:
        ptSl = [2.0, 2.0]

    # Garantizar indice temporal
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        dc = "datetime" if "datetime" in df.columns else ("date" if "date" in df.columns else None)
        if dc is None:
            raise ValueError("df sin indice temporal ni columna datetime/date")
        df[dc] = pd.to_datetime(df[dc])
        df = df.set_index(dc).sort_index()

    close = df["close"].astype(float)
    num_bars = T1_HORIZON.get(freq)
    if num_bars is None:
        raise ValueError(f"freq '{freq}' no soportada (usar 4h o daily)")

    trgt = get_daily_vol(close, span=vol_span)

    # Filtro CUSUM: solo eventos donde la acumulacion supera la vol promedio
    h = float(trgt.dropna().mean())
    t_events = get_cusum_events(close, h=h)
    # Quedarse solo con los que tienen trgt valida
    t_events = t_events[t_events.isin(trgt.dropna().index)]

    print(f"  [triple-barrier] freq={freq} | eventos CUSUM={len(t_events)} "
          f"(de {len(trgt.dropna())} totales) | num_bars t1={num_bars} | ptSl={ptSl}")

    events = get_events(
        close=close,
        t_events=t_events,
        ptSl=ptSl,
        trgt=trgt,
        min_ret=min_ret,
        num_bars=num_bars,
        side=None,
    )
    if events.empty:
        print("  [triple-barrier] WARN: sin eventos generados")
        return pd.DataFrame()

    bins = get_bins(events, close)
    merged = events.join(bins, how="inner")

    counts = merged["bin"].value_counts().to_dict()
    total = len(merged)
    print(f"  [triple-barrier] labels: "
          f"+1={counts.get(1, 0)} ({100*counts.get(1,0)/max(total,1):.1f}%) | "
          f"0={counts.get(0, 0)} ({100*counts.get(0,0)/max(total,1):.1f}%) | "
          f"-1={counts.get(-1, 0)} ({100*counts.get(-1,0)/max(total,1):.1f}%)")

    return merged
