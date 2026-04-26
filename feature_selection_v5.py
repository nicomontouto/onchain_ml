"""
feature_selection_v5.py -- Seleccion de features para pipeline v5.

Cambios vs v4:
  - Eliminadas features diarias con leakage:
      whale_balance_pct, whale_delta_pct, herfindahl_index,
      holders_growth, fear_greed_value
  - Reemplazadas por features horarias genuinas:
      transfer_count, whale_volume_ratio, unique_senders,
      transfer_count_pct_change, dollar_bar_duration
  - Todas las fuentes son horarias (Binance, The Graph, Dune, dollar bar)
  - Sin leakage al tradear en 4h
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 10 features seleccionadas por MI + dCor + Spearman
# Descartadas vs v5.0:
#   log_return          (MI=0, dCor bajo)
#   volume_tvl_ratio    (MI=0, Spearman=0.997 con feesUSD)
#   feesUSD             (Spearman=0.972 con volumeUSD_roll_std72, redundante)
#   transfer_count_pct_change (MI~0, dCor bajo)
#   price_divergence_lag1     (redundante con price_divergence)
CANDIDATE_FEATURES = [
    # Volatilidad / retorno (Binance)
    "log_return_lag1",
    "log_return_lag4",
    "volatility_24h",
    "high_low_range",
    "log_return_roll_std72",
    # Protocolo DEX (The Graph horario)
    "price_divergence",
    "volumeUSD_roll_std72",
    # Actividad on-chain (Dune horario)
    "transfer_count",
    "whale_volume_ratio",
    # Dollar bar
    "dollar_bar_duration",
]

CORR_THRESHOLD = 0.85


def _build_candidates(df: pd.DataFrame) -> tuple[pd.DataFrame, list, list]:
    out     = pd.DataFrame(index=df.index)
    used    = []
    derived = []
    missing = []

    def _try_lag(base: str, lag: int) -> bool:
        if base in df.columns:
            out[f"{base}_lag{lag}"] = df[base].shift(lag)
            return True
        return False

    def _try_roll_std(base: str, w: int) -> bool:
        if base in df.columns:
            out[f"{base}_roll_std{w}"] = df[base].shift(1).rolling(w).std()
            return True
        return False

    for col in CANDIDATE_FEATURES:
        if col in df.columns:
            out[col] = df[col]
            used.append(col)
            continue

        # Derivar si la base existe
        if col == "log_return_lag1"        and _try_lag("log_return", 1):
            used.append(col); derived.append(col); continue
        if col == "log_return_lag4"        and _try_lag("log_return", 4):
            used.append(col); derived.append(col); continue
        if col == "price_divergence_lag1"  and _try_lag("price_divergence", 1):
            used.append(col); derived.append(col); continue
        if col == "volumeUSD_roll_std72"   and _try_roll_std("volumeUSD", 18):
            # 18 barras de 4h = 72h
            out = out.rename(columns={"volumeUSD_roll_std18": "volumeUSD_roll_std72"})
            used.append(col); derived.append(col); continue
        if col == "log_return_roll_std72"  and _try_roll_std("log_return", 18):
            out = out.rename(columns={"log_return_roll_std18": "log_return_roll_std72"})
            used.append(col); derived.append(col); continue

        missing.append(col)

    if missing:
        print(f"  [features-v5] candidatas ausentes: {missing}")

    return out, used, derived


def _mean_abs_corr_with_rest(corr: pd.DataFrame, col: str, cols: list) -> float:
    others = [c for c in cols if c != col]
    if not others:
        return 0.0
    return float(np.abs(corr.loc[col, others]).mean())


def filter_correlated(
    df_candidates: pd.DataFrame,
    threshold: float = CORR_THRESHOLD,
) -> tuple[list, pd.DataFrame, list[dict]]:
    cols  = list(df_candidates.columns)
    corr  = df_candidates.dropna().corr(method="pearson")
    elims = []

    while True:
        upper   = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        max_val = upper.abs().stack().max() if not upper.abs().stack().empty else 0.0

        if pd.isna(max_val) or max_val <= threshold:
            break

        max_pair = upper.abs().stack().idxmax()
        a, b     = max_pair
        raw_corr = float(corr.loc[a, b])

        mean_a = _mean_abs_corr_with_rest(corr, a, cols)
        mean_b = _mean_abs_corr_with_rest(corr, b, cols)

        if mean_a < mean_b:
            drop, keep = a, b
        elif mean_b < mean_a:
            drop, keep = b, a
        else:
            idx_a = CANDIDATE_FEATURES.index(a) if a in CANDIDATE_FEATURES else 999
            idx_b = CANDIDATE_FEATURES.index(b) if b in CANDIDATE_FEATURES else 999
            drop, keep = (b, a) if idx_a <= idx_b else (a, b)

        elims.append({
            "dropped":               drop,
            "kept":                  keep,
            "pair_corr":             round(raw_corr, 4),
            "mean_abs_corr_dropped": round(mean_a if drop == a else mean_b, 4),
            "mean_abs_corr_kept":    round(mean_b if drop == a else mean_a, 4),
            "reason": (
                f"|corr({a},{b})|={abs(raw_corr):.3f} > {threshold} -- "
                f"se elimina {drop} (menor corr promedio con el resto)"
            ),
        })

        cols.remove(drop)
        corr = corr.drop(index=drop, columns=drop)

    return cols, corr, elims


def select_features(df: pd.DataFrame, threshold: float = CORR_THRESHOLD) -> dict:
    """
    Pipeline completo de seleccion de features v5.
    Retorna dict con candidatas, matrices de corr y features finales.
    """
    df_cand, used, derived = _build_candidates(df)
    df_cand = df_cand.dropna(axis=1, how="all")

    corr_initial = df_cand.dropna().corr(method="pearson")
    final, corr_final, elims = filter_correlated(df_cand, threshold)

    print(f"  [features-v5] candidatas iniciales: {len(df_cand.columns)} | "
          f"finales: {len(final)}")
    for e in elims:
        print(f"    - elim: {e['dropped']} (|r|={abs(e['pair_corr']):.3f} vs {e['kept']})")

    return {
        "df_candidates":       df_cand,
        "candidate_features":  list(df_cand.columns),
        "derived_features":    derived,
        "corr_matrix_initial": corr_initial,
        "final_features":      final,
        "corr_matrix_final":   corr_final,
        "eliminations":        elims,
    }
