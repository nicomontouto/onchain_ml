"""
vol_regime_analysis.py — Analisis por regimen de volatilidad en el test set.

Divide el test set en regimen de alta y baja volatilidad (mediana de vol rolling)
y re-corre la simulacion para cada regimen con LGBM y XGB.
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))


# ---------------------------------------------------------------------------
# 1. Reconstruir el test set exactamente igual que main_v6
# ---------------------------------------------------------------------------

def build_test_set():
    from process_data_v5 import build_dataset_4h
    from triple_barrier_v4 import build_labels_triple_barrier
    from meta_labeling_v6 import SFS_CANDIDATE_FEATURES

    print("[1] Cargando dataset...")
    df_raw = build_dataset_4h()

    df_idx = df_raw.copy()
    df_idx["datetime"] = pd.to_datetime(df_idx["datetime"])
    df_idx = df_idx.set_index("datetime").sort_index()

    side_series = df_idx["ema_signal"] if "ema_signal" in df_idx.columns else None

    print("[2] Triple Barrier labels...")
    events = build_labels_triple_barrier(df_idx, freq="4h", side_series=side_series)

    labels = events["bin"].astype(int).rename("label")
    cols = [c for c in SFS_CANDIDATE_FEATURES if c in df_idx.columns]
    extra = ["close"]
    if "ema_signal" in df_idx.columns:
        extra.append("ema_signal")
    if "fear_greed_value" in df_idx.columns:
        extra.append("fear_greed_value")

    joined = df_idx[cols + extra].join(labels, how="inner").dropna(subset=["label"])
    joined = joined.reset_index()
    if "index" in joined.columns:
        joined = joined.rename(columns={"index": "datetime"})
    joined["datetime"] = pd.to_datetime(joined["datetime"])

    # Split 80/20 temporal
    idx = int(len(joined) * 0.8)
    train = joined.iloc[:idx].copy().reset_index(drop=True)
    test  = joined.iloc[idx:].copy().reset_index(drop=True)

    print(f"[3] Train={len(train)} | Test={len(test)}")
    print(f"    Test: {test['datetime'].min().date()} -> {test['datetime'].max().date()}")
    return train, test


# ---------------------------------------------------------------------------
# 2. Entrenar modelos (mismo setup que main_v6)
# ---------------------------------------------------------------------------

def train_models(train, test, features_m1, features_m2):
    from meta_labeling_v6 import MetaPipeline

    val_start = int(len(train) * 0.8)
    df_tr  = train.iloc[:val_start]
    df_val = train.iloc[val_start:]
    y_tr   = train["label"].values[:val_start].astype(int)
    y_val  = train["label"].values[val_start:].astype(int)

    pipelines = {}
    for model_type in ["lgbm", "xgb"]:
        print(f"[train] {model_type.upper()}...")
        p = MetaPipeline(model_type=model_type,
                         features_m1=features_m1,
                         features_m2=features_m2)
        p.fit(df_tr, y_tr, df_val, y_val)
        pipelines[model_type] = p
    return pipelines


# ---------------------------------------------------------------------------
# 3. Regimenes de volatilidad
# ---------------------------------------------------------------------------

def assign_vol_regime(test: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    """
    Volatilidad rolling sobre el close del test set (std del log-return).
    window=24 barras de 4h = 4 dias.
    Regimen: "HIGH" si vol >= mediana, "LOW" si vol < mediana.
    """
    df = test.copy()
    log_ret = np.log(df["close"] / df["close"].shift(1))
    df["vol_rolling"] = log_ret.rolling(window, min_periods=window // 2).std()
    median_vol = df["vol_rolling"].median()
    df["regime"] = np.where(df["vol_rolling"] >= median_vol, "HIGH", "LOW")
    print(f"\n[vol] Mediana volatilidad: {median_vol:.5f}")
    print(f"[vol] HIGH: {(df['regime']=='HIGH').sum()} eventos | "
          f"LOW: {(df['regime']=='LOW').sum()} eventos")
    return df, median_vol


# ---------------------------------------------------------------------------
# 4. Simular en subconjunto (replica simulate_v4 simplificada)
# ---------------------------------------------------------------------------

def simulate_subset(df_sub: pd.DataFrame, proba_sub: np.ndarray,
                    model_kind: str,
                    initial_capital: float = 10_000.0,
                    trade_size: float = 500.0,
                    fee_rate: float = 0.001) -> dict:
    from simulate_v4 import (
        simulate_strategy_v4, compute_financial_stats, classification_metrics
    )

    y_true = df_sub["label"].values.astype(int)
    trades, equity = simulate_strategy_v4(
        df_sub.reset_index(drop=True),
        proba_sub,
        model_kind=model_kind,
        initial_capital=initial_capital,
        trade_size=trade_size,
        fee_rate=fee_rate,
    )
    fin = compute_financial_stats(trades, equity, freq="4h")
    cls = classification_metrics(y_true, proba_sub)
    return {"fin": fin, "cls": cls, "trades": trades, "equity": equity,
            "n_events": len(df_sub)}


# ---------------------------------------------------------------------------
# 5. Plot precio + regimenes
# ---------------------------------------------------------------------------

def plot_price_regimes(test_vol: pd.DataFrame, median_vol: float,
                       save_path: str | None = None):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                             gridspec_kw={"height_ratios": [3, 1, 1]})

    # Panel 1: precio con regimenes coloreados
    ax = axes[0]
    dates = test_vol["datetime"]
    close = test_vol["close"]

    ax.plot(dates, close, color="black", linewidth=0.8, zorder=3)

    # Colorear fondo por regimen
    in_high = False
    start_x = None
    for i, row in test_vol.iterrows():
        if row["regime"] == "HIGH" and not in_high:
            in_high = True
            start_x = row["datetime"]
        elif row["regime"] == "LOW" and in_high:
            in_high = False
            ax.axvspan(start_x, row["datetime"], alpha=0.2, color="tomato", zorder=1)
    if in_high:
        ax.axvspan(start_x, dates.iloc[-1], alpha=0.2, color="tomato", zorder=1)

    patch_h = mpatches.Patch(color="tomato",   alpha=0.4, label="Alta volatilidad")
    patch_l = mpatches.Patch(color="white",    alpha=0.8, label="Baja volatilidad")
    ax.legend(handles=[patch_h, patch_l], loc="upper left")
    ax.set_title("UNI/USDT — Test set con regimenes de volatilidad", fontsize=12)
    ax.set_ylabel("Precio (USDT)")
    ax.grid(alpha=0.3)

    # Panel 2: volatilidad rolling
    ax2 = axes[1]
    ax2.plot(dates, test_vol["vol_rolling"], color="steelblue", linewidth=1)
    ax2.axhline(median_vol, color="crimson", linestyle="--", linewidth=1,
                label=f"Mediana={median_vol:.4f}")
    ax2.fill_between(dates, test_vol["vol_rolling"], median_vol,
                     where=test_vol["vol_rolling"] >= median_vol,
                     alpha=0.3, color="tomato")
    ax2.set_ylabel("Vol rolling 4d")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # Panel 3: retornos acumulados
    ax3 = axes[2]
    cum_ret = (close / close.iloc[0] - 1) * 100
    ax3.plot(dates, cum_ret, color="darkgreen", linewidth=1)
    ax3.axhline(0, color="gray", linewidth=0.5)
    ax3.set_ylabel("Retorno acum. (%)")
    ax3.set_xlabel("Fecha")
    ax3.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[plot] Guardado: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    # Reconstruir test set
    train, test = build_test_set()

    # SFS features (fijas — las que encontro el pipeline)
    FEATURES_M1 = [
        "return_skew_72h", "unique_senders", "log_return_roll_std72",
        "macd_histogram", "fee_apr_proxy", "price_divergence",
        "rsi_14", "body_ratio", "stoch_d", "amihud_illiquidity", "bvc_imbalance",
    ]
    FEATURES_M2 = [
        "log_return_lag4", "bvc_imbalance", "net_flow_proxy",
        "macd_histogram", "donchian_pos", "fee_apr_proxy", "volumeUSD_roll_std72",
    ]

    # Entrenar modelos
    pipelines = train_models(train, test, FEATURES_M1, FEATURES_M2)

    # Asignar regimenes de volatilidad al test
    test_vol, median_vol = assign_vol_regime(test)

    # Plot precio + regimenes
    plot_price_regimes(test_vol, median_vol,
                       save_path=str(BASE_DIR / "vol_regime_price.png"))

    # Calcular probas sobre todo el test para cada modelo
    side_test = test_vol["ema_signal"].values if "ema_signal" in test_vol.columns else None
    probas = {}
    for model_type, pipe in pipelines.items():
        probas[model_type] = pipe.predict_proba(test_vol, side=side_test)

    # Separar en HIGH y LOW
    mask_high = test_vol["regime"] == "HIGH"
    mask_low  = test_vol["regime"] == "LOW"

    print("\n" + "=" * 70)
    print("  RESULTADOS POR REGIMEN DE VOLATILIDAD")
    print("=" * 70)

    header = (f"  {'Modelo':<10} {'Regimen':<6} {'N':>5}  {'Net P&L':>10}  "
              f"{'Sharpe':>7}  {'MaxDD%':>7}  {'Trades':>7}  {'Acc':>6}")
    print(header)
    print("  " + "-" * 66)

    for model_type in ["lgbm", "xgb"]:
        proba_full = probas[model_type]
        for label, mask in [("HIGH", mask_high), ("LOW", mask_low)]:
            df_sub    = test_vol[mask].reset_index(drop=True)
            proba_sub = proba_full[mask.values]

            if len(df_sub) == 0:
                continue

            res = simulate_subset(df_sub, proba_sub, model_kind=model_type)
            fin = res["fin"]
            cls = res["cls"]
            pnl    = fin.get("net_pnl_usd", 0)
            sharpe = fin.get("sharpe_ratio", 0)
            dd     = fin.get("max_dd_pct", 0)
            tr     = fin.get("total_trades", 0)
            acc    = cls.get("accuracy", float("nan"))
            n      = res["n_events"]

            print(f"  {model_type.upper():<10} {label:<6} {n:>5}  "
                  f"${pnl:>+9,.2f}  {sharpe:>7.3f}  {dd:>6.2f}%  "
                  f"{tr:>7}  {acc:>6.3f}")

        print("  " + "-" * 66)

    # Buy & Hold por regimen
    from simulate_v4 import simulate_buy_hold, compute_financial_stats
    print(f"\n  {'BH':<10} {'HIGH':<6} {mask_high.sum():>5}  ", end="")
    try:
        t, e = simulate_buy_hold(test_vol[mask_high].reset_index(drop=True))
        fin  = compute_financial_stats(t, e, freq="4h")
        print(f"${fin.get('net_pnl_usd',0):>+9,.2f}  {fin.get('sharpe_ratio',0):>7.3f}  "
              f"{fin.get('max_dd_pct',0):>6.2f}%")
    except Exception as ex:
        print(f"ERROR: {ex}")

    print(f"  {'BH':<10} {'LOW':<6}  {mask_low.sum():>5}  ", end="")
    try:
        t, e = simulate_buy_hold(test_vol[mask_low].reset_index(drop=True))
        fin  = compute_financial_stats(t, e, freq="4h")
        print(f"${fin.get('net_pnl_usd',0):>+9,.2f}  {fin.get('sharpe_ratio',0):>7.3f}  "
              f"{fin.get('max_dd_pct',0):>6.2f}%")
    except Exception as ex:
        print(f"ERROR: {ex}")

    print(f"\n  Plot guardado en: vol_regime_price.png")


if __name__ == "__main__":
    main()
