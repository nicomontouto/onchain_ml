"""
simulate_v4.py -- Capa de simulacion multiclase para el pipeline v4.

Probabilidades (n_samples, 3) en orden (P(-1), P(0), P(+1)).
  XGB / LGBM  : umbral dinamico Fear & Greed (variante B de v3, adaptado)
  MLP         : umbral fijo 0.40 (Softmax reparte entre 3 clases)

Capital 10_000 | trade size 500 | fee 0.1% por lado.

Tambien expone metricas de clasificacion multiclase
(accuracy, F1 por clase, matriz de confusion, ROC-AUC OvR).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)

INITIAL_CAPITAL = 10_000.0
TRADE_SIZE = 500.0
FEE_RATE = 0.001

ANNUALIZE_MAP = {
    "4h":    252 * 6,
    "daily": 252,
}

CLASS_ORDER = (-1, 0, 1)   # corresponde a las columnas de proba (0,1,2)


# ---------------------------------------------------------------------------
# Umbrales
# ---------------------------------------------------------------------------

def get_fg_thresholds(fg_value: float) -> tuple[float, float]:
    if fg_value <= 25:
        return 0.72, 0.32
    if fg_value <= 45:
        return 0.65, 0.38
    if fg_value <= 55:
        return 0.60, 0.40
    if fg_value <= 75:
        return 0.55, 0.45
    return 0.52, 0.48


def signal_from_proba_fg(p_minus: float, p_zero: float, p_plus: float,
                         fg_value: float) -> str:
    """Para XGB/LGBM con umbrales dinamicos F&G."""
    t_long, t_short = get_fg_thresholds(fg_value)
    if p_plus > t_long:
        return "LONG"
    if p_minus > t_short:
        return "SHORT"
    return "FLAT"


def signal_from_proba_fixed(p_minus: float, p_zero: float, p_plus: float,
                            thr: float = 0.40) -> str:
    """Para MLP con umbral fijo 0.40."""
    if p_plus > thr:
        return "LONG"
    if p_minus > thr:
        return "SHORT"
    return "FLAT"


# ---------------------------------------------------------------------------
# Metricas de clasificacion
# ---------------------------------------------------------------------------

def classification_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """
    y_true en labels originales (-1, 0, +1).
    y_proba en columnas ordenadas (P(-1), P(0), P(+1)).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred_idx = np.argmax(y_proba, axis=1)
    y_pred = np.array([CLASS_ORDER[i] for i in y_pred_idx])

    acc = accuracy_score(y_true, y_pred)
    f1_per = f1_score(y_true, y_pred, labels=list(CLASS_ORDER),
                      average=None, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, labels=list(CLASS_ORDER),
                        average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(CLASS_ORDER))

    # ROC-AUC OvR (solo si hay al menos 2 clases en y_true)
    try:
        # one-hot de y_true en orden CLASS_ORDER
        y_bin = np.zeros((len(y_true), 3), dtype=int)
        for i, v in enumerate(y_true):
            y_bin[i, CLASS_ORDER.index(int(v))] = 1
        # Descartar clases sin samples para evitar warning de sklearn
        present = y_bin.sum(axis=0) > 0
        if present.sum() >= 2:
            auc_ovr = roc_auc_score(y_bin[:, present], y_proba[:, present],
                                    average="macro", multi_class="ovr")
        else:
            auc_ovr = float("nan")
    except Exception:
        auc_ovr = float("nan")

    return {
        "accuracy":     float(acc),
        "f1_minus1":    float(f1_per[0]),
        "f1_zero":      float(f1_per[1]),
        "f1_plus1":     float(f1_per[2]),
        "f1_macro":     float(f1_macro),
        "roc_auc_ovr":  float(auc_ovr),
        "confusion":    cm,
        "labels_order": list(CLASS_ORDER),
        "report":       classification_report(
                            y_true, y_pred,
                            labels=list(CLASS_ORDER),
                            target_names=["-1", "0", "+1"],
                            zero_division=0, output_dict=True,
                        ),
    }


# ---------------------------------------------------------------------------
# Simulacion de trading (LONG / SHORT / FLAT)
# ---------------------------------------------------------------------------

def simulate_strategy_v4(
    df_test: pd.DataFrame,
    y_proba: np.ndarray,
    model_kind: str,
    initial_capital: float = INITIAL_CAPITAL,
    trade_size: float = TRADE_SIZE,
    fee_rate: float = FEE_RATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    model_kind: 'xgb' | 'lgbm' | 'mlp'
      xgb/lgbm -> umbrales F&G
      mlp      -> umbral fijo 0.40

    df_test debe tener columnas: close, fear_greed_value (opcional) y
    datetime o date.
    """
    df = df_test.reset_index(drop=True).copy()
    n = min(len(df), len(y_proba))
    date_col = "datetime" if "datetime" in df.columns else "date"

    equity_cash = float(initial_capital)
    position: str | None = None
    entry_price = None
    entry_time = None
    entry_fee_paid = 0.0

    trades = []
    equity_records = []

    def _close(close_px: float, ts):
        nonlocal equity_cash, position, entry_price, entry_time, entry_fee_paid
        if position == "LONG":
            pnl_gross = trade_size * (close_px - entry_price) / entry_price
        else:
            pnl_gross = trade_size * (entry_price - close_px) / entry_price
        fee_exit = trade_size * fee_rate
        pnl_net = pnl_gross - entry_fee_paid - fee_exit
        equity_cash += trade_size + pnl_gross - fee_exit
        trades.append({
            "entry_time":    entry_time,
            "exit_time":     ts,
            "direction":     position,
            "entry_price":   entry_price,
            "exit_price":    close_px,
            "pnl_gross":     round(pnl_gross, 4),
            "fee_entry":     round(entry_fee_paid, 4),
            "fee_exit":      round(fee_exit, 4),
            "fee_total":     round(entry_fee_paid + fee_exit, 4),
            "pnl_net":       round(pnl_net, 4),
            "capital_after": round(equity_cash, 4),
        })
        position = None
        entry_price = None
        entry_time = None
        entry_fee_paid = 0.0

    for i in range(n):
        row = df.iloc[i]
        close = float(row["close"])
        ts = row[date_col]
        p_minus, p_zero, p_plus = map(float, y_proba[i])

        if model_kind in ("xgb", "lgbm"):
            fg_val = float(row["fear_greed_value"]) if (
                "fear_greed_value" in row and not pd.isna(row["fear_greed_value"])
            ) else 50.0
            new_signal = signal_from_proba_fg(p_minus, p_zero, p_plus, fg_val)
            t_long, t_short = get_fg_thresholds(fg_val)
        else:
            new_signal = signal_from_proba_fixed(p_minus, p_zero, p_plus, 0.40)
            t_long, t_short = 0.40, 0.40
            fg_val = float(row["fear_greed_value"]) if (
                "fear_greed_value" in row and not pd.isna(row["fear_greed_value"])
            ) else 50.0

        if position is not None and new_signal != position:
            _close(close, ts)

        if new_signal != "FLAT" and position is None:
            fee_entry = trade_size * fee_rate
            equity_cash -= trade_size + fee_entry
            entry_price = close
            entry_time = ts
            entry_fee_paid = fee_entry
            position = new_signal

        if position == "LONG":
            trade_value = trade_size * (close / entry_price)
        elif position == "SHORT":
            trade_value = trade_size * (2.0 - close / entry_price)
        else:
            trade_value = 0.0

        total_equity = equity_cash + trade_value if position else equity_cash

        equity_records.append({
            date_col:           ts,
            "close":            close,
            "proba_minus1":     p_minus,
            "proba_zero":       p_zero,
            "proba_plus1":      p_plus,
            "signal":           new_signal,
            "thresh_long":      t_long,
            "thresh_short":     t_short,
            "equity_total":     round(total_equity, 4),
            "fear_greed_value": fg_val,
        })

    if position is not None:
        last = df.iloc[n - 1]
        _close(float(last["close"]), last[date_col])
        equity_records[-1]["equity_total"] = round(equity_cash, 4)
        equity_records[-1]["signal"] = "FLAT"

    return pd.DataFrame(trades), pd.DataFrame(equity_records)


# ---------------------------------------------------------------------------
# Buy & Hold
# ---------------------------------------------------------------------------

def simulate_buy_hold(
    df_test: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    fee_rate: float = FEE_RATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Buy & Hold con capital completo ($10,000) expuesto desde el inicio.
    Compra initial_capital en UNI al primer precio, vende al ultimo.
    """
    df = df_test.reset_index(drop=True).copy()
    date_col = "datetime" if "datetime" in df.columns else "date"

    entry_price = float(df["close"].iloc[0])
    exit_price  = float(df["close"].iloc[-1])
    fee_entry   = initial_capital * fee_rate
    fee_exit    = initial_capital * fee_rate
    invested    = initial_capital - fee_entry

    records = []
    for _, row in df.iterrows():
        close = float(row["close"])
        current_value = invested * (close / entry_price)
        records.append({
            date_col:       row[date_col],
            "close":        close,
            "equity_total": round(current_value, 4),
        })

    final_value = invested * (exit_price / entry_price) - fee_exit
    records[-1]["equity_total"] = round(final_value, 4)
    pnl_net = final_value - initial_capital

    trades = [{
        "entry_time":    df[date_col].iloc[0],
        "exit_time":     df[date_col].iloc[-1],
        "direction":     "LONG",
        "entry_price":   entry_price,
        "exit_price":    exit_price,
        "pnl_gross":     round(invested * (exit_price - entry_price) / entry_price, 4),
        "fee_entry":     round(fee_entry, 4),
        "fee_exit":      round(fee_exit, 4),
        "fee_total":     round(fee_entry + fee_exit, 4),
        "pnl_net":       round(pnl_net, 4),
        "capital_after": round(initial_capital + pnl_net, 4),
    }]
    return pd.DataFrame(trades), pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Metricas financieras
# ---------------------------------------------------------------------------

def compute_financial_stats(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    freq: str,
    initial_capital: float = INITIAL_CAPITAL,
) -> dict:
    n_periods = len(equity_df) if equity_df is not None else 0
    if trades_df is None or trades_df.empty:
        return {
            "capital_initial":   initial_capital,
            "capital_final":     initial_capital,
            "net_pnl_usd":       0.0,
            "net_pnl_pct":       0.0,
            "total_trades":      0,
            "trades_long":       0,
            "trades_short":      0,
            "winning_trades":    0,
            "losing_trades":     0,
            "win_rate":          0.0,
            "gross_profit_usd":  0.0,
            "total_fees_usd":    0.0,
            "best_trade_usd":    0.0,
            "worst_trade_usd":   0.0,
            "periods_in_flat":   n_periods,
            "pct_flat":          100.0,
            "sharpe_ratio":      0.0,
            "max_dd_pct":        0.0,
            "max_dd_usd":        0.0,
            "freq":              freq,
        }

    cap_final = float(equity_df["equity_total"].iloc[-1])
    net = cap_final - initial_capital

    pnl_net = trades_df["pnl_net"]
    pnl_gross = trades_df["pnl_gross"]
    fees = trades_df["fee_total"]

    winning = int((pnl_net > 0).sum())
    losing = int((pnl_net < 0).sum())
    n = len(trades_df)

    eq = equity_df["equity_total"]
    eq_ret = eq.pct_change().dropna()
    ann = ANNUALIZE_MAP.get(freq, 252)
    sharpe = (float(eq_ret.mean() / eq_ret.std() * np.sqrt(ann))
              if len(eq_ret) > 1 and eq_ret.std() > 0 else 0.0)

    run_max = eq.cummax()
    dd_pct = (eq - run_max) / run_max.replace(0, np.nan)
    dd_usd = eq - run_max

    if "signal" in equity_df.columns:
        n_flat = int((equity_df["signal"] == "FLAT").sum())
    else:
        n_flat = 0
    pct_flat = 100.0 * n_flat / max(len(equity_df), 1)

    return {
        "capital_initial":  initial_capital,
        "capital_final":    round(cap_final, 2),
        "net_pnl_usd":      round(net, 2),
        "net_pnl_pct":      round(100 * net / initial_capital, 2),
        "total_trades":     n,
        "trades_long":      int((trades_df["direction"] == "LONG").sum()),
        "trades_short":     int((trades_df["direction"] == "SHORT").sum()),
        "winning_trades":   winning,
        "losing_trades":    losing,
        "win_rate":         round(100 * winning / max(n, 1), 1),
        "gross_profit_usd": round(float(pnl_gross[pnl_gross > 0].sum()), 2),
        "total_fees_usd":   round(float(fees.sum()), 2),
        "best_trade_usd":   round(float(pnl_net.max()), 2),
        "worst_trade_usd":  round(float(pnl_net.min()), 2),
        "periods_in_flat":  n_flat,
        "pct_flat":         round(pct_flat, 1),
        "sharpe_ratio":     round(sharpe, 3),
        "max_dd_pct":       round(float(dd_pct.min()) * 100, 2),
        "max_dd_usd":       round(float(dd_usd.min()), 2),
        "freq":             freq,
    }
