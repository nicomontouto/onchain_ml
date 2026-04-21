"""
simulate_v3.py -- Motor de simulacion LONG/SHORT/FLAT con capital real en USD.

V3: Tres estados de posicion (LONG, SHORT, FLAT)
    Capital inicial $10,000 | Trade size fijo $500
    Fee 0.1% por entrada + 0.1% por salida = 0.2% round trip
    Umbrales dinamicos (Fear & Greed) o fijos segun variante

Variantes:
  A -- XGBoost umbral fijo (0.60 long / 0.40 short)
  B -- XGBoost umbral dinamico segun Fear & Greed index
  C -- LSTM umbral fijo asimetrico (0.58 long / 0.42 short)
"""

import numpy as np
import pandas as pd

INITIAL_CAPITAL = 10_000.0
TRADE_SIZE = 500.0
FEE_RATE = 0.001          # 0.1% por half-turn
ANNUALIZE_MAP = {
    "4h": 252 * 6,
    "daily": 252,
}


# ---------------------------------------------------------------------------
# Umbrales por variante / Fear & Greed
# ---------------------------------------------------------------------------

def get_fg_thresholds(fg_value: float) -> tuple:
    """
    Retorna (thresh_long, thresh_short) segun el valor del Fear & Greed Index.
    Logica: en fear el mercado cae facil -> menos conviccion para SHORT,
            mas para LONG.  En greed al reves.
    """
    if fg_value <= 25:    # Extreme Fear
        return 0.72, 0.32
    elif fg_value <= 45:  # Fear
        return 0.65, 0.38
    elif fg_value <= 55:  # Neutral
        return 0.60, 0.40
    elif fg_value <= 75:  # Greed
        return 0.55, 0.45
    else:                 # Extreme Greed
        return 0.52, 0.48


def fg_regime_label(fg_value: float) -> str:
    if fg_value <= 25:   return "Extreme Fear"
    elif fg_value <= 45: return "Fear"
    elif fg_value <= 55: return "Neutral"
    elif fg_value <= 75: return "Greed"
    else:                return "Extreme Greed"


def apply_thresholds(proba: float, thresh_long: float, thresh_short: float) -> str:
    """Convierte probabilidad en senal LONG / SHORT / FLAT."""
    if np.isnan(proba):
        return "FLAT"
    if proba > thresh_long:
        return "LONG"
    if proba < thresh_short:
        return "SHORT"
    return "FLAT"


def get_thresholds_for_variant(variant: str, fg_value: float) -> tuple:
    if variant == "A":
        return 0.60, 0.40
    elif variant == "B":
        return get_fg_thresholds(fg_value)
    elif variant == "C":
        return 0.58, 0.42
    return 0.60, 0.40


# ---------------------------------------------------------------------------
# Simulacion principal
# ---------------------------------------------------------------------------

def simulate_strategy_v3(
    df_test: pd.DataFrame,
    y_proba: np.ndarray,
    variant: str = "A",
    initial_capital: float = INITIAL_CAPITAL,
    trade_size: float = TRADE_SIZE,
    fee_rate: float = FEE_RATE,
) -> tuple:
    """
    Simula estrategia LONG/SHORT/FLAT con capital real en USD.

    Contabilidad:
      - equity_cash: dinero liquido (fuera del trade abierto)
      - Al abrir posicion: equity_cash -= trade_size + fee_entry
      - Valor del trade abierto = trade_size * (close / entry) para LONG
                                = trade_size * (2 - close / entry) para SHORT
      - equity_total = equity_cash + trade_value (si abierta) else equity_cash
      - Al cerrar: equity_cash += trade_size + pnl_gross - fee_exit

    Retorna:
      trades_df  -- log por trade (entry/exit/pnl/fees/capital)
      equity_df  -- curva de equity por periodo
    """
    df = df_test.copy().reset_index(drop=True)
    n = min(len(df), len(y_proba))
    date_col = "datetime" if "datetime" in df.columns else "date"

    equity_cash = float(initial_capital)
    position = None       # None | "LONG" | "SHORT"
    entry_price = None
    entry_time = None
    entry_fee_paid = 0.0

    trades = []
    equity_records = []

    def _close_position(close: float, timestamp):
        nonlocal equity_cash, position, entry_price, entry_time, entry_fee_paid
        if position == "LONG":
            pnl_gross = trade_size * (close - entry_price) / entry_price
        else:  # SHORT
            pnl_gross = trade_size * (entry_price - close) / entry_price
        fee_exit = trade_size * fee_rate
        pnl_net = pnl_gross - entry_fee_paid - fee_exit
        equity_cash += trade_size + pnl_gross - fee_exit
        trades.append({
            "entry_time":    entry_time,
            "exit_time":     timestamp,
            "direction":     position,
            "entry_price":   entry_price,
            "exit_price":    close,
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
        proba = float(y_proba[i])
        timestamp = row[date_col]
        fg_val = float(row["fear_greed_value"]) if "fear_greed_value" in row and not pd.isna(row["fear_greed_value"]) else 50.0

        thresh_l, thresh_s = get_thresholds_for_variant(variant, fg_val)
        new_signal = apply_thresholds(proba, thresh_l, thresh_s)

        # Cerrar posicion si la senal cambia
        if position is not None and new_signal != position:
            _close_position(close, timestamp)

        # Abrir nueva posicion
        if new_signal != "FLAT" and position is None:
            fee_entry = trade_size * fee_rate
            equity_cash -= trade_size + fee_entry
            entry_price = close
            entry_time = timestamp
            entry_fee_paid = fee_entry
            position = new_signal

        # Valor actual del trade abierto
        if position == "LONG":
            trade_value = trade_size * (close / entry_price)
        elif position == "SHORT":
            trade_value = trade_size * (2.0 - close / entry_price)
        else:
            trade_value = 0.0

        total_equity = equity_cash + trade_value if position else equity_cash

        equity_records.append({
            date_col:              timestamp,
            "close":               close,
            "proba":               proba,
            "signal":              new_signal,
            "thresh_long":         thresh_l,
            "thresh_short":        thresh_s,
            "equity_total":        round(total_equity, 4),
            "fear_greed_value":    fg_val,
        })

    # Cerrar posicion abierta al final del periodo
    if position is not None:
        last = df.iloc[n - 1]
        _close_position(float(last["close"]), last[date_col])
        # Actualizar ultimo registro de equity
        equity_records[-1]["equity_total"] = round(equity_cash, 4)
        equity_records[-1]["signal"] = "FLAT"

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_records)

    return trades_df, equity_df


# ---------------------------------------------------------------------------
# Buy & Hold benchmark
# ---------------------------------------------------------------------------

def simulate_buy_hold(
    df_test: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    trade_size: float = TRADE_SIZE,
    fee_rate: float = FEE_RATE,
) -> tuple:
    """
    Compra $500 de UNI al inicio y mantiene hasta el final.
    El resto ($9,500) queda en USD sin exposicion.
    """
    df = df_test.copy().reset_index(drop=True)
    date_col = "datetime" if "datetime" in df.columns else "date"

    entry_price = float(df["close"].iloc[0])
    fee_entry = trade_size * fee_rate
    fee_exit = trade_size * fee_rate
    equity_cash = initial_capital - trade_size - fee_entry

    equity_records = []
    for i, row in df.iterrows():
        close = float(row["close"])
        trade_value = trade_size * (close / entry_price)
        total_equity = equity_cash + trade_value
        equity_records.append({
            date_col:        row[date_col],
            "close":         close,
            "equity_total":  round(total_equity, 4),
        })

    # Descontar fee de salida al final
    exit_price = float(df["close"].iloc[-1])
    pnl_gross = trade_size * (exit_price - entry_price) / entry_price
    pnl_net = pnl_gross - fee_entry - fee_exit
    capital_final = initial_capital + pnl_net

    trades = [{
        "entry_time":    df[date_col].iloc[0],
        "exit_time":     df[date_col].iloc[-1],
        "direction":     "LONG",
        "entry_price":   entry_price,
        "exit_price":    exit_price,
        "pnl_gross":     round(pnl_gross, 4),
        "fee_entry":     round(fee_entry, 4),
        "fee_exit":      round(fee_exit, 4),
        "fee_total":     round(fee_entry + fee_exit, 4),
        "pnl_net":       round(pnl_net, 4),
        "capital_after": round(capital_final, 4),
    }]

    # Ajustar equity final con fee de salida
    equity_records[-1]["equity_total"] = round(
        equity_cash + trade_size * (exit_price / entry_price) - fee_exit, 4
    )

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_records)
    return trades_df, equity_df


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def compute_stats(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
    freq: str = "daily",
) -> dict:
    """
    Calcula todas las metricas financieras en USD para una corrida.
    """
    n_periods = len(equity_df) if equity_df is not None and not equity_df.empty else 0
    if trades_df.empty:
        return {
            "capital_initial":  initial_capital,
            "capital_final":    initial_capital,
            "net_pnl_usd":      0.0,
            "net_pnl_pct":      0.0,
            "total_trades":     0,
            "trades_long":      0,
            "trades_short":     0,
            "periods_in_flat":  n_periods,
            "pct_flat":         100.0,
            "winning_trades":   0,
            "losing_trades":    0,
            "win_rate":         0.0,
            "gross_profit_usd": 0.0,
            "total_fees_usd":   0.0,
            "net_profit_usd":   0.0,
            "best_trade_usd":   0.0,
            "worst_trade_usd":  0.0,
            "max_win_streak":   0,
            "max_loss_streak":  0,
            "sharpe_ratio":     0.0,
            "max_dd_pct":       0.0,
            "max_dd_usd":       0.0,
            "days_test":        n_periods,
            "freq":             freq,
        }

    capital_final = float(equity_df["equity_total"].iloc[-1])
    net_pnl = capital_final - initial_capital

    pnl_net_series = trades_df["pnl_net"]
    pnl_gross_series = trades_df["pnl_gross"]
    fee_series = trades_df["fee_total"]

    gross_profit = float(pnl_gross_series[pnl_gross_series > 0].sum())
    gross_loss = float(pnl_gross_series[pnl_gross_series < 0].sum())
    total_fees = float(fee_series.sum())

    winning = int((pnl_net_series > 0).sum())
    losing = int((pnl_net_series < 0).sum())
    n_trades = len(trades_df)
    win_rate = winning / n_trades if n_trades > 0 else 0.0

    trades_long = int((trades_df["direction"] == "LONG").sum())
    trades_short = int((trades_df["direction"] == "SHORT").sum())

    best_trade = float(pnl_net_series.max()) if n_trades > 0 else 0.0
    worst_trade = float(pnl_net_series.min()) if n_trades > 0 else 0.0

    # Rachas
    def max_streak(mask):
        max_s = cur = 0
        for v in mask:
            cur = cur + 1 if v else 0
            max_s = max(max_s, cur)
        return max_s

    win_mask = (pnl_net_series > 0).tolist()
    loss_mask = (pnl_net_series <= 0).tolist()
    max_win_streak = max_streak(win_mask)
    max_loss_streak = max_streak(loss_mask)

    # Sharpe de la curva de equity
    eq = equity_df["equity_total"]
    eq_returns = eq.pct_change().dropna()
    ann_factor = ANNUALIZE_MAP.get(freq, 252)
    if len(eq_returns) > 1 and eq_returns.std() > 0:
        sharpe = float(eq_returns.mean() / eq_returns.std() * np.sqrt(ann_factor))
    else:
        sharpe = 0.0

    # Max drawdown
    running_max = eq.cummax()
    drawdown_pct = (eq - running_max) / running_max.replace(0, np.nan)
    drawdown_usd = eq - running_max
    max_dd_pct = float(drawdown_pct.min())
    max_dd_usd = float(drawdown_usd.min())

    # Periodos en FLAT
    if "signal" in equity_df.columns:
        n_flat = int((equity_df["signal"] == "FLAT").sum())
        n_total = len(equity_df)
        pct_flat = n_flat / n_total if n_total > 0 else 0.0
    else:
        n_flat = 0
        n_total = len(equity_df)
        pct_flat = 0.0

    # Dias en test
    date_col = "datetime" if "datetime" in equity_df.columns else "date"
    try:
        start = pd.to_datetime(equity_df[date_col].iloc[0])
        end = pd.to_datetime(equity_df[date_col].iloc[-1])
        days_test = (end - start).days
    except Exception:
        days_test = n_total if freq == "daily" else n_total * (4 if freq == "4h" else 1) // 24

    return {
        "capital_initial":    initial_capital,
        "capital_final":      round(capital_final, 2),
        "net_pnl_usd":        round(net_pnl, 2),
        "net_pnl_pct":        round(net_pnl / initial_capital * 100, 2),
        "total_trades":       n_trades,
        "trades_long":        trades_long,
        "trades_short":       trades_short,
        "periods_in_flat":    n_flat,
        "pct_flat":           round(pct_flat * 100, 1),
        "winning_trades":     winning,
        "losing_trades":      losing,
        "win_rate":           round(win_rate * 100, 1),
        "gross_profit_usd":   round(gross_profit, 2),
        "total_fees_usd":     round(total_fees, 2),
        "net_profit_usd":     round(net_pnl, 2),
        "best_trade_usd":     round(best_trade, 2),
        "worst_trade_usd":    round(worst_trade, 2),
        "max_win_streak":     max_win_streak,
        "max_loss_streak":    max_loss_streak,
        "sharpe_ratio":       round(sharpe, 3),
        "max_dd_pct":         round(max_dd_pct * 100, 2),
        "max_dd_usd":         round(max_dd_usd, 2),
        "days_test":          days_test,
        "freq":               freq,
    }
