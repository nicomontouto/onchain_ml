"""
generate_report_v3.py -- Genera reporte_uni_ml_v3.pdf

V3: Decision layer con LONG/SHORT/FLAT, capital real USD,
    umbrales dinamicos Fear & Greed, 6 variantes + Buy & Hold.

Solo texto ASCII. Usa fpdf2 + matplotlib.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import traceback

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

FIGS_DIR = Path(__file__).parent / "data" / "figures_v3"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path(__file__).parent / "reporte_uni_ml_v3.pdf"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ascii(text: str) -> str:
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2192": "->", "\u2022": "*",
        "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2264": "<=",
        "\u2265": ">=", "\u00b1": "+/-", "\u221a": "sqrt",
        "\u00e9": "e", "\u00e1": "a", "\u00ed": "i", "\u00f3": "o",
        "\u00fa": "u", "\u00f1": "n", "\u00c9": "E", "\u00c1": "A",
        "\u00cd": "I", "\u00d3": "O", "\u00da": "U", "\u00d1": "N",
        "\u00fc": "u", "\u00e4": "a", "\u00f6": "o", "\u2502": "|",
        "\u2500": "-", "\u2550": "=",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("ascii", errors="replace").decode("ascii")


def _save_fig(fig, name: str) -> Path:
    path = FIGS_DIR / f"{name}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_equity_curve(equity_df: pd.DataFrame, trades_df: pd.DataFrame,
                      bh_equity_df: pd.DataFrame, label: str) -> Path:
    """
    Curva de equity en USD con marcadores de entrada/salida.
    Triangulos verdes = entradas LONG
    Triangulos rojos  = entradas SHORT
    Circulos grises   = cierres (FLAT / salidas)
    """
    date_col = "datetime" if "datetime" in equity_df.columns else "date"
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.set_title(f"Equity Curve (USD) -- {label}", fontsize=12, fontweight="bold")

    # Curva principal
    ax.plot(equity_df[date_col], equity_df["equity_total"],
            color="#1f77b4", linewidth=1.5, label=label, zorder=3)

    # Buy & Hold
    if bh_equity_df is not None and not bh_equity_df.empty:
        bh_dc = "datetime" if "datetime" in bh_equity_df.columns else "date"
        ax.plot(bh_equity_df[bh_dc], bh_equity_df["equity_total"],
                color="black", linewidth=1.2, linestyle="--", alpha=0.7,
                label="Buy & Hold", zorder=2)

    # Marcadores de trades
    if trades_df is not None and not trades_df.empty:
        for _, trade in trades_df.iterrows():
            et = trade["entry_time"]
            xt = trade["exit_time"]
            # Valor de equity en esos momentos
            eq_at = equity_df.set_index(date_col)["equity_total"]
            try:
                # Match by closest timestamp
                eq_entry = eq_at.iloc[eq_at.index.get_indexer([et], method="nearest")[0]]
                eq_exit  = eq_at.iloc[eq_at.index.get_indexer([xt], method="nearest")[0]]
            except Exception:
                continue
            if trade["direction"] == "LONG":
                ax.scatter(et, eq_entry, marker="^", color="#2ca02c", s=60, zorder=5)
            else:
                ax.scatter(et, eq_entry, marker="v", color="#d62728", s=60, zorder=5)
            ax.scatter(xt, eq_exit, marker="o", color="#7f7f7f", s=30, zorder=4, alpha=0.7)

    ax.axhline(10_000, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
    ax.set_ylabel("Capital (USD)")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.grid(alpha=0.3)

    # Leyenda manual
    handles = [
        mpatches.Patch(color="#1f77b4", label=label),
        mpatches.Patch(color="black", label="Buy & Hold"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="#2ca02c",
                   markersize=8, label="LONG entry"),
        plt.Line2D([0], [0], marker="v", color="w", markerfacecolor="#d62728",
                   markersize=8, label="SHORT entry"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#7f7f7f",
                   markersize=6, label="Exit"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    fig.tight_layout()
    return _save_fig(fig, f"equity_v3_{label.replace(' ', '_').replace('/', '_')}")


def plot_fees_vs_profit(trades_df: pd.DataFrame, equity_df: pd.DataFrame,
                        label: str) -> Path:
    """Fees acumuladas vs profit bruto acumulado."""
    date_col = "datetime" if "datetime" in equity_df.columns else "date"
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_title(f"Fees vs Gross Profit Acumulados -- {label}", fontsize=11, fontweight="bold")

    if trades_df is None or trades_df.empty:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center")
        return _save_fig(fig, f"fees_profit_v3_{label.replace(' ', '_')}")

    cum_fees  = trades_df["fee_total"].cumsum()
    cum_gross = trades_df["pnl_gross"].cumsum()
    trade_dates = trades_df["exit_time"].reset_index(drop=True)

    ax.plot(trade_dates, cum_gross, color="#2ca02c", linewidth=1.5,
            label="Gross Profit Acum.")
    ax.plot(trade_dates, cum_fees,  color="#d62728", linewidth=1.5,
            label="Fees Acum.")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("USD")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.1f}"))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_fig(fig, f"fees_profit_v3_{label.replace(' ', '_').replace('/', '_')}")


def plot_pnl_distribution(trades_df: pd.DataFrame, label: str) -> Path:
    """Histograma de PnL neto por trade, separado por LONG/SHORT."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(f"Distribucion PnL por Trade -- {label}", fontsize=11, fontweight="bold")

    if trades_df is None or trades_df.empty:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center")
        return _save_fig(fig, f"pnl_dist_v3_{label.replace(' ', '_')}")

    long_pnl  = trades_df.loc[trades_df["direction"] == "LONG",  "pnl_net"].dropna()
    short_pnl = trades_df.loc[trades_df["direction"] == "SHORT", "pnl_net"].dropna()

    all_vals = trades_df["pnl_net"].dropna()
    if len(all_vals) == 0:
        ax.text(0.5, 0.5, "No trades", ha="center", va="center")
        return _save_fig(fig, f"pnl_dist_v3_{label.replace(' ', '_')}")

    bins = np.linspace(all_vals.min() - 1, all_vals.max() + 1, 30)
    if len(long_pnl) > 0:
        ax.hist(long_pnl, bins=bins, alpha=0.6, color="#1f77b4", label=f"LONG ({len(long_pnl)})")
    if len(short_pnl) > 0:
        ax.hist(short_pnl, bins=bins, alpha=0.6, color="#ff7f0e", label=f"SHORT ({len(short_pnl)})")

    ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xlabel("PnL neto (USD)")
    ax.set_ylabel("Frecuencia")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_fig(fig, f"pnl_dist_v3_{label.replace(' ', '_').replace('/', '_')}")


def plot_fg_thresholds(equity_df_4h: pd.DataFrame, equity_df_daily: pd.DataFrame,
                       trades_df_4h: pd.DataFrame = None,
                       trades_df_daily: pd.DataFrame = None) -> Path:
    """
    Analisis de umbrales para variante B.
    Muestra como variaron los umbrales segun Fear & Greed.
    """
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=False)
    fig.suptitle("Analisis de Umbrales Dinamicos -- Variante B (Fear & Greed)",
                 fontsize=12, fontweight="bold")

    # Usar daily para el analisis (mas limpio)
    df = equity_df_daily if equity_df_daily is not None and not equity_df_daily.empty else equity_df_4h
    if df is None or df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return _save_fig(fig, "fg_thresholds_v3")

    date_col = "datetime" if "datetime" in df.columns else "date"

    # Plot 1: Fear & Greed value con bandas de regimenes
    ax0 = axes[0]
    ax0.set_title("Fear & Greed Index con Regimenes de Sentimiento", fontsize=10)
    dates = df[date_col]
    fg = df["fear_greed_value"]
    ax0.plot(dates, fg, color="#7f7f7f", linewidth=0.8, alpha=0.8)
    ax0.fill_between(dates, 0,  25, alpha=0.12, color="#d62728", label="Extreme Fear")
    ax0.fill_between(dates, 25, 45, alpha=0.10, color="#ff7f0e", label="Fear")
    ax0.fill_between(dates, 45, 55, alpha=0.10, color="#bcbd22", label="Neutral")
    ax0.fill_between(dates, 55, 75, alpha=0.10, color="#2ca02c", label="Greed")
    ax0.fill_between(dates, 75, 100, alpha=0.12, color="#17becf", label="Extreme Greed")
    ax0.set_ylim(0, 100)
    ax0.set_ylabel("F&G Value")
    ax0.legend(loc="upper right", fontsize=7, ncol=5)
    ax0.grid(alpha=0.2)

    # Plot 2: Umbrales largo y corto
    ax1 = axes[1]
    ax1.set_title("Umbrales Long / Short segun F&G (daily)", fontsize=10)
    if "thresh_long" in df.columns and "thresh_short" in df.columns:
        ax1.plot(dates, df["thresh_long"],  color="#2ca02c", linewidth=1.2,
                 label="Umbral LONG")
        ax1.plot(dates, df["thresh_short"], color="#d62728", linewidth=1.2,
                 label="Umbral SHORT")
        ax1.fill_between(dates, df["thresh_short"], df["thresh_long"],
                         alpha=0.1, color="#7f7f7f", label="Zona FLAT")
    ax1.set_ylim(0.25, 0.80)
    ax1.set_ylabel("Probabilidad")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.2)

    # Plot 3: Conteo de senales por regimen
    ax2 = axes[2]
    ax2.set_title("Distribucion de Senales por Regimen (daily)", fontsize=10)

    if "fear_greed_value" in df.columns and "signal" in df.columns:
        regimes = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
        cuts = [0, 25, 45, 55, 75, 100]
        x_pos = np.arange(len(regimes))
        w = 0.25
        long_counts  = []
        short_counts = []
        flat_counts  = []
        for i, (lo, hi) in enumerate(zip(cuts[:-1], cuts[1:])):
            mask = (df["fear_greed_value"] > lo) & (df["fear_greed_value"] <= hi)
            sub = df[mask]
            long_counts.append((sub["signal"] == "LONG").sum())
            short_counts.append((sub["signal"] == "SHORT").sum())
            flat_counts.append((sub["signal"] == "FLAT").sum())
        ax2.bar(x_pos - w, long_counts,  w, color="#2ca02c", alpha=0.8, label="LONG")
        ax2.bar(x_pos,     short_counts, w, color="#d62728", alpha=0.8, label="SHORT")
        ax2.bar(x_pos + w, flat_counts,  w, color="#7f7f7f", alpha=0.8, label="FLAT")
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(regimes, fontsize=8)
        ax2.set_ylabel("Periodos")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.2, axis="y")

    fig.tight_layout()
    return _save_fig(fig, "fg_thresholds_v3")


def plot_master_equity(all_results: dict) -> Path:
    """Todas las curvas de equity en un solo grafico."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    freqs = ["4h", "daily"]
    titles = ["Frecuencia 4h", "Frecuencia Daily"]

    colors = {
        "A": "#1f77b4",
        "B": "#ff7f0e",
        "C": "#2ca02c",
        "BH": "black",
    }

    for ax, freq, title in zip(axes, freqs, titles):
        ax.set_title(title, fontsize=11, fontweight="bold")
        bh_plotted = False
        for key, data in all_results.items():
            if freq not in key and key != "BH":
                continue
            eq_df = data.get("equity_df")
            if eq_df is None or eq_df.empty:
                continue
            date_col = "datetime" if "datetime" in eq_df.columns else "date"
            variant = key.split("-")[0] if "-" in key else key
            color = colors.get(variant, "#7f7f7f")
            lw = 1.8 if variant == "BH" else 1.2
            ls = "--" if variant == "BH" else "-"
            lbl = key
            ax.plot(eq_df[date_col], eq_df["equity_total"],
                    color=color, linewidth=lw, linestyle=ls, label=lbl, alpha=0.85)
        ax.axhline(10_000, color="gray", linewidth=0.6, linestyle=":")
        ax.set_ylabel("Capital (USD)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    return _save_fig(fig, "master_equity_v3")


# ---------------------------------------------------------------------------
# PDF class
# ---------------------------------------------------------------------------

class ReporteV3PDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(80, 80, 80)
        self.cell(0, 7, "ML Pipeline v3 - UNI (Uniswap) | LONG/SHORT/FLAT | Real Capital USD", 0, 1, "R")
        self.ln(1)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10,
                  f"Pagina {self.page_no()} | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                  0, 0, "C")

    def section_title(self, text: str):
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(30, 30, 120)
        self.set_fill_color(230, 235, 250)
        self.cell(0, 9, _ascii(text), 0, 1, "L", fill=True)
        self.ln(2)
        self.set_text_color(0)

    def subsection(self, text: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(60, 60, 60)
        self.cell(0, 7, _ascii(text), 0, 1)
        self.set_text_color(0)

    def body_text(self, text: str, size: int = 10):
        self.set_font("Helvetica", "", size)
        self.set_text_color(40, 40, 40)
        self.multi_cell(0, 5.5, _ascii(text))
        self.ln(1)

    def add_image_safe(self, path, w: int = 170):
        if path and Path(str(path)).exists():
            try:
                self.image(str(path), x=None, y=None, w=w)
                self.ln(3)
            except Exception as e:
                self.body_text(f"[Image error: {e}]")

    def metric_table(self, headers: list, rows: list, col_widths: list = None):
        n_cols = len(headers)
        if col_widths is None:
            col_widths = [170 / n_cols] * n_cols
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(200, 210, 240)
        for h, w in zip(headers, col_widths):
            self.cell(w, 6, _ascii(str(h)), 1, 0, "C", fill=True)
        self.ln()
        self.set_font("Helvetica", "", 8)
        for i, row in enumerate(rows):
            fill = i % 2 == 0
            if fill:
                self.set_fill_color(245, 248, 255)
            else:
                self.set_fill_color(255, 255, 255)
            for cell, w in zip(row, col_widths):
                self.cell(w, 5.5, _ascii(str(cell)), 1, 0, "C", fill=fill)
            self.ln()
        self.ln(2)

    def pnl_box(self, stats: dict):
        """Caja de P&L en USD estilo planilla."""
        self.set_font("Courier", "", 9)
        self.set_fill_color(245, 248, 255)
        lines = [
            f" Capital inicial:        ${stats.get('capital_initial', 10000):>12,.2f}",
            f" Capital final:          ${stats.get('capital_final', 0):>12,.2f}",
            f" Profit/Loss neto:       ${stats.get('net_pnl_usd', 0):>+12,.2f}  ({stats.get('net_pnl_pct', 0):+.2f}%)",
            " " + "-" * 45,
            f" Total trades:           {stats.get('total_trades', 0):>13}",
            f" Trades LONG:            {stats.get('trades_long', 0):>13}",
            f" Trades SHORT:           {stats.get('trades_short', 0):>13}",
            f" Periodos en FLAT:       {stats.get('periods_in_flat', 0):>8}  ({stats.get('pct_flat', 0):.1f}%)",
            " " + "-" * 45,
            f" Trades ganadores:       {stats.get('winning_trades', 0):>8}  ({stats.get('win_rate', 0):.1f}%)",
            f" Trades perdedores:      {stats.get('losing_trades', 0):>13}",
            " " + "-" * 45,
            f" Gross profit:           ${stats.get('gross_profit_usd', 0):>12,.2f}",
            f" Total fees pagadas:     ${stats.get('total_fees_usd', 0):>12,.2f}",
            f" Net profit:             ${stats.get('net_profit_usd', 0):>+12,.2f}",
            " " + "-" * 45,
            f" Mejor trade:            ${stats.get('best_trade_usd', 0):>+12,.2f}",
            f" Peor trade:             ${stats.get('worst_trade_usd', 0):>+12,.2f}",
            f" Racha ganadora max:     {stats.get('max_win_streak', 0):>8} trades",
            f" Racha perdedora max:    {stats.get('max_loss_streak', 0):>8} trades",
            " " + "-" * 45,
            f" Sharpe ratio:           {stats.get('sharpe_ratio', 0):>13.3f}",
            f" Max drawdown:           {stats.get('max_dd_pct', 0):>8.2f}%  (${stats.get('max_dd_usd', 0):,.2f})",
            f" Dias en test:           {stats.get('days_test', 0):>13}",
        ]
        for line in lines:
            self.cell(0, 5, _ascii(line), 0, 1, "L", fill=True)
        self.ln(3)


# ---------------------------------------------------------------------------
# Funcion principal
# ---------------------------------------------------------------------------

def generate_report_v3(
    all_results: dict,
    errors_log: list = None,
) -> Path:
    """
    Genera reporte_uni_ml_v3.pdf.

    all_results: dict con keys 'A-4h', 'B-4h', 'C-4h', 'A-daily', 'B-daily', 'C-daily', 'BH'
    Cada valor: {'trades_df': df, 'equity_df': df, 'stats': dict, 'bh_equity_df': df}
    """
    if not FPDF_AVAILABLE:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2", "-q"])
        from fpdf import FPDF

    if errors_log is None:
        errors_log = []

    print("\n[Report v3] Generando figuras...")

    # Identificar la mejor combinacion por net profit
    best_key = None
    best_net = float("-inf")
    for key, data in all_results.items():
        if key == "BH":
            continue
        net = data.get("stats", {}).get("net_pnl_usd", float("-inf"))
        if net > best_net:
            best_net = net
            best_key = key

    # Figuras individuales por combinacion
    figs = {}
    for key, data in all_results.items():
        eq_df = data.get("equity_df")
        tr_df = data.get("trades_df")
        # Usar el B&H de la misma frecuencia como overlay (si existe)
        freq_of_key = key.split("-")[1] if "-" in key else None
        bh_df = data.get("bh_equity_df")
        if bh_df is None and freq_of_key:
            bh_df = all_results.get(f"BH-{freq_of_key}", {}).get("equity_df")
        if bh_df is None:
            bh_df = all_results.get("BH", {}).get("equity_df")

        figs[key] = {}
        try:
            figs[key]["equity"] = plot_equity_curve(eq_df, tr_df, bh_df, key)
        except Exception as e:
            errors_log.append(f"plot equity {key}: {e}")
        try:
            if key != "BH" and tr_df is not None:
                figs[key]["fees"] = plot_fees_vs_profit(tr_df, eq_df, key)
        except Exception as e:
            errors_log.append(f"plot fees {key}: {e}")
        try:
            if key != "BH" and tr_df is not None:
                figs[key]["dist"] = plot_pnl_distribution(tr_df, key)
        except Exception as e:
            errors_log.append(f"plot dist {key}: {e}")

    # Figura de umbrales F&G (variante B)
    try:
        b4h_eq   = all_results.get("B-4h", {}).get("equity_df")
        bday_eq  = all_results.get("B-daily", {}).get("equity_df")
        b4h_tr   = all_results.get("B-4h", {}).get("trades_df")
        bday_tr  = all_results.get("B-daily", {}).get("trades_df")
        fig_fg = plot_fg_thresholds(b4h_eq, bday_eq, b4h_tr, bday_tr)
    except Exception as e:
        fig_fg = None
        errors_log.append(f"plot fg thresholds: {e}")

    # Figura maestra
    try:
        fig_master = plot_master_equity(all_results)
    except Exception as e:
        fig_master = None
        errors_log.append(f"plot master equity: {e}")

    print("[Report v3] Construyendo PDF...")

    pdf = ReporteV3PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---- Portada ----
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(20, 20, 100)
    pdf.ln(10)
    pdf.cell(0, 12, "Pipeline ML On-Chain v3", 0, 1, "C")
    pdf.cell(0, 12, "Token UNI (Uniswap) | LONG/SHORT/FLAT", 0, 1, "C")
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8, f"Generado: {datetime.now().strftime('%d/%m/%Y %H:%M')}", 0, 1, "C")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, "Frecuencias: 4h, daily (se descarta 1h)", 0, 1, "C")
    pdf.cell(0, 7, "Variante A: XGBoost umbral fijo | B: XGBoost F&G dinamico | C: LSTM fijo", 0, 1, "C")
    pdf.cell(0, 7, "Capital: $10,000 | Trade size: $500 | Fee: 0.2% round trip", 0, 1, "C")
    pdf.ln(8)

    if best_key:
        net_str = f"${best_net:+,.2f}" if not np.isnan(best_net) else "N/A"
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(20, 100, 20)
        pdf.cell(0, 8, f"Mejor combinacion: {best_key} | Net P&L: {net_str}", 0, 1, "C")
    pdf.set_text_color(0)

    # ---- Cambios vs V2 ----
    pdf.add_page()
    pdf.section_title("1. Cambios respecto a V2")
    pdf.body_text(
        "V3 introduce cambios exclusivamente en la capa de decision y simulacion.\n"
        "Los modelos XGBoost y LSTM se reutilizan de V2 sin reentrenamiento.\n\n"
        "Cambios principales:\n"
        "  1. Se descarta la frecuencia 1h completamente.\n"
        "  2. Se agrega estado FLAT: tres estados posibles LONG / SHORT / FLAT.\n"
        "  3. Umbral dinamico basado en Fear & Greed para XGBoost (variante B).\n"
        "  4. Umbral fijo asimetrico para LSTM (variante C).\n"
        "  5. Capital inicial $10,000 USD con trade size fijo de $500.\n"
        "  6. Reporte financiero detallado en USD con tracking por trade.\n\n"
        "Logica de umbrales:\n"
        "  LONG  si proba > umbral_long\n"
        "  SHORT si proba < umbral_short\n"
        "  FLAT  si umbral_short <= proba <= umbral_long\n\n"
        "Variante A (XGB fijo):      long=0.60  short=0.40\n"
        "Variante B (XGB F&G):\n"
        "  Extreme Fear (0-25):      long=0.72  short=0.32\n"
        "  Fear (25-45):             long=0.65  short=0.38\n"
        "  Neutral (45-55):          long=0.60  short=0.40\n"
        "  Greed (55-75):            long=0.55  short=0.45\n"
        "  Extreme Greed (75-100):   long=0.52  short=0.48\n"
        "Variante C (LSTM fijo):     long=0.58  short=0.42\n\n"
        "Contabilidad USD:\n"
        "  - Capital base $10,000 en USD. Trade size $500 fijo.\n"
        "  - Fee 0.1% por entrada + 0.1% por salida = 0.2% round trip.\n"
        "  - Solo una posicion abierta a la vez.\n"
        "  - Al cambiar de LONG a SHORT: cierra LONG (fee salida) + abre SHORT (fee entrada)."
    )

    # ---- Resultados por combinacion ----
    COMBINATIONS = ["A-4h", "B-4h", "C-4h", "A-daily", "B-daily", "C-daily"]

    for combo in COMBINATIONS:
        data = all_results.get(combo)
        if not data:
            continue

        pdf.add_page()
        pdf.section_title(f"2. Resultados -- {combo}")

        stats = data.get("stats", {})

        # Caja de P&L
        pdf.subsection("2.1 Tabla de P&L en USD")
        pdf.pnl_box(stats)

        # Equity curve
        if figs.get(combo, {}).get("equity"):
            pdf.subsection("2.2 Equity Curve en USD")
            pdf.add_image_safe(figs[combo]["equity"])

        # Fees vs profit
        if figs.get(combo, {}).get("fees"):
            pdf.subsection("2.3 Fees Acumuladas vs Profit Bruto")
            pdf.add_image_safe(figs[combo]["fees"])

        # PnL distribution
        if figs.get(combo, {}).get("dist"):
            pdf.subsection("2.4 Distribucion PnL por Trade")
            pdf.add_image_safe(figs[combo]["dist"])

        # Nota si 0 trades
        if stats.get("total_trades", 0) == 0:
            pdf.body_text(
                "NOTA: Esta combinacion produjo 0 trades durante el periodo de test.\n"
                "El modelo no genero probabilidades que crucen los umbrales configurados.\n"
                "Resultado: capital permanecio en $10,000 (100% FLAT).\n"
                "Ver seccion de errores para detalles de la distribucion de probabilidades.",
                size=10
            )

        # Tabla de trades (primeros 20)
        tr_df = data.get("trades_df")
        if tr_df is not None and not tr_df.empty and len(tr_df) > 0:
            pdf.subsection(f"2.5 Log de Trades (primeros {min(20, len(tr_df))})")
            tr_rows = []
            for _, tr in tr_df.head(20).iterrows():
                et_str = str(tr["entry_time"])[:16] if pd.notna(tr["entry_time"]) else "-"
                xt_str = str(tr["exit_time"])[:16]  if pd.notna(tr["exit_time"])  else "-"
                tr_rows.append([
                    et_str, xt_str,
                    tr["direction"],
                    f"${tr['entry_price']:.4f}",
                    f"${tr['exit_price']:.4f}",
                    f"${tr['pnl_net']:+.2f}",
                    f"${tr['fee_total']:.2f}",
                    f"${tr['capital_after']:,.2f}",
                ])
            pdf.metric_table(
                ["Entrada", "Salida", "Dir", "P.entry", "P.exit", "PnL neto", "Fee", "Capital"],
                tr_rows,
                col_widths=[28, 28, 10, 18, 18, 18, 14, 26],
            )

    # ---- Buy & Hold ----
    bh_data = all_results.get("BH")
    if bh_data:
        pdf.add_page()
        pdf.section_title("3. Benchmark -- Buy & Hold")
        pdf.pnl_box(bh_data.get("stats", {}))
        if figs.get("BH", {}).get("equity"):
            pdf.add_image_safe(figs["BH"]["equity"])

    # ---- Tabla Maestra ----
    pdf.add_page()
    pdf.section_title("4. Tabla Maestra Comparativa")

    master_rows = []
    for key in COMBINATIONS + ["BH"]:
        data = all_results.get(key, {})
        stats = data.get("stats", {})
        if not stats:
            continue
        variant = key.split("-")[0] if "-" in key else key
        freq = key.split("-")[1] if "-" in key else "-"
        master_rows.append([
            variant, freq,
            str(stats.get("total_trades", "-")),
            f"{stats.get('pct_flat', 0):.1f}%",
            f"${stats.get('total_fees_usd', 0):.2f}",
            f"${stats.get('net_pnl_usd', 0):+,.2f}",
            f"{stats.get('net_pnl_pct', 0):+.2f}%",
            f"{stats.get('sharpe_ratio', 0):.3f}",
            f"{stats.get('max_dd_pct', 0):.2f}%",
        ])

    pdf.metric_table(
        ["Var", "Freq", "Trades", "%Flat", "Fees", "Net P&L", "Ret%", "Sharpe", "MaxDD"],
        master_rows,
        col_widths=[13, 14, 14, 14, 20, 25, 18, 20, 18],
    )

    # Figura maestra
    if fig_master:
        pdf.add_image_safe(fig_master)

    # ---- Analisis Fear & Greed (variante B) ----
    if fig_fg:
        pdf.add_page()
        pdf.section_title("5. Analisis de Umbrales Dinamicos -- Variante B")
        pdf.body_text(
            "La variante B ajusta los umbrales de entrada/salida segun el Fear & Greed Index:\n"
            "  - En 'Extreme Fear': exige mas conviccion para LONG (0.72) y menos para SHORT (0.32).\n"
            "    Logica: mercado cae mas facil en panico, vale mas apostar al short.\n"
            "  - En 'Extreme Greed': umbrales casi simetricos (0.52/0.48), zona FLAT muy estrecha.\n"
            "    Logica: en euforia las senales son mas confiables en ambas direcciones.\n\n"
            "Nota: el LSTM (variante C) NO usa este mecanismo porque Fear & Greed ya es\n"
            "un feature de entrenamiento del modelo."
        )
        pdf.add_image_safe(fig_fg)

        # Stats por regimen para variante B daily
        bday_eq = all_results.get("B-daily", {}).get("equity_df")
        if bday_eq is not None and "fear_greed_value" in bday_eq.columns:
            pdf.subsection("5.1 Dias por Regimen de Sentimiento (daily)")
            regimes = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
            cuts = [0, 25, 45, 55, 75, 100]
            regime_rows = []
            for lo, hi, label in zip(cuts[:-1], cuts[1:], regimes):
                mask = (bday_eq["fear_greed_value"] > lo) & (bday_eq["fear_greed_value"] <= hi)
                sub = bday_eq[mask]
                n = len(sub)
                pct = 100 * n / max(len(bday_eq), 1)
                if "signal" in sub.columns:
                    n_long  = (sub["signal"] == "LONG").sum()
                    n_short = (sub["signal"] == "SHORT").sum()
                    n_flat  = (sub["signal"] == "FLAT").sum()
                else:
                    n_long = n_short = n_flat = 0
                regime_rows.append([
                    label, str(n), f"{pct:.1f}%",
                    str(n_long), str(n_short), str(n_flat),
                ])
            pdf.metric_table(
                ["Regimen", "Dias", "%", "LONG", "SHORT", "FLAT"],
                regime_rows,
                col_widths=[42, 16, 16, 22, 22, 22],
            )

    # ---- Conclusion ----
    pdf.add_page()
    pdf.section_title("6. Conclusion")

    best_data = all_results.get(best_key, {}) if best_key else {}
    best_stats = best_data.get("stats", {})

    pdf.body_text(
        "Resumen de la corrida V3:\n\n"
        f"Mejor combinacion variante x frecuencia (por net profit en USD):\n"
        f"  -> {best_key or 'N/A'}\n"
        f"     Net P&L: ${best_stats.get('net_pnl_usd', 0):+,.2f} "
        f"({best_stats.get('net_pnl_pct', 0):+.2f}%)\n"
        f"     Sharpe:  {best_stats.get('sharpe_ratio', 0):.3f}\n"
        f"     Max DD:  {best_stats.get('max_dd_pct', 0):.2f}%\n"
        f"     Trades:  {best_stats.get('total_trades', 0)} "
        f"({best_stats.get('pct_flat', 0):.1f}% en FLAT)\n\n"
        "Observaciones generales:\n"
        "* El estado FLAT reduce el numero de trades y las fees acumuladas,\n"
        "  pero puede perder oportunidades en mercados con tendencia clara.\n"
        "* La variante B (umbrales dinamicos F&G) agrega sensibilidad al sentimiento\n"
        "  del mercado sin depender del modelo directamente.\n"
        "* La variante C (LSTM) ya incorpora Fear & Greed como feature, por lo que\n"
        "  usar umbrales fijos evita doble conteo del sentimiento.\n"
        "* El trade size fijo de $500 (5% del capital) limita la exposicion maxima\n"
        "  y evita el impacto de posiciones de gran tamano.\n\n"
        "Disclaimer: Este reporte es solo con fines educativos. No constituye\n"
        "asesoramiento de inversion. Trading de criptomonedas implica riesgo de\n"
        "perdida total del capital invertido."
    )

    # ---- Errores ----
    if errors_log:
        pdf.add_page()
        pdf.section_title("7. Errores durante la Ejecucion")
        for err in errors_log:
            pdf.body_text(f"* {str(err)[:200]}", size=9)
    else:
        pdf.section_title("7. Errores durante la Ejecucion")
        pdf.body_text("No se registraron errores durante la ejecucion.")

    pdf.output(str(REPORT_PATH))
    print(f"\n[Report v3] PDF guardado: {REPORT_PATH}")
    return REPORT_PATH
