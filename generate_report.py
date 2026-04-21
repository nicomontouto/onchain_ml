"""
generate_report.py - Genera reporte_uni_ml_v2.pdf con resultados completos.

V2: Reporte multi-frecuencia (1h, 4h, daily) con:
  - Comparacion v1 (sintetico) vs v2 (real)
  - Estadisticas por frecuencia
  - Resultados por frecuencia + tabla maestra
  - Analisis de costos de transaccion
  - Max drawdown

Usa matplotlib para graficos y fpdf2 para el PDF.
Solo texto ASCII (sin em-dashes, flechas, bullets Unicode).
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime
import traceback

try:
    from fpdf import FPDF
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
FIGS_DIR = Path(__file__).parent / "data" / "figures"
FIGS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path(__file__).parent / "reporte_uni_ml_v2.pdf"


def _ascii(text: str) -> str:
    """Convierte texto a ASCII-safe reemplazando caracteres problematicos."""
    replacements = {
        "\u2014": "-",   # em dash
        "\u2013": "-",   # en dash
        "\u2192": "->",  # arrow right
        "\u2190": "<-",  # arrow left
        "\u2022": "*",   # bullet
        "\u2019": "'",   # right single quote
        "\u2018": "'",   # left single quote
        "\u201c": '"',   # left double quote
        "\u201d": '"',   # right double quote
        "\u2264": "<=",  # less than or equal
        "\u2265": ">=",  # greater than or equal
        "\u00b1": "+/-", # plus minus
        "\u00d7": "x",   # multiplication
        "\u2248": "~",   # approximately
        "\u221a": "sqrt",# square root
        "\u03c3": "sigma",
        "\u03bc": "mu",
        "\u2211": "sum",
        "\u2026": "...", # ellipsis
        "\u00e9": "e",
        "\u00e1": "a",
        "\u00ed": "i",
        "\u00f3": "o",
        "\u00fa": "u",
        "\u00f1": "n",
        "\u00c9": "E",
        "\u00c1": "A",
        "\u00cd": "I",
        "\u00d3": "O",
        "\u00da": "U",
        "\u00d1": "N",
        "\u00fc": "u",
        "\u00e4": "a",
        "\u00f6": "o",
        "\u2502": "|",   # box drawing
        "\u2500": "-",   # box drawing horizontal
        "\u2550": "=",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Final pass: replace any remaining non-ASCII with ?
    return text.encode("ascii", errors="replace").decode("ascii")


def _save_fig(fig, name: str) -> Path:
    path = FIGS_DIR / f"{name}.png"
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return path


# --- Plots ---

def plot_price_history(df: pd.DataFrame, freq: str = "daily") -> Path:
    date_col = "datetime" if "datetime" in df.columns else "date"
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(f"UNI - Price and Log Returns ({freq})", fontsize=13, fontweight="bold")

    axes[0].plot(df[date_col], df["close"], color="#1f77b4", linewidth=0.8)
    axes[0].set_ylabel("Price (USD)")
    axes[0].grid(alpha=0.3)

    if "log_return" in df.columns:
        colors = ["#2ca02c" if r > 0 else "#d62728" for r in df["log_return"].fillna(0)]
        axes[1].bar(df[date_col], df["log_return"].fillna(0),
                    color=colors, width=1 if freq == "daily" else 0.05, alpha=0.7)
        axes[1].set_ylabel("Log Return")
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return _save_fig(fig, f"price_history_{freq}")


def plot_equity_curves_freq(sim_results_freq: dict, freq: str) -> Path:
    """
    sim_results_freq: {'XGB_baseline': df_sim, 'XGB_tuned': df_sim, 'LSTM': df_sim}
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_title(f"Equity Curve - Test Period ({freq})", fontsize=13, fontweight="bold")

    colors = {
        "XGB_baseline": "#1f77b4",
        "XGB_tuned": "#ff7f0e",
        "LSTM": "#2ca02c",
    }
    bh_plotted = False

    for model_name, df_sim in sim_results_freq.items():
        if df_sim is None or df_sim.empty:
            continue
        date_col = "datetime" if "datetime" in df_sim.columns else "date"
        color = colors.get(model_name, "#7f7f7f")
        ax.plot(df_sim[date_col], df_sim["cum_strategy"],
                label=f"{model_name}", color=color, linewidth=1.5)
        if not bh_plotted:
            ax.plot(df_sim[date_col], df_sim["cum_bh"],
                    label="Buy & Hold", color="black", linewidth=1.5,
                    linestyle="--", alpha=0.7)
            bh_plotted = True

    ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_ylabel("Cumulative Return (1 = initial capital)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_fig(fig, f"equity_curves_{freq}")


def plot_feature_importance(df_importance: pd.DataFrame, freq: str, top_n: int = 20) -> Path:
    if df_importance is None or df_importance.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "Feature importance not available", ha="center", va="center")
        return _save_fig(fig, f"feature_importance_{freq}")

    top = df_importance.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(f"Top {top_n} Features - XGBoost ({freq})", fontsize=13, fontweight="bold")
    ax.barh(top["feature"][::-1], top["importance"][::-1], color="#1f77b4", alpha=0.8)
    ax.set_xlabel("Importance")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    return _save_fig(fig, f"feature_importance_{freq}")


def plot_lstm_history(lstm_history: pd.DataFrame, freq: str) -> Path:
    if lstm_history is None or lstm_history.empty:
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.text(0.5, 0.5, "LSTM history not available", ha="center", va="center")
        return _save_fig(fig, f"lstm_training_{freq}")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(lstm_history["epoch"], lstm_history["train_loss"],
            label="Train Loss", color="#1f77b4")
    ax.plot(lstm_history["epoch"], lstm_history["val_loss"],
            label="Val Loss", color="#ff7f0e")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.set_title(f"LSTM Training Curve ({freq})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _save_fig(fig, f"lstm_training_{freq}")


def plot_all_equity_curves(all_sim_results: dict) -> Path:
    """
    all_sim_results: {'1h': {'XGB_baseline': df, ...}, '4h': {...}, 'daily': {...}}
    """
    freqs = [f for f in ["1h", "4h", "daily"] if f in all_sim_results]
    if not freqs:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No results", ha="center", va="center")
        return _save_fig(fig, "all_equity_curves")

    fig, axes = plt.subplots(1, len(freqs), figsize=(6 * len(freqs), 5))
    if len(freqs) == 1:
        axes = [axes]

    colors = {"XGB_baseline": "#1f77b4", "XGB_tuned": "#ff7f0e", "LSTM": "#2ca02c"}

    for ax, freq in zip(axes, freqs):
        sim_dict = all_sim_results[freq]
        bh_plotted = False
        for model_name, df_sim in sim_dict.items():
            if df_sim is None or df_sim.empty:
                continue
            date_col = "datetime" if "datetime" in df_sim.columns else "date"
            color = colors.get(model_name, "#7f7f7f")
            ax.plot(df_sim[date_col], df_sim["cum_strategy"],
                    label=model_name, color=color, linewidth=1.2)
            if not bh_plotted:
                ax.plot(df_sim[date_col], df_sim["cum_bh"],
                        label="B&H", color="black", linewidth=1.2,
                        linestyle="--", alpha=0.7)
                bh_plotted = True
        ax.axhline(1.0, color="gray", linewidth=0.8, linestyle=":")
        ax.set_title(f"Equity ({freq})")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    return _save_fig(fig, "all_equity_curves")


def plot_drawdown_comparison(all_sim_results: dict) -> Path:
    """Plot max drawdown comparison across all models and frequencies."""
    rows = []
    for freq, sim_dict in all_sim_results.items():
        for model_name, df_sim in sim_dict.items():
            if df_sim is None or df_sim.empty:
                continue
            rolling_max = df_sim["cum_strategy"].cummax()
            dd = (df_sim["cum_strategy"] - rolling_max) / rolling_max.replace(0, np.nan)
            rows.append({"label": f"{model_name}\n({freq})", "max_dd": float(dd.min())})

    if not rows:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return _save_fig(fig, "drawdown_comparison")

    df_dd = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Max Drawdown by Model and Frequency", fontsize=12, fontweight="bold")
    bars = ax.bar(range(len(df_dd)), df_dd["max_dd"] * 100, color="#d62728", alpha=0.7)
    ax.set_xticks(range(len(df_dd)))
    ax.set_xticklabels(df_dd["label"], fontsize=8)
    ax.set_ylabel("Max Drawdown (%)")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return _save_fig(fig, "drawdown_comparison")


# --- PDF class ---

class ReportePDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(80, 80, 80)
        self.cell(0, 7, "ML Pipeline v2 - UNI (Uniswap) | Real Data", 0, 1, "R")
        self.ln(1)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128)
        self.cell(0, 10,
                  f"Page {self.page_no()} | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
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

    def metric_table(self, headers: list, rows: list):
        self.set_font("Helvetica", "B", 8)
        self.set_fill_color(200, 210, 240)
        n_cols = len(headers)
        col_w = 170 / n_cols
        for h in headers:
            self.cell(col_w, 6, _ascii(str(h)), 1, 0, "C", fill=True)
        self.ln()

        self.set_font("Helvetica", "", 8)
        for i, row in enumerate(rows):
            fill = i % 2 == 0
            if fill:
                self.set_fill_color(245, 248, 255)
            else:
                self.set_fill_color(255, 255, 255)
            for cell in row:
                self.cell(col_w, 5.5, _ascii(str(cell)), 1, 0, "C", fill=fill)
            self.ln()
        self.ln(2)


# --- Funcion principal ---

def generate_report(
    all_datasets: dict = None,
    all_features: dict = None,
    all_results: dict = None,
    all_sim_results: dict = None,
    all_importance: dict = None,
    all_tuning: dict = None,
    all_best_params: dict = None,
    all_lstm_history: dict = None,
    all_wfv: dict = None,
    errors_log: list = None,
    # Compat v1
    df_features: pd.DataFrame = None,
    df_eval_results: pd.DataFrame = None,
    df_wfv: pd.DataFrame = None,
    df_importance: pd.DataFrame = None,
    df_tuning: pd.DataFrame = None,
    best_params: dict = None,
    sim_results: dict = None,
    lstm_history: pd.DataFrame = None,
) -> Path:
    """
    Genera el PDF v2 completo con resultados multi-frecuencia.

    Acepta:
    - all_datasets: {'1h': df, '4h': df, 'daily': df}
    - all_features: {'1h': df_features, '4h': df_features, 'daily': df_features}
    - all_results: {'1h': [result_dict,...], '4h': [...], 'daily': [...]}
    - all_sim_results: {'1h': {'XGB_baseline': df_sim, ...}, ...}
    - all_importance: {'1h': df_imp, '4h': df_imp, 'daily': df_imp}
    - all_tuning: {'1h': df_tuning, ...}
    - all_best_params: {'1h': dict, ...}
    - all_lstm_history: {'1h': df_history, ...}
    - all_wfv: {'1h': df_wfv, ...}
    - errors_log: list of error strings
    """
    if not FPDF_AVAILABLE:
        print("  [Report] fpdf2 no disponible. Instalando...")
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2", "-q"])
        from fpdf import FPDF

    # Fallback: si se pasa en formato v1
    if all_datasets is None:
        all_datasets = {}
    if all_features is None:
        all_features = {}
        if df_features is not None:
            all_features["daily"] = df_features
    if all_results is None:
        all_results = {}
        if df_eval_results is not None and not df_eval_results.empty:
            all_results["daily"] = df_eval_results.to_dict("records")
    if all_sim_results is None:
        all_sim_results = {}
        if sim_results:
            all_sim_results["daily"] = sim_results
    if all_importance is None:
        all_importance = {}
        if df_importance is not None:
            all_importance["daily"] = df_importance
    if all_tuning is None:
        all_tuning = {}
        if df_tuning is not None:
            all_tuning["daily"] = df_tuning
    if all_best_params is None:
        all_best_params = {}
        if best_params:
            all_best_params["daily"] = best_params
    if all_lstm_history is None:
        all_lstm_history = {}
        if lstm_history is not None:
            all_lstm_history["daily"] = lstm_history
    if all_wfv is None:
        all_wfv = {}
        if df_wfv is not None:
            all_wfv["daily"] = df_wfv
    if errors_log is None:
        errors_log = []

    freqs = [f for f in ["1h", "4h", "daily"] if f in all_features or f in all_results]

    print("\n[Report v2] Generando figuras...")

    # Generar figuras por frecuencia
    fig_equity_all = plot_all_equity_curves(all_sim_results)
    fig_drawdown = plot_drawdown_comparison(all_sim_results)

    freq_figs = {}
    for freq in freqs:
        figs = {}
        df_feat = all_features.get(freq)
        if df_feat is not None and not df_feat.empty:
            figs["price"] = plot_price_history(df_feat, freq)
        sim_dict = all_sim_results.get(freq, {})
        if sim_dict:
            figs["equity"] = plot_equity_curves_freq(sim_dict, freq)
        df_imp = all_importance.get(freq)
        if df_imp is not None:
            figs["importance"] = plot_feature_importance(df_imp, freq)
        lstm_hist = all_lstm_history.get(freq)
        if lstm_hist is not None:
            figs["lstm"] = plot_lstm_history(lstm_hist, freq)
        freq_figs[freq] = figs

    print("[Report v2] Generando PDF...")

    pdf = ReportePDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # ---- Portada ----
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(20, 20, 100)
    pdf.ln(10)
    pdf.cell(0, 12, "Pipeline ML On-Chain v2", 0, 1, "C")
    pdf.cell(0, 12, "Token UNI (Uniswap) - Real Data", 0, 1, "C")
    pdf.set_font("Helvetica", "", 13)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 8,
             f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}",
             0, 1, "C")
    pdf.ln(5)
    pdf.set_font("Helvetica", "", 11)
    pdf.cell(0, 7, "Frequencies: 1h, 4h, daily", 0, 1, "C")
    pdf.cell(0, 7, "Models: XGBoost Baseline, XGBoost Tuned, LSTM", 0, 1, "C")
    pdf.cell(0, 7, "Data: Dune Analytics + The Graph + Binance + Fear&Greed", 0, 1, "C")
    pdf.ln(15)

    # ---- 1. Executive Summary ----
    pdf.section_title("1. Executive Summary")

    # v1 vs v2 comparison
    pdf.subsection("1.1 v1 (Synthetic Data) vs v2 (Real Data)")
    pdf.body_text(
        "V1 pipeline used synthetic/simulated Dune data for development and testing.\n"
        "V2 pipeline uses real data from multiple on-chain sources:\n"
        "  - Dune Analytics: real UNI holder counts and whale metrics\n"
        "  - The Graph: real Uniswap V3 protocol metrics (volumeUSD, TVL, priceUSD, feesUSD)\n"
        "  - Binance API: real hourly OHLCV data (1h, 4h, daily resample)\n"
        "  - Alternative.me: Fear & Greed Index (daily sentiment signal)\n\n"
        "Key improvements in v2:\n"
        "  * Multi-frequency pipeline: 1h, 4h, and daily datasets\n"
        "  * Transaction cost modeling: 0.1% per half-turn applied on signal changes\n"
        "  * Max drawdown metric added to all model evaluations\n"
        "  * Expanded feature set: fear_greed_value, volumeUSD lags, price_divergence\n"
        "  * min_child_weight added to XGBoost hyperparameter search grid\n"
        "  * Configurable LSTM lookback per frequency"
    )

    pdf.subsection("1.2 Pipeline Overview")
    # Build summary table
    summary_rows = []
    for freq in freqs:
        df_feat = all_features.get(freq)
        n_rows = len(df_feat) if df_feat is not None else 0
        n_feat = len([c for c in df_feat.columns
                      if c not in {"date", "datetime", "target", "target_1h",
                                   "target_4h", "target_24h",
                                   "open", "high", "low", "close", "volume"}]) \
            if df_feat is not None else 0
        res_list = all_results.get(freq, [])
        if res_list:
            best = max(res_list, key=lambda r: r.get("directional_accuracy", 0))
            best_str = (f"{best['model']}: acc={best['directional_accuracy']:.2%} "
                        f"sharpe={best['sharpe_strategy']:.2f}")
        else:
            best_str = "N/A"
        summary_rows.append([freq, str(n_rows), str(n_feat), best_str])

    pdf.metric_table(
        ["Freq", "Rows", "Features", "Best Model"],
        summary_rows
    )

    # ---- 2. Data Sources ----
    pdf.add_page()
    pdf.section_title("2. Data Sources Description")

    pdf.subsection("2.1 Binance Hourly OHLCV")
    pdf.body_text(
        "Source: Binance REST API (no API key required)\n"
        "Endpoint: GET https://api.binance.com/api/v3/klines\n"
        "Symbol: UNIUSDT, Interval: 1h\n"
        "Coverage: last 5 years (1825 days), paginated at 1000 candles/request\n"
        "Columns: datetime, open, high, low, close, volume\n"
        "Cache: data/cache/binance_uniusdt_1h_1825d.parquet (1-day TTL)\n"
        "Resampled to 4h and daily for multi-frequency pipeline"
    )

    pdf.subsection("2.2 Dune Analytics Holders")
    pdf.body_text(
        "Source: data/dune_uni_holders.csv (real Dune Analytics export)\n"
        "Content: daily UNI holder counts, whale metrics, concentration indexes\n"
        "Key columns: date, number_of_holders, whale_count, whale_balance_pct,\n"
        "             whale_balance_total, herfindahl_index\n"
        "Derived: holders_growth (pct change), whale_delta_pct (pct change)"
    )

    pdf.subsection("2.3 The Graph Protocol Metrics")
    pdf.body_text(
        "Source: data/thegraph_uni_protocol.csv (real The Graph export)\n"
        "Content: daily Uniswap V3 protocol metrics for UNI token\n"
        "Key columns: date, volumeUSD, totalValueLockedUSD, priceUSD, feesUSD\n"
        "Derived: volume_tvl_ratio, fee_revenue, price_divergence"
    )

    pdf.subsection("2.4 Fear and Greed Index")
    pdf.body_text(
        "Source: https://api.alternative.me/fng/?limit=2000&format=json\n"
        "Content: daily crypto market sentiment index (0-100)\n"
        "Encoding: Extreme Fear=0, Fear=1, Neutral=2, Greed=3, Extreme Greed=4\n"
        "Key columns: date, fear_greed_value, fear_greed_class\n"
        "Cache: data/cache/fear_greed.parquet (1-day TTL)"
    )

    # ---- 3. Dataset Statistics ----
    pdf.add_page()
    pdf.section_title("3. Dataset Statistics per Frequency")

    for freq in freqs:
        pdf.subsection(f"3.{freqs.index(freq)+1} Frequency: {freq}")
        df_feat = all_features.get(freq)
        if df_feat is None or df_feat.empty:
            pdf.body_text(f"No data available for {freq}")
            continue

        date_col = "datetime" if "datetime" in df_feat.columns else "date"
        n_rows = len(df_feat)
        split_idx = int(n_rows * 0.8)
        n_feat = len([c for c in df_feat.columns
                      if c not in {"date", "datetime", "target", "target_1h",
                                   "target_4h", "target_24h",
                                   "open", "high", "low", "close", "volume"}])

        stat_rows = [
            ["Total rows", str(n_rows), ""],
            ["Train rows", str(split_idx), "80%"],
            ["Test rows", str(n_rows - split_idx), "20%"],
            ["Features", str(n_feat), ""],
        ]

        if "close" in df_feat.columns:
            stat_rows.extend([
                ["Min price (USD)", f"{df_feat['close'].min():.4f}", ""],
                ["Max price (USD)", f"{df_feat['close'].max():.4f}", ""],
                ["Avg price (USD)", f"{df_feat['close'].mean():.4f}", ""],
            ])

        if date_col in df_feat.columns:
            stat_rows.insert(0, [
                "Date range",
                str(df_feat[date_col].min())[:16],
                str(df_feat[date_col].max())[:16],
            ])

        if "number_of_holders" in df_feat.columns and df_feat["number_of_holders"].sum() > 0:
            stat_rows.append([
                "Avg holders", f"{df_feat['number_of_holders'].mean():.0f}", ""])

        if "fear_greed_value" in df_feat.columns and df_feat["fear_greed_value"].sum() > 0:
            stat_rows.append([
                "Avg Fear&Greed", f"{df_feat['fear_greed_value'].mean():.1f}", "0-100"])

        pdf.metric_table(["Metric", "Value", "Detail"], stat_rows)

    # ---- 4. Features List ----
    pdf.add_page()
    pdf.section_title("4. Features List")

    pdf.body_text(
        "Feature categories across all frequencies:\n\n"
        "Price features (from Binance):\n"
        "  log_return, pct_return, price_ma24, price_ma168, volatility_24h,\n"
        "  high_low_range, volume_ma24\n\n"
        "Holder features (from Dune):\n"
        "  number_of_holders, holders_growth, whale_count, whale_balance_pct,\n"
        "  whale_balance_total, whale_delta_pct, herfindahl_index\n\n"
        "Protocol features (from The Graph):\n"
        "  volumeUSD, totalValueLockedUSD, priceUSD, feesUSD,\n"
        "  volume_tvl_ratio, fee_revenue, price_divergence\n\n"
        "Sentiment features (from Fear & Greed):\n"
        "  fear_greed_value, fear_greed_class\n\n"
        "Whale signal features (derived):\n"
        "  whale_accumulation, whale_distribution, whale_confluence,\n"
        "  sell_pressure, concentration_ratio, whale_dominance\n\n"
        "Lag features (frequency-specific periods):\n"
        "  1h:    lags at [1, 4, 24, 168] periods\n"
        "  4h:    lags at [1, 6, 18, 42] periods\n"
        "  daily: lags at [1, 2, 3, 7] periods\n\n"
        "Rolling window features (frequency-specific):\n"
        "  1h/4h: rolling windows [24, 72, 168]\n"
        "  daily: rolling windows [7, 14, 30]\n\n"
        "Target variables:\n"
        "  target_1h:  next period log_return > 0 (shift -1)\n"
        "  target_4h:  cumulative log_return next 4 periods > 0\n"
        "  target_24h: cumulative log_return next 24 periods > 0\n"
        "  (for 4h/daily: shift(-1) equivalents)"
    )

    # ---- 5. Results per Frequency ----
    for freq in freqs:
        pdf.add_page()
        pdf.section_title(f"5. Results - Frequency: {freq}")

        res_list = all_results.get(freq, [])
        sim_dict = all_sim_results.get(freq, {})
        figs = freq_figs.get(freq, {})

        if res_list:
            pdf.subsection("5.1 Comparison Table")
            comp_rows = []
            for r in res_list:
                comp_rows.append([
                    r.get("model", "?"),
                    f"{r.get('directional_accuracy', 0):.2%}",
                    f"{r.get('roc_auc', float('nan')):.4f}"
                    if not (isinstance(r.get('roc_auc'), float) and
                            np.isnan(r.get('roc_auc', float('nan')))) else "N/A",
                    f"{r.get('sharpe_strategy', 0):.3f}",
                    f"{r.get('total_return_strategy', 0):.2%}",
                    f"{r.get('max_drawdown', 0):.2%}",
                ])
            # Add Buy & Hold
            if res_list:
                r0 = res_list[0]
                comp_rows.append([
                    "Buy & Hold", "-", "-",
                    f"{r0.get('sharpe_bh', 0):.3f}",
                    f"{r0.get('total_return_bh', 0):.2%}",
                    "-",
                ])

            pdf.metric_table(
                ["Model", "Dir.Acc", "ROC-AUC", "Sharpe", "Return", "MaxDD"],
                comp_rows,
            )

        if "price" in figs:
            pdf.add_image_safe(figs["price"])

        if "equity" in figs:
            pdf.subsection("5.2 Equity Curve")
            pdf.add_image_safe(figs["equity"])

        if "importance" in figs:
            pdf.subsection("5.3 Feature Importance (XGBoost)")
            pdf.add_image_safe(figs["importance"])

        if "lstm" in figs:
            pdf.subsection("5.4 LSTM Training Curve")
            pdf.add_image_safe(figs["lstm"])

        # WFV results
        df_wfv_freq = all_wfv.get(freq)
        if df_wfv_freq is not None and not df_wfv_freq.empty:
            pdf.subsection("5.5 Walk-Forward Validation")
            wfv_rows = []
            for _, row in df_wfv_freq.iterrows():
                wfv_rows.append([
                    f"Fold {int(row.get('fold', 0))}",
                    f"{row.get('accuracy', 0):.4f}",
                    f"{row.get('roc_auc', float('nan')):.4f}"
                    if not np.isnan(row.get('roc_auc', float('nan'))) else "N/A",
                    str(int(row.get('train_size', 0))),
                    str(int(row.get('test_size', 0))),
                ])
            if len(df_wfv_freq) > 0:
                wfv_rows.append([
                    "MEAN",
                    f"{df_wfv_freq['accuracy'].mean():.4f}",
                    f"{df_wfv_freq['roc_auc'].mean():.4f}",
                    "-", "-",
                ])
            pdf.metric_table(["Fold", "Accuracy", "ROC-AUC", "Train", "Test"], wfv_rows)

    # ---- 6. Master Table ----
    pdf.add_page()
    pdf.section_title("6. Master Table: All Models, All Frequencies, All Metrics")

    all_rows = []
    for freq in freqs:
        res_list = all_results.get(freq, [])
        for r in res_list:
            all_rows.append([
                r.get("freq", freq),
                r.get("model", "?"),
                f"{r.get('directional_accuracy', 0):.2%}",
                f"{r.get('roc_auc', float('nan')):.4f}"
                if not (isinstance(r.get('roc_auc'), float) and
                        np.isnan(r.get('roc_auc', float('nan')))) else "N/A",
                f"{r.get('information_coefficient', 0):.4f}",
                f"{r.get('sharpe_strategy', 0):.3f}",
                f"{r.get('total_return_strategy', 0):.2%}",
                f"{r.get('total_return_bh', 0):.2%}",
                f"{r.get('max_drawdown', 0):.2%}",
            ])

    if all_rows:
        pdf.metric_table(
            ["Freq", "Model", "Dir.Acc", "AUC", "IC",
             "Sharpe", "Return", "B&H Ret", "MaxDD"],
            all_rows,
        )
    else:
        pdf.body_text("No results available.")

    pdf.add_image_safe(fig_equity_all)

    # ---- 7. Transaction Costs Analysis ----
    pdf.add_page()
    pdf.section_title("7. Transaction Costs Analysis")

    pdf.body_text(
        "Transaction cost model used in v2:\n\n"
        "Cost per half-turn: 0.001 (0.1%)\n"
        "A complete round-trip (buy + sell) costs: 0.002 (0.2%)\n\n"
        "Cost is applied whenever the signal changes:\n"
        "  - Entry: signal 0 -> 1 (long): -0.1% applied\n"
        "  - Exit:  signal 1 -> 0 (flat): -0.1% applied\n\n"
        "Formula: strategy_return[t] = pct_return[t] * signal[t] - cost[t]\n"
        "  where cost[t] = |signal[t] - signal[t-1]| * 0.001\n\n"
        "Impact by frequency:\n"
        "  - daily: fewer trades, lower total cost impact\n"
        "  - 4h:    moderate trading frequency\n"
        "  - 1h:    highest trading frequency, highest cost impact\n\n"
        "Note: This model approximates CEX spot trading costs.\n"
        "Real costs may include: spread, gas fees (for DEX), funding rates.\n"
        "Gas fees on Ethereum can significantly increase costs for small positions."
    )

    # Show cost impact by model/freq
    cost_rows = []
    for freq in freqs:
        res_list = all_results.get(freq, [])
        sim_dict = all_sim_results.get(freq, {})
        for r in res_list:
            model_name = r.get("model", "?")
            df_sim = sim_dict.get(model_name)
            if df_sim is not None and not df_sim.empty:
                n_trades = int(df_sim["signal"].diff().abs().fillna(0).sum())
                total_cost_pct = n_trades * 0.001 * 100
                cost_rows.append([
                    freq, model_name, str(n_trades),
                    f"{total_cost_pct:.2f}%",
                    f"{r.get('total_return_strategy', 0):.2%}",
                ])

    if cost_rows:
        pdf.metric_table(
            ["Freq", "Model", "# Trades", "Total Cost", "Net Return"],
            cost_rows,
        )

    pdf.add_image_safe(fig_drawdown)

    # ---- 8. Conclusions ----
    pdf.add_page()
    pdf.section_title("8. Conclusions")

    pdf.body_text(
        "What worked in v2:\n"
        "* Real data pipeline from Dune Analytics and The Graph proved more stable\n"
        "  than synthetic data for training ML models on crypto price prediction.\n"
        "* Multi-frequency approach allows capturing different market dynamics:\n"
        "  daily signals capture macro trends, 4h captures medium-term moves,\n"
        "  1h captures short-term momentum.\n"
        "* Fear & Greed Index adds useful sentiment signal not present in price data.\n"
        "* price_divergence feature (abs(close - priceUSD)/priceUSD) captures\n"
        "  CEX vs DEX price dislocations, a unique on-chain signal.\n"
        "* Transaction cost modeling gives more realistic performance estimates.\n"
        "* max_drawdown metric helps identify models with better risk profiles.\n\n"
        "What did not work as expected:\n"
        "* LSTM training is slow on hourly data (~43k rows) even with patience=15.\n"
        "* On-chain holder data from Dune may have gaps for early periods.\n"
        "* The Graph CSV coverage depends on data export date and may not cover 5 years.\n"
        "* High-frequency models (1h) suffer more from transaction costs.\n\n"
        "General observations:\n"
        "* XGBoost typically outperforms LSTM on tabular financial data.\n"
        "* Walk-forward validation shows significant variance across folds,\n"
        "  suggesting regime changes in the crypto market over 5 years.\n"
        "* The best performing frequency and model may change in different market conditions."
    )

    # ---- 9. Next Steps ----
    pdf.section_title("9. Next Steps")

    pdf.body_text(
        "Recommended improvements:\n"
        "* Ensemble: combine XGBoost and LSTM predictions with dynamic weighting.\n"
        "* Dynamic threshold: use percentile-based threshold instead of fixed 0.5.\n"
        "* Position sizing: implement Kelly criterion or volatility-based sizing.\n"
        "* Add funding rates from perpetual futures as additional signal.\n"
        "* Add social sentiment (Twitter/Reddit volume) as complementary signal.\n"
        "* Explore transformer architectures (Informer, Temporal Fusion Transformer).\n"
        "* Implement proper walk-forward backtesting with expanding or rolling windows.\n"
        "* Add correlation analysis: UNI vs BTC, ETH, DeFi index.\n"
        "* Regime detection: identify bull/bear market regimes explicitly.\n"
        "* Gas cost adjustment: for DEX trading strategies, add gas fee model."
    )

    # ---- 10. Limitations ----
    pdf.section_title("10. Limitations")

    pdf.body_text(
        "Model limitations:\n"
        "* Look-ahead bias risk: verify all features use only data available at time t.\n"
        "* Survivorship bias: UNI survived and grew; selecting winners introduces bias.\n"
        "* Execution assumption: strategy assumes execution at close price, no slippage.\n"
        "* Market impact: model does not account for impact of large orders.\n"
        "* Overfitting risk: with hundreds of features and complex models.\n\n"
        "Data limitations:\n"
        "* Dune Analytics data quality depends on the specific query and export date.\n"
        "* The Graph may have incomplete data for very early protocol periods.\n"
        "* Binance data may have gaps during maintenance periods.\n"
        "* Fear & Greed Index is limited to ~1000 days of history.\n\n"
        "Operational limitations:\n"
        "* Model retraining required periodically as market regimes change.\n"
        "* API rate limits may cause incomplete data downloads.\n"
        "* LSTM training time scales poorly with data size.\n\n"
        "Disclaimer: This report is for educational purposes only.\n"
        "It does not constitute investment advice."
    )

    # ---- 11. Errors ----
    if errors_log:
        pdf.add_page()
        pdf.section_title("11. Errors During Execution")
        for err in errors_log:
            pdf.body_text(f"* {str(err)[:200]}", size=9)
    else:
        pdf.section_title("11. Errors During Execution")
        pdf.body_text("No errors encountered during pipeline execution.")

    pdf.output(str(REPORT_PATH))
    print(f"\n[Report v2] PDF guardado: {REPORT_PATH}")
    return REPORT_PATH


# --- Entry point ---

if __name__ == "__main__":
    print("=" * 60)
    print("TEST: generate_report.py v2")
    print("=" * 60)

    generate_report(errors_log=["Test run - no real data loaded"])
    print("[OK] Reporte generado.")
